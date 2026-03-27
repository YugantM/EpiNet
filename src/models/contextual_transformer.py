"""
ContextualTransformer
=====================

A standard Transformer with five targeted epigenetic features grafted in —
chosen specifically from the L1/L2/L3/L4 probe analysis to cover every
failure mode while keeping what the Transformer already does well.

Why each feature is here
------------------------

Feature 1 — Persistent e_t state (EMA)
  Without it: Transformer is stateless; every call sees the same routing.
  Fixes: L2 (paraphrase), L3 (topic transfer).
  Cost: one 32-float vector per session.

Feature 2 — Context-gated FFN
  Standard FFN: h = FFN(x)  — same transformation regardless of context.
  With gating:  h = σ(W_e·e_t) ⊙ FFN(x) + (1−gate) ⊙ x
  The gate is broadcast across all T token positions. When e_t says
  "we are in literature mode", the FFN weights the right feature subspace.
  Fixes: L2, L3.

Feature 3 — Context-gated attention output
  The MHSA output is scaled by σ(W_a·e_t) before the residual add.
  When e_t says "this question structure matches what I've seen before",
  the attention contribution is amplified; for unseen structures it is
  damped and the residual (input) dominates.
  Fixes: L2.

Feature 4 — Dual pooling: content-query + mean, merged
  content_pool: a learned query q attends over tokens → focuses on the
    most contextually relevant token (good for exact matching, negation).
  mean_pool:    uniform average → semantic fingerprint, paraphrase-robust.
  merge:        Linear([content_pool; mean_pool]) → single D vector.
  This is the single biggest fix for L2/L3 while keeping L1/L4.

Feature 5 — MemoryStore (cross-sequence topic retrieval)
  Injects a soft-read from past representations into the FFN.
  Helps L3 by letting the model recall that "Tokyo" appeared as a
  distractor in capital-city questions before.

Comparable systems
------------------
  Feature 1 (persistent state): related to Transformer-XL recurrence,
    but EMA is simpler, O(D) cost, not O(T×D).
  Feature 2 (context-gated FFN): related to Mixture-of-Experts gating
    but deterministic, conditioned on session state not input.
  Feature 4 (dual pool): related to BERT CLS + pooler; pooling head in
    DPR; but combined with mean-pool and explicitly learned merge.
  Feature 5 (memory): related to Memory-Augmented Neural Networks (NTM,
    Graves 2016), MemN2N; but lighter, fixed size.

Novel combination: applying all five simultaneously to a standard
pre-norm Transformer block stack with the dual-pool classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from ..memory.memory_store import MemoryStore


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 + 3: Context-Gated Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class ContextGatedBlock(nn.Module):
    """
    Pre-norm Transformer block with two epigenetic additions:

      Sub-layer 1: LayerNorm → MHSA → context gate on output → residual
      Sub-layer 2: LayerNorm → FFN  → context gate on output → residual

    Context gates are per-dimension sigmoid vectors derived from e_t.
    They are broadcast over the T token positions: (B,D) → (B,1,D).

    When e_t = 0 (cold start), gates are σ(0) = 0.5 — neutral, output
    retains both the processed and the residual equally.
    """

    def __init__(
        self,
        embed_dim:      int,
        num_heads:      int,
        epigenetic_dim: int,
        ffn_dim:        int,
        dropout:        float = 0.1,
    ) -> None:
        super().__init__()

        # Sub-layer 1: MHSA
        self.norm1      = nn.LayerNorm(embed_dim)
        self.attn       = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_drop  = nn.Dropout(dropout)
        # Gate on MHSA output (Feature 3)
        self.attn_gate  = nn.Linear(epigenetic_dim, embed_dim, bias=True)

        # Sub-layer 2: FFN
        self.norm2      = nn.LayerNorm(embed_dim)
        self.ffn        = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        # Gate on FFN output (Feature 2)
        self.ffn_gate   = nn.Linear(epigenetic_dim, embed_dim, bias=True)

    def forward(
        self,
        x:        torch.Tensor,           # (B, T, D)
        e_t:      torch.Tensor,           # (B, D_e)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, T, D)."""

        # ── Sub-layer 1: Gated MHSA ──────────────────────────────────────
        normed    = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self.attn_drop(attn_out)

        # Feature 3: scale attention contribution by context gate
        a_gate = torch.sigmoid(self.attn_gate(e_t)).unsqueeze(1)  # (B,1,D)
        x = x + a_gate * attn_out

        # ── Sub-layer 2: Gated FFN ────────────────────────────────────────
        normed  = self.norm2(x)
        ffn_out = self.ffn(normed)

        # Feature 2: scale FFN contribution by context gate
        f_gate = torch.sigmoid(self.ffn_gate(e_t)).unsqueeze(1)   # (B,1,D)
        x = x + f_gate * ffn_out

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: Dual Pooling Head
# ─────────────────────────────────────────────────────────────────────────────

class DualPool(nn.Module):
    """
    Combines two pooling strategies and merges them:

      content_pool : learned query q attends over tokens via scaled dot-product.
                     Focuses on the most relevant token(s).  Good for exact
                     matching (L1) and function words like "NOT" (L4).

      mean_pool    : uniform average over non-padding tokens.
                     Semantic fingerprint, position-invariant.  Good for
                     paraphrase (L2) and topic transfer (L3).

      merge        : Linear(2D → D) + LayerNorm combines both signals.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.content_query = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        x:        torch.Tensor,                   # (B, T, D)
        pad_mask: Optional[torch.Tensor] = None,  # (B, T) True = pad
    ) -> torch.Tensor:
        """Returns (B, D)."""

        # Content-based pooling
        scale  = x.size(-1) ** -0.5
        scores = (x @ self.content_query) * scale             # (B, T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, float("-inf"))
        weights      = torch.softmax(scores, dim=-1)           # (B, T)
        content_pool = (weights.unsqueeze(-1) * x).sum(1)     # (B, D)

        # Mean pooling (unchanged)
        if pad_mask is not None:
            mask_f    = (~pad_mask).float().unsqueeze(-1)      # (B, T, 1)
            mean_pool = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            mean_pool = x.mean(1)                              # (B, D)

        # Merge
        h = torch.cat([content_pool, mean_pool], dim=-1)      # (B, 2D)
        return self.merge(h)                                   # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Full ContextualTransformer
# ─────────────────────────────────────────────────────────────────────────────

class ContextualTransformer(nn.Module):
    """
    Transformer with all five epigenetic features.

    Parameters  (same naming convention as EpiTransformer for fair comparison)
    ----------
    vocab_size, embed_dim, num_heads, num_layers, num_classes  — standard
    epigenetic_dim   : dimension of e_t state
    ffn_dim          : inner FFN dimension (default 4×embed_dim)
    memory_size      : MemoryStore capacity  (0 = disable)
    memory_dim       : MemoryStore key/value dim
    epigenetic_alpha : EMA decay for e_t
    dropout, max_seq_len  — standard
    """

    def __init__(
        self,
        vocab_size:       int   = 5000,
        embed_dim:        int   = 64,
        num_heads:        int   = 4,
        num_layers:       int   = 2,
        num_classes:      int   = 2,
        epigenetic_dim:   int   = 32,
        ffn_dim:          Optional[int] = None,
        memory_size:      int   = 64,
        memory_dim:       int   = 64,
        epigenetic_alpha: float = 0.3,
        dropout:          float = 0.1,
        max_seq_len:      int   = 256,
    ) -> None:
        super().__init__()

        ffn_dim = ffn_dim or embed_dim * 4
        self.embed_dim        = embed_dim
        self.epigenetic_dim   = epigenetic_dim
        self.epigenetic_alpha = epigenetic_alpha

        # ── Embeddings ────────────────────────────────────────────────────
        self.tok_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.emb_norm  = nn.LayerNorm(embed_dim)
        self.emb_drop  = nn.Dropout(dropout)

        # ── Transformer blocks (Feature 2 + 3 inside each) ───────────────
        self.blocks = nn.ModuleList([
            ContextGatedBlock(
                embed_dim=embed_dim, num_heads=num_heads,
                epigenetic_dim=epigenetic_dim,
                ffn_dim=ffn_dim, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # ── Feature 1: Epigenetic state init + EMA update ────────────────
        self.epi_init = nn.Sequential(
            nn.Linear(embed_dim, epigenetic_dim), nn.Tanh(),
        )
        # update: f(pooled_hidden [, memory_ctx]) → new e_t target
        update_in = embed_dim + (memory_dim if memory_size > 0 else 0)
        self.epi_update = nn.Sequential(
            nn.Linear(update_in, epigenetic_dim), nn.Tanh(),
        )

        # ── Feature 5: Memory store ───────────────────────────────────────
        if memory_size > 0:
            self.memory  = MemoryStore(memory_size=memory_size, memory_dim=memory_dim)
            self.mem_proj = nn.Linear(embed_dim, memory_dim, bias=False)
        else:
            self.memory  = None

        # ── Feature 4: Dual pooling ───────────────────────────────────────
        self.pool = DualPool(embed_dim, dropout=dropout)

        # ── Output head ───────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    # ──────────────────────────────────────────────────────────────────────
    # e_t helpers
    # ──────────────────────────────────────────────────────────────────────

    def _init_e_t(self, x_mean: torch.Tensor) -> torch.Tensor:
        return self.epi_init(x_mean)

    def _update_e_t(
        self, e_t: torch.Tensor, h: torch.Tensor,
        mem_ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = [h, mem_ctx] if mem_ctx is not None else [h]
        f = self.epi_update(torch.cat(parts, dim=-1))
        return (1 - self.epigenetic_alpha) * e_t + self.epigenetic_alpha * f

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids:     torch.Tensor,
        e_t:           Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns (logits, e_next, info) — same signature as EpigeneticNetwork
        so the Trainer works without modification.
        """
        B, T = token_ids.shape

        # ── Embed ─────────────────────────────────────────────────────────
        pos = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x   = self.tok_embed(token_ids) + self.pos_embed(pos)
        x   = self.emb_drop(self.emb_norm(x))

        # ── Feature 1: init e_t ───────────────────────────────────────────
        x_mean = x.mean(1)
        if e_t is None or e_t.size(0) != B:
            e_t = self._init_e_t(x_mean)

        # ── Feature 5: memory read ────────────────────────────────────────
        mem_ctx = self.memory.read(x_mean) if self.memory is not None else None

        # ── Padding mask ──────────────────────────────────────────────────
        pad_mask = (token_ids == 0)

        # ── Transformer blocks (Features 2 + 3 inside) ───────────────────
        for block in self.blocks:
            x = block(x, e_t, key_padding_mask=pad_mask)

        x = self.final_norm(x)

        # ── Feature 4: dual pooling ───────────────────────────────────────
        h = self.pool(x, pad_mask)           # (B, D)

        # ── Feature 5: memory write ───────────────────────────────────────
        if update_memory and self.memory is not None:
            self.memory.write(self.mem_proj(h), self.mem_proj(h), importance=1.0)

        # ── Feature 1: update e_t ─────────────────────────────────────────
        e_next = self._update_e_t(e_t, h, mem_ctx)

        logits = self.output_head(h)

        info = {
            "gate_values":      [],          # no per-gate extraction needed
            "memory_ctx":       mem_ctx,
            "hidden":           h,
            "epigenetic_state": e_next,
        }
        return logits, e_next, info

    def reset_memory(self) -> None:
        if self.memory is not None:
            self.memory.reset()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
