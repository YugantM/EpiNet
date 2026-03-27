"""
EpiTransformer — Hybrid Transformer + Generalised Epigenetic Architecture
==========================================================================

Each block contains TWO sub-layers (instead of Transformer's one FFN):

    ┌──────────────────────────────────────────────────────────┐
    │  EpiTransformerBlock                                     │
    │                                                          │
    │  X (B,T,D)                                               │
    │    │                                                     │
    │    ├─ LayerNorm → Multi-Head Self-Attention → residual   │
    │    │                                                     │
    │    └─ LayerNorm → GeneralizedEpigeneticFFN               │
    │                   (each token gated by e_t)   → residual │
    │                                                          │
    │  out: H (B,T,D),  gate_values (B,T,D)                    │
    └──────────────────────────────────────────────────────────┘

The epigenetic state e_t (B, epigenetic_dim) is:
  • Shared across all blocks in a forward pass
  • Updated via EMA after each full forward pass:
        e_next = (1-α)·e_t + α·Tanh(Linear(pooled_hidden))
  • Persists between calls — it is NOT reset between batches unless
    the caller explicitly passes e_t=None

Design decisions
----------------
  MHSA handles within-sequence token dependencies (Transformer's strength).
  GeneralizedEpigeneticFFN modulates *what* each token's features mean given
  the accumulated context state — without growing the sequence or context window.

  The two mechanisms are independent:
    MHSA:  "which tokens are relevant to each other?"
    e_t gate: "which features are relevant given current context?"

Comparable systems
------------------
  - Standard Transformer (Vaswani 2017): MHSA + FFN, no persistent state
  - Evolved Transformer: searched FFN variants, no external state
  - FNet: replaces attention with FFT; no persistent state
  - This: MHSA + epigenetically-gated FFN + EMA persistent state  ← novel combo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

from .generalized_epigenetic_neuron import GeneralizedEpigeneticNeuron
from ..memory.memory_store import MemoryStore


# ─────────────────────────────────────────────────────────────────────────────
# Generalised Epigenetic FFN sublayer
# ─────────────────────────────────────────────────────────────────────────────

class GeneralizedEpigeneticFFN(nn.Module):
    """
    Drop-in replacement for a Transformer FFN sublayer.

    Applies GeneralizedEpigeneticNeuron independently to every token position,
    broadcasting e_t (which is per-sequence, not per-position) across T.

    inner_dim defaults to 4×embed_dim, matching standard Transformer FFN.
    """

    def __init__(
        self,
        embed_dim:      int,
        epigenetic_dim: int,
        inner_dim:      Optional[int] = None,
        memory_lambda:  float = 0.1,
        skip_weight:    float = 0.1,
        dropout:        float = 0.1,
    ) -> None:
        super().__init__()

        inner_dim = inner_dim or (embed_dim * 4)

        # Two-layer structure mirroring Transformer FFN depth
        self.neuron1 = GeneralizedEpigeneticNeuron(
            input_dim=embed_dim, output_dim=inner_dim,
            epigenetic_dim=epigenetic_dim,
            memory_lambda=memory_lambda, activation="gelu",
            skip_weight=0.0,  # skip only on final layer
        )
        self.neuron2 = GeneralizedEpigeneticNeuron(
            input_dim=inner_dim, output_dim=embed_dim,
            epigenetic_dim=epigenetic_dim,
            memory_lambda=0.0,  # memory injection only on first layer
            activation="gelu",
            skip_weight=skip_weight,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x:   torch.Tensor,            # (B, T, D)
        e_t: torch.Tensor,            # (B, D_e)
        m:   Optional[torch.Tensor],  # (B, D) or None — memory context
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out        : (B, T, D)
        gate_vals  : (B, T, D_inner)  gate activations from first neuron
        """
        B, T, D = x.shape

        # Flatten token dimension for neuron forward
        x_flat = x.view(B * T, D)

        # Broadcast e_t over token positions
        e_t_flat = e_t.unsqueeze(1).expand(B, T, -1).contiguous().view(B * T, -1)

        # Broadcast memory context over token positions (or None)
        if m is not None:
            m_flat = m.unsqueeze(1).expand(B, T, -1).contiguous().view(B * T, -1)
        else:
            m_flat = None

        h, gate_vals = self.neuron1(x_flat, e_t_flat, m_flat)
        h = self.dropout(h)

        # Second neuron — no memory injection, e_t still conditions gate
        h_e_t = e_t.unsqueeze(1).expand(B, T, -1).contiguous().view(B * T, -1)
        out_flat, _ = self.neuron2(h, h_e_t, None)

        out = out_flat.view(B, T, D)
        gate_vals = gate_vals.view(B, T, -1)

        # Residual + norm
        out = self.norm(out + x)
        return out, gate_vals


# ─────────────────────────────────────────────────────────────────────────────
# Single EpiTransformer block
# ─────────────────────────────────────────────────────────────────────────────

class EpiTransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with GeneralizedEpigeneticFFN replacing
    the standard position-wise FFN.

    Sub-layers:
        1. LayerNorm → MHSA        → residual
        2. LayerNorm → EpigeneticFFN (conditioned on e_t) → residual
    """

    def __init__(
        self,
        embed_dim:      int,
        num_heads:      int,
        epigenetic_dim: int,
        inner_dim:      Optional[int] = None,
        memory_lambda:  float = 0.1,
        skip_weight:    float = 0.1,
        dropout:        float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.epi_ffn = GeneralizedEpigeneticFFN(
            embed_dim=embed_dim,
            epigenetic_dim=epigenetic_dim,
            inner_dim=inner_dim,
            memory_lambda=memory_lambda,
            skip_weight=skip_weight,
            dropout=dropout,
        )

    def forward(
        self,
        x:        torch.Tensor,            # (B, T, D)
        e_t:      torch.Tensor,            # (B, D_e)
        memory_ctx: Optional[torch.Tensor], # (B, D) or None
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out       : (B, T, D)
        gate_vals : (B, T, inner_dim)
        """
        # ── Sub-layer 1: MHSA ────────────────────────────────────────────
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_drop(attn_out)

        # ── Sub-layer 2: GeneralizedEpigeneticFFN ────────────────────────
        normed = self.norm2(x)
        ffn_out, gate_vals = self.epi_ffn(normed, e_t, memory_ctx)
        x = x + ffn_out

        return x, gate_vals


# ─────────────────────────────────────────────────────────────────────────────
# Full EpiTransformer model
# ─────────────────────────────────────────────────────────────────────────────

class EpiTransformer(nn.Module):
    """
    Full EpiTransformer: stacked EpiTransformerBlocks with persistent e_t state.

    Parameters
    ----------
    vocab_size      : vocabulary size
    embed_dim       : token embedding / model dimension
    num_heads       : attention heads per block
    num_layers      : number of EpiTransformerBlocks
    epigenetic_dim  : dimension of persistent context state e_t
    num_classes     : output classes
    inner_dim       : FFN inner dimension (default 4×embed_dim)
    memory_size     : MemoryStore capacity
    memory_dim      : MemoryStore key/value dimension (should equal embed_dim)
    epigenetic_alpha: EMA decay for e_t update
    memory_lambda   : λ for memory injection in FFN
    skip_weight     : β for input residual in GeneralizedEpigeneticNeuron
    dropout         : dropout throughout
    max_seq_len     : positional encoding range
    """

    def __init__(
        self,
        vocab_size:       int   = 5000,
        embed_dim:        int   = 64,
        num_heads:        int   = 4,
        num_layers:       int   = 2,
        epigenetic_dim:   int   = 32,
        num_classes:      int   = 2,
        inner_dim:        Optional[int] = None,
        memory_size:      int   = 64,
        memory_dim:       int   = 64,
        epigenetic_alpha: float = 0.3,
        memory_lambda:    float = 0.1,
        skip_weight:      float = 0.1,
        dropout:          float = 0.1,
        max_seq_len:      int   = 256,
    ) -> None:
        super().__init__()

        self.embed_dim        = embed_dim
        self.epigenetic_dim   = epigenetic_dim
        self.epigenetic_alpha = epigenetic_alpha

        # ── Token + positional embedding ──────────────────────────────────
        self.tok_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.emb_drop  = nn.Dropout(dropout)
        self.emb_norm  = nn.LayerNorm(embed_dim)

        # ── EpiTransformer blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList([
            EpiTransformerBlock(
                embed_dim=embed_dim, num_heads=num_heads,
                epigenetic_dim=epigenetic_dim,
                inner_dim=inner_dim or embed_dim * 4,
                memory_lambda=memory_lambda,
                skip_weight=skip_weight,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Epigenetic state initialiser (from input encoding) ────────────
        self.epi_init = nn.Sequential(
            nn.Linear(embed_dim, epigenetic_dim),
            nn.Tanh(),
        )

        # ── Epigenetic state update network ───────────────────────────────
        # f(pooled_hidden, memory_ctx) → new e_t target
        epi_update_in = embed_dim + (memory_dim if memory_size > 0 else 0)
        self.epi_update_net = nn.Sequential(
            nn.Linear(epi_update_in, epigenetic_dim),
            nn.Tanh(),
        )

        # ── Memory store ──────────────────────────────────────────────────
        if memory_size > 0:
            self.memory = MemoryStore(memory_size=memory_size, memory_dim=memory_dim)
            self.mem_proj = nn.Linear(embed_dim, memory_dim, bias=False)
        else:
            self.memory = None

        # ── Final norm + output head ──────────────────────────────────────
        self.final_norm  = nn.LayerNorm(embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Epigenetic state helpers
    # ──────────────────────────────────────────────────────────────────────

    def _init_e_t(self, x_mean: torch.Tensor) -> torch.Tensor:
        return self.epi_init(x_mean)   # (B, epigenetic_dim)

    def _update_e_t(
        self,
        e_t:       torch.Tensor,
        h_pooled:  torch.Tensor,
        mem_ctx:   Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = [h_pooled]
        if mem_ctx is not None:
            parts.append(mem_ctx)
        f_val = self.epi_update_net(torch.cat(parts, dim=-1))
        return (1 - self.epigenetic_alpha) * e_t + self.epigenetic_alpha * f_val

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        token_ids:     torch.Tensor,            # (B, T)
        e_t:           Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Parameters
        ----------
        token_ids     : (B, T) long
        e_t           : (B, epigenetic_dim) or None — init from input if None
        update_memory : write final hidden state to MemoryStore

        Returns
        -------
        logits  : (B, num_classes)
        e_next  : (B, epigenetic_dim)
        info    : dict with gate_values, memory_ctx, hidden
        """
        B, T = token_ids.shape
        device = token_ids.device

        # ── Embed ─────────────────────────────────────────────────────────
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.tok_embed(token_ids) + self.pos_embed(positions)
        x = self.emb_drop(self.emb_norm(x))              # (B, T, D)

        # ── Init e_t ─────────────────────────────────────────────────────
        x_mean = x.mean(dim=1)                           # (B, D)
        if e_t is None:
            e_t = self._init_e_t(x_mean)

        # Batch-size guard
        if e_t.size(0) != B:
            e_t = self._init_e_t(x_mean)

        # ── Memory read ───────────────────────────────────────────────────
        if self.memory is not None:
            mem_ctx = self.memory.read(x_mean)           # (B, memory_dim)
        else:
            mem_ctx = None

        # ── Padding mask (pad token = 0) ──────────────────────────────────
        pad_mask = (token_ids == 0)                      # (B, T) True = ignore

        # ── Blocks ────────────────────────────────────────────────────────
        all_gate_vals = []
        for block in self.blocks:
            x, gate_vals = block(x, e_t, mem_ctx, key_padding_mask=pad_mask)
            all_gate_vals.append(gate_vals)

        # ── Pool + output ─────────────────────────────────────────────────
        x = self.final_norm(x)
        # Mean pool over non-padding positions
        non_pad = (~pad_mask).float().unsqueeze(-1)      # (B, T, 1)
        h = (x * non_pad).sum(1) / non_pad.sum(1).clamp(min=1)  # (B, D)

        logits = self.output_head(h)                     # (B, C)

        # ── Memory write ──────────────────────────────────────────────────
        if update_memory and self.memory is not None:
            mem_key = self.mem_proj(h)
            self.memory.write(mem_key, mem_key, importance=1.0)

        # ── Update e_t ────────────────────────────────────────────────────
        e_next = self._update_e_t(e_t, h, mem_ctx)

        info = {
            "gate_values":       all_gate_vals,
            "memory_ctx":        mem_ctx,
            "hidden":            h,
            "epigenetic_state":  e_next,
        }
        return logits, e_next, info

    def reset_memory(self) -> None:
        if self.memory is not None:
            self.memory.reset()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
