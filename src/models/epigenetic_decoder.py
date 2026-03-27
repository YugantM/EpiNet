"""
EpigeneticDecoder — seq2seq text-generation decoder built on top of EpigeneticNetwork.

Architecture
------------

    Question tokens
          │
          ▼
    EpigeneticNetwork (encoder)
          │  encoder_hidden (B, D)
          │  e_t (persistent modulation state)
          │
          ▼
    ┌──────────────────────────────────────────────┐
    │  Autoregressive Decoder                      │
    │                                              │
    │  for each generation step s:                 │
    │    prev_token → Embedding                    │
    │    cat(token_emb, encoder_hidden, e_t)       │
    │      → GRU cell  (hidden state across steps) │
    │      → Linear projection → vocab logits      │
    │                                              │
    │    e_t updated via EMA at each step:         │
    │    e_{s+1} = (1-α)*e_s + α*f(gru_out, e_s)  │
    └──────────────────────────────────────────────┘

Why GRU?
  The GRU tracks short-term decoding state (what has been generated so far)
  while e_t tracks the longer-horizon context (topic, style).  Separating
  these two timescales is the key design decision; a pure transformer decoder
  would have no persistent cross-step routing signal.

Why condition on e_t at every step?
  The encoder's modulation state carries topic/context information extracted
  from the question.  Injecting it at every decoder step steers the generator
  toward on-topic tokens even without a separate cross-attention layer, giving
  a lightweight cross-step guidance mechanism at O(D) cost.

Comparable systems
  - Standard seq2seq: encoder LSTM → decoder LSTM + attention (Bahdanau 2015)
  - T5/BART: transformer encoder-decoder with cross-attention
  - This decoder: EpiNet encoder → GRU decoder with e_t injection (no cross-attn)

Novel vs existing:
  - GRU token decoder: existing
  - Conditioning decoder on a persistent encoder-side modulation vector at every
    step (without cross-attention): the specific coupling mechanism is novel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .epigenetic_network import EpigeneticNetwork


class EpigeneticSeq2Seq(nn.Module):
    """
    Seq2seq model for open-ended text generation.

    Encoder : EpigeneticNetwork (shared vocab)
    Decoder : single-layer GRU conditioned on encoder_hidden and e_t

    Parameters
    ----------
    vocab_size      : size of shared vocabulary
    embed_dim       : token embedding dimension
    epigenetic_dim  : dimension of e_t (must match the encoder)
    hidden_dims     : layer sizes inside the EpigenethicNetwork encoder
    decoder_hidden  : GRU hidden size
    memory_size     : MemoryStore capacity
    memory_dim      : MemoryStore key/value dimension
    dropout         : dropout applied throughout
    max_seq_len     : maximum input sequence length for positional encoding
    """

    def __init__(
        self,
        vocab_size:     int   = 651,
        embed_dim:      int   = 64,
        epigenetic_dim: int   = 32,
        hidden_dims:    Optional[List[int]] = None,
        decoder_hidden: int   = 128,
        memory_size:    int   = 64,
        memory_dim:     int   = 64,
        dropout:        float = 0.1,
        max_seq_len:    int   = 64,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.vocab_size     = vocab_size
        self.embed_dim      = embed_dim
        self.epigenetic_dim = epigenetic_dim
        self.decoder_hidden = decoder_hidden

        # ── Encoder (EpigeneticNetwork, classification head unused) ──────
        self.encoder = EpigeneticNetwork(
            vocab_size       = vocab_size,
            embed_dim        = embed_dim,
            hidden_dims      = hidden_dims,
            epigenetic_dim   = epigenetic_dim,
            num_classes      = 2,             # head not used during generation
            memory_size      = memory_size,
            memory_dim       = memory_dim,
            memory_lambda    = 0.1,
            epigenetic_alpha = 0.3,
            num_heads        = 4,
            dropout          = dropout,
            max_seq_len      = max_seq_len,
        )

        enc_out_dim = hidden_dims[-1]  # dimension produced by EpigeneticNetwork

        # ── Decoder token embedding (shared with encoder vocabulary) ──────
        self.dec_embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dec_drop   = nn.Dropout(dropout)

        # GRU input = token embedding + encoder hidden + epigenetic state
        gru_input_dim = embed_dim + enc_out_dim + epigenetic_dim
        self.gru = nn.GRUCell(gru_input_dim, decoder_hidden)

        # e_t update network inside the decoder (separate from encoder's)
        self.epi_update = nn.Sequential(
            nn.Linear(decoder_hidden + epigenetic_dim, epigenetic_dim),
            nn.Tanh(),
        )
        self.epigenetic_alpha = 0.3

        # ── Output projection ─────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, vocab_size),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Encode
    # ──────────────────────────────────────────────────────────────────────

    def encode(
        self, src_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the EpigeneticNetwork encoder on source tokens.

        Returns
        -------
        enc_hidden : (batch, hidden_dims[-1])   pooled encoder representation
        e_t        : (batch, epigenetic_dim)    final epigenetic state
        """
        _, e_t, info = self.encoder(src_tokens, e_t=None, update_memory=True)
        enc_hidden = info["hidden"]   # (B, hidden_dims[-1])
        return enc_hidden, e_t

    # ──────────────────────────────────────────────────────────────────────
    # Decode one step
    # ──────────────────────────────────────────────────────────────────────

    def decode_step(
        self,
        prev_token:  torch.Tensor,   # (B,) long
        enc_hidden:  torch.Tensor,   # (B, enc_out_dim)
        e_t:         torch.Tensor,   # (B, epigenetic_dim)
        gru_state:   torch.Tensor,   # (B, decoder_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single autoregressive decode step.

        Returns
        -------
        logits    : (B, vocab_size)   raw unnormalised scores
        e_next    : (B, epigenetic_dim)
        gru_next  : (B, decoder_hidden)
        """
        tok_emb = self.dec_drop(self.dec_embed(prev_token))   # (B, D)
        gru_in  = torch.cat([tok_emb, enc_hidden, e_t], dim=-1)
        gru_out = self.gru(gru_in, gru_state)                 # (B, dec_hidden)

        # Update e_t
        e_update_in = torch.cat([gru_out, e_t], dim=-1)
        f_val  = self.epi_update(e_update_in)
        e_next = (1 - self.epigenetic_alpha) * e_t + self.epigenetic_alpha * f_val

        logits = self.out_proj(gru_out)                       # (B, vocab_size)
        return logits, e_next, gru_out

    # ──────────────────────────────────────────────────────────────────────
    # Teacher-forced forward (training)
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        src_tokens: torch.Tensor,   # (B, src_len)
        tgt_tokens: torch.Tensor,   # (B, tgt_len)   includes BOS, not EOS
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass for training.

        Returns
        -------
        logits : (B, tgt_len, vocab_size)
        """
        B, tgt_len = tgt_tokens.shape
        enc_hidden, e_t = self.encode(src_tokens)
        gru_state = torch.zeros(B, self.decoder_hidden, device=src_tokens.device)

        logits_seq = []
        for s in range(tgt_len):
            logits, e_t, gru_state = self.decode_step(
                tgt_tokens[:, s], enc_hidden, e_t, gru_state
            )
            logits_seq.append(logits)

        return torch.stack(logits_seq, dim=1)   # (B, tgt_len, vocab_size)

    # ──────────────────────────────────────────────────────────────────────
    # Autoregressive generation (inference)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        src_tokens:     torch.Tensor,
        bos_id:         int,
        eos_id:         int,
        max_new_tokens: int = 20,
        temperature:    float = 1.0,
        top_k:          int = 0,
        top_p:          float = 0.9,
    ) -> List[int]:
        """
        Greedy / nucleus-sampled generation for a single sequence.

        Parameters
        ----------
        src_tokens     : (1, src_len) question tokens
        bos_id         : vocabulary index of BOS / SEP token to start decoding
        eos_id         : vocabulary index of EOS token that terminates generation
        max_new_tokens : hard cap on output length
        temperature    : logit scaling; 1.0 = no change; < 1 = sharper
        top_k          : if > 0, keep only top-k tokens before sampling
        top_p          : nucleus (top-p) probability threshold

        Returns
        -------
        generated_ids : list of int  (excluding the BOS seed token)
        """
        assert src_tokens.size(0) == 1, "generate() works on single sequences"
        self.eval()

        enc_hidden, e_t = self.encode(src_tokens)
        gru_state = torch.zeros(1, self.decoder_hidden, device=src_tokens.device)

        generated = []
        current_token = torch.tensor([bos_id], device=src_tokens.device)

        for _ in range(max_new_tokens):
            logits, e_t, gru_state = self.decode_step(
                current_token, enc_hidden, e_t, gru_state
            )
            logits = logits[0] / max(temperature, 1e-8)   # (vocab_size,)

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[-1]] = float("-inf")

            # Top-p (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=0)
                # Remove tokens with cumulative prob above threshold (keep ≥1)
                remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            if token_id == eos_id:
                break
            generated.append(token_id)
            current_token = next_token

        return generated

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
