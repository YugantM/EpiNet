"""
EpigeneticNetwork — the full end-to-end model.

Architecture overview
---------------------

  Input tokens / features
        │
        ▼
  ┌─────────────┐
  │  Encoder    │   (embedding + positional encoding, or MLP for vectors)
  └──────┬──────┘
         │  x ∈ R^(B, T, D)   (or R^(B, D) for non-sequential)
         ▼
  ┌─────────────────────────────────────────┐
  │  EpigeneticLayer 1                      │◄─── e_t (epigenetic state)
  │  EpigeneticLayer 2  ...                 │◄─── memory context
  └──────────────────┬──────────────────────┘
                     │
                     ▼
             pooling (mean / CLS)
                     │
                     ▼
  ┌─────────────┐
  │ Output Head │   (linear → logits)
  └─────────────┘

The epigenetic state e_t is maintained *per-batch* at inference and evolves
through the EpigeneticController (see controllers/epigenetic_controller.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

from .epigenetic_layer import EpigeneticLayer
from ..memory.memory_store import MemoryStore


class TextEncoder(nn.Module):
    """
    Simple embedding encoder for token sequences.

    Converts token indices → dense vectors with sinusoidal positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (batch, seq_len) long tensor

        Returns
        -------
        (batch, seq_len, embed_dim)
        """
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)  # (1, T)
        x = self.embed(token_ids) + self.pos_embed(positions)
        return self.dropout(self.norm(x))


class MLPEncoder(nn.Module):
    """
    Two-layer MLP encoder for dense feature vectors (non-text tasks).
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, input_dim)  →  (batch, embed_dim)"""
        return self.net(x)


class EpigeneticNetwork(nn.Module):
    """
    Full Epigenetic Neural Network.

    Parameters
    ----------
    vocab_size : int
        Size of token vocabulary (set 0 for non-text / MLP encoder).
    embed_dim : int
        Encoder output / layer input dimension.
    hidden_dims : list of int
        Output dimension for each successive EpigeneticLayer.
    epigenetic_dim : int
        Dimension of the epigenetic state vector e.
    num_classes : int
        Number of output classes.
    memory_size : int
        Number of memory slots.
    memory_dim : int
        Dimension of memory vectors (should equal embed_dim or hidden_dims[0]).
    memory_lambda : float
        Weight λ of memory injection in each neuron.
    epigenetic_alpha : float
        α for EMA-based epigenetic state update.
    num_heads : int
        Number of attention heads per EpigeneticLayer.
    dropout : float
    max_seq_len : int
        Maximum sequence length for positional embedding.
    use_memory : bool
        Whether to enable the memory subsystem.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        epigenetic_dim: int = 32,
        num_classes: int = 2,
        memory_size: int = 64,
        memory_dim: int = 64,
        memory_lambda: float = 0.1,
        epigenetic_alpha: float = 0.3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        use_memory: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.epigenetic_alpha = epigenetic_alpha
        self.epigenetic_dim   = epigenetic_dim
        self.embed_dim        = embed_dim
        self.use_memory       = use_memory

        # ── Encoder ──────────────────────────────────────────────────────
        if vocab_size > 0:
            self.encoder = TextEncoder(
                vocab_size  = vocab_size,
                embed_dim   = embed_dim,
                max_seq_len = max_seq_len,
                dropout     = dropout,
            )
            self.encoder_type = "text"
        else:
            self.encoder = MLPEncoder(embed_dim, embed_dim, dropout)
            self.encoder_type = "mlp"

        # ── Epigenetic Layer Stack ────────────────────────────────────────
        dims = [embed_dim] + hidden_dims
        self.epi_layers = nn.ModuleList([
            EpigeneticLayer(
                input_dim      = dims[i],
                output_dim     = dims[i + 1],
                epigenetic_dim = epigenetic_dim,
                num_heads      = num_heads,
                memory_lambda  = memory_lambda,
                activation     = "relu",
                dropout        = dropout,
            )
            for i in range(len(dims) - 1)
        ])

        # ── Epigenetic State Initialiser ──────────────────────────────────
        # Maps encoded input → initial epigenetic state
        self.epi_init = nn.Linear(embed_dim, epigenetic_dim)

        # ── Epigenetic Update Network f(·) ────────────────────────────────
        # f(input, hidden, memory) → Δe used in EMA update
        epi_update_in = embed_dim + hidden_dims[-1] + (memory_dim if use_memory else 0)
        self.epi_update_net = nn.Sequential(
            nn.Linear(epi_update_in, epigenetic_dim),
            nn.Tanh(),
        )

        # ── Memory Store ─────────────────────────────────────────────────
        if use_memory:
            self.memory = MemoryStore(
                memory_size = memory_size,
                memory_dim  = memory_dim,
            )
            # Project hidden state → memory key/value dimension if needed
            self.mem_key_proj = nn.Linear(hidden_dims[-1], memory_dim, bias=False)
        else:
            self.memory = None

        # ── Output Head ──────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes),
        )

        self._init_epigenetic_state_buffer()

    # ------------------------------------------------------------------
    # Epigenetic state management
    # ------------------------------------------------------------------

    def _init_epigenetic_state_buffer(self) -> None:
        """Register a zero-initialised epigenetic state (not a parameter)."""
        self.register_buffer("_e_t", torch.zeros(1, self.epigenetic_dim))

    def init_epigenetic_state(self, batch_size: int, x_encoded: torch.Tensor) -> torch.Tensor:
        """
        Initialise e_t from the encoded representation of the first token/frame.

        Parameters
        ----------
        batch_size : int
        x_encoded : (batch, embed_dim)

        Returns
        -------
        e_t : (batch, epigenetic_dim)
        """
        e_t = torch.tanh(self.epi_init(x_encoded))
        return e_t

    def update_epigenetic_state(
        self,
        e_t: torch.Tensor,
        x_summary: torch.Tensor,
        hidden: torch.Tensor,
        memory_ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        EMA update rule:

            e_{t+1} = (1 - α) · e_t  +  α · f(x, h, m)

        Parameters
        ----------
        e_t       : (batch, epigenetic_dim)   current state
        x_summary : (batch, embed_dim)        encoded input summary
        hidden    : (batch, hidden_dim)       last layer hidden state
        memory_ctx: (batch, memory_dim) or None

        Returns
        -------
        e_{t+1} : (batch, epigenetic_dim)
        """
        parts = [x_summary, hidden]
        if memory_ctx is not None and self.use_memory:
            parts.append(memory_ctx)
        combined = torch.cat(parts, dim=-1)
        f_val = self.epi_update_net(combined)       # (B, epigenetic_dim)
        e_next = (1 - self.epigenetic_alpha) * e_t + self.epigenetic_alpha * f_val
        return e_next

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        e_t: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Full forward pass.

        Parameters
        ----------
        x : (batch, seq_len) long tensor  (text mode)
            or (batch, feature_dim) float tensor (MLP mode)
        e_t : (batch, epigenetic_dim) or None
            If None, will be initialised from the input.
        update_memory : bool
            Whether to write the current hidden state to memory.

        Returns
        -------
        logits   : (batch, num_classes)
        e_next   : (batch, epigenetic_dim)  updated epigenetic state
        info     : dict with 'gate_values', 'memory_ctx', 'hidden'
        """
        # ── Encode ───────────────────────────────────────────────────────
        if self.encoder_type == "text":
            enc = self.encoder(x)                    # (B, T, D)
            x_summary = enc.mean(dim=1)              # (B, D)  mean pooling
        else:
            enc = self.encoder(x)                    # (B, D)
            x_summary = enc

        batch_size = x_summary.size(0)

        # ── Initialise epigenetic state if not provided ──────────────────
        if e_t is None:
            e_t = self.init_epigenetic_state(batch_size, x_summary)

        # ── Read from memory ─────────────────────────────────────────────
        if self.use_memory and self.memory is not None:
            memory_ctx = self.memory.read(x_summary)   # (B, memory_dim)
        else:
            memory_ctx = None

        # ── Pass through epigenetic layers ───────────────────────────────
        h = x_summary
        all_gate_values = []
        for layer in self.epi_layers:
            # Memory ctx projected to match layer input dim if needed
            mem_in = (
                memory_ctx
                if (memory_ctx is not None and memory_ctx.size(-1) == layer.input_dim)
                else None
            )
            h, gate_vals = layer(h, e_t, mem_in)
            all_gate_values.append(gate_vals)

        # ── Write to memory ──────────────────────────────────────────────
        if update_memory and self.use_memory and self.memory is not None:
            mem_key = self.mem_key_proj(h)             # (B, memory_dim)
            self.memory.write(mem_key, mem_key, importance=1.0)

        # ── Update epigenetic state ───────────────────────────────────────
        e_next = self.update_epigenetic_state(e_t, x_summary, h, memory_ctx)

        # ── Output logits ─────────────────────────────────────────────────
        logits = self.output_head(h)                   # (B, num_classes)

        info = {
            "gate_values":  all_gate_values,
            "memory_ctx":   memory_ctx,
            "hidden":       h,
            "epigenetic_state": e_next,
        }
        return logits, e_next, info

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Simple inference wrapper — returns class predictions."""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(x, update_memory=False)
        return logits.argmax(dim=-1)

    def reset_memory(self) -> None:
        """Clear the memory store (call between tasks in continual learning)."""
        if self.memory is not None:
            self.memory.reset()

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
