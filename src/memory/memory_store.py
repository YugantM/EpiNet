"""
Associative memory store for the Epigenetic Neural Network.

Design motivation
-----------------
Biological neurons modulate behavior based not just on current input but on
*context accumulated over time*.  This module implements a differentiable
key-value memory that the network can read from and write to at every forward
pass.

Architecture
------------
- Fixed capacity of *memory_size* slots.
- Each slot stores:  (key vector, value vector, importance scalar)
- **Read**  : soft attention over keys → weighted sum of values
- **Write** : replace the least-important slot (importance-weighted LRU)
- **Decay** : importance decays by `decay` factor each time step so old
              memories fade unless repeatedly reinforced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MemoryStore(nn.Module):
    """
    Differentiable key-value episodic memory.

    Parameters
    ----------
    memory_size : int
        Number of memory slots.
    memory_dim : int
        Dimension of keys and values (should match hidden layer dim).
    decay : float
        Multiplicative decay applied to all importance scores each step.
    top_k : int
        Number of top-scoring slots used for the read operation.
    """

    def __init__(
        self,
        memory_size: int = 64,
        memory_dim: int = 64,
        decay: float = 0.99,
        top_k: int = 4,
    ) -> None:
        super().__init__()

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.decay = decay
        self.top_k = min(top_k, memory_size)

        # Persistent buffers (not parameters — not trained by gradient descent)
        self.register_buffer("keys",       torch.zeros(memory_size, memory_dim))
        self.register_buffer("values",     torch.zeros(memory_size, memory_dim))
        self.register_buffer("importance", torch.zeros(memory_size))
        self.register_buffer("age",        torch.zeros(memory_size))   # write counter

        # Learnable projection to align query/key spaces
        self.query_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.key_proj   = nn.Linear(memory_dim, memory_dim, bias=False)

        # Temperature for softmax attention
        self.temperature = nn.Parameter(torch.tensor(1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve a context vector from memory using soft attention.

        Parameters
        ----------
        query : Tensor of shape (batch, memory_dim)

        Returns
        -------
        context : Tensor of shape (batch, memory_dim)
        """
        # Project query and keys into shared comparison space.
        # Clone + detach the buffers so that the in-place slot writes that happen
        # later in the same forward pass do not bump the version counter of the
        # tensors already in the autograd graph.
        q = self.query_proj(query)                     # (B, D)
        k = self.key_proj(self.keys.clone().detach())  # (M, D)

        # Cosine similarity attention
        q_norm = F.normalize(q, dim=-1)                # (B, D)
        k_norm = F.normalize(k, dim=-1)                # (M, D)
        scores = torch.matmul(q_norm, k_norm.t())      # (B, M)

        # Weight scores by slot importance so recent/important memories dominate
        imp_weight = F.softmax(self.importance.detach(), dim=0).unsqueeze(0)  # (1, M)
        scores = scores + imp_weight

        # Restrict to top-k for efficiency (zero out the rest)
        if self.top_k < self.memory_size:
            topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            scores = scores * mask + (1 - mask) * (-1e9)

        attn = F.softmax(scores / (self.temperature.abs() + 1e-6), dim=-1)  # (B, M)
        context = torch.matmul(attn, self.values.clone().detach())   # (B, D)
        return context

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: float = 1.0,
    ) -> None:
        """
        Write a new key-value pair into the least important slot.

        Parameters
        ----------
        key : Tensor of shape (memory_dim,) or (batch, memory_dim)
            If batched, we write the *mean* key/value (summarising the batch).
        value : Tensor same shape as key.
        importance : float
            Initial importance assigned to this slot.
        """
        # Memory writes are non-differentiable side effects — use no_grad to
        # avoid in-place modification of tensors in the autograd graph.
        with torch.no_grad():
            # Decay all existing importances first
            self.importance.mul_(self.decay)

            # Aggregate batch dimension if present
            if key.dim() == 2:
                key   = key.mean(dim=0).detach()
                value = value.mean(dim=0).detach()
            else:
                key   = key.detach()
                value = value.detach()

            # Find the slot with lowest importance (eviction target)
            slot_idx = self.importance.argmin().item()

            self.keys[slot_idx]       = key
            self.values[slot_idx]     = value
            self.importance[slot_idx] = importance
            self.age[slot_idx]        = self.age.max() + 1

    def reset(self) -> None:
        """Clear all memory slots (called between tasks in continual learning)."""
        self.keys.zero_()
        self.values.zero_()
        self.importance.zero_()
        self.age.zero_()

    def utilization(self) -> float:
        """Fraction of slots with non-zero importance."""
        return (self.importance > 1e-6).float().mean().item()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"memory_size={self.memory_size}, memory_dim={self.memory_dim}, "
            f"decay={self.decay}, top_k={self.top_k}"
        )
