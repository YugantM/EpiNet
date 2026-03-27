"""
EpigeneticNeuron — the atomic computational unit of EpiNet.

Mathematical definition
-----------------------
Given:
  x  ∈ R^d_in   — input vector
  e  ∈ R^d_e    — epigenetic state vector
  m  ∈ R^d_in   — memory context vector

Step 1 — linear pre-activation:
    u = W·x + b

Step 2 — epigenetic gate (per-neuron modulation):
    g = sigmoid(W_e · e)     g ∈ (0,1)^d_out

Step 3 — gated activation:
    u' = g ⊙ u               element-wise product

Step 4 — memory injection:
    u'' = u' + λ · m

Step 5 — output:
    y = activation(u'')

The epigenetic gate g acts as a *soft switch* that can silence or amplify
individual neurons based on the accumulated epigenetic state e.  This is
analogous to DNA methylation / histone acetylation toggling gene expression
without changing the underlying DNA sequence (base weights W, b).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class EpigeneticNeuron(nn.Module):
    """
    A single epigenetically-modulated neuron (or neuron group).

    Parameters
    ----------
    input_dim : int
        Dimension of input vector x.
    output_dim : int
        Number of output neurons in this unit.
    epigenetic_dim : int
        Dimension of the epigenetic state vector e.
    memory_lambda : float
        Weight λ of memory context added to the pre-activation.
    activation : str
        Non-linearity applied after memory injection.
        One of {'relu', 'gelu', 'tanh', 'sigmoid', 'identity'}.
    """

    _ACTIVATIONS: dict = {
        "relu":     F.relu,
        "gelu":     F.gelu,
        "tanh":     torch.tanh,
        "sigmoid":  torch.sigmoid,
        "identity": lambda x: x,
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        epigenetic_dim: int,
        memory_lambda: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(self._ACTIVATIONS)}"
            )

        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.epigenetic_dim = epigenetic_dim
        self.memory_lambda  = memory_lambda
        self._activation_fn: Callable = self._ACTIVATIONS[activation]

        # ── Genetic core: base weights (W, b) ──────────────────────────────
        # These are the "DNA" — stable parameters trained by gradient descent.
        self.W = nn.Linear(input_dim, output_dim, bias=True)

        # ── Epigenetic gating: W_e maps e → gate per neuron ────────────────
        # W_e is also trainable, but what matters at inference time is the
        # *dynamic* epigenetic state e, not W_e itself.
        self.W_e = nn.Linear(epigenetic_dim, output_dim, bias=False)

        # ── Memory projection: align memory dim with output_dim ────────────
        # Memory context may come from a different-dimensional space.
        self.memory_proj = nn.Linear(input_dim, output_dim, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming-uniform init for base weights; small init for gate weights."""
        nn.init.kaiming_uniform_(self.W.weight, nonlinearity="relu")
        nn.init.zeros_(self.W.bias)
        nn.init.normal_(self.W_e.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.memory_proj.weight, mean=0.0, std=0.01)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the epigenetically-modulated forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)
            Current input.
        e : Tensor of shape (batch, epigenetic_dim) or (1, epigenetic_dim)
            Current epigenetic state.
        memory : Tensor of shape (batch, input_dim) or None
            Memory context retrieved from the MemoryStore.
            If None, the memory contribution is zero.

        Returns
        -------
        y : Tensor of shape (batch, output_dim)
        """
        # Step 1 — linear pre-activation  u = W·x + b
        u = self.W(x)                                   # (B, output_dim)

        # Step 2 — epigenetic gate  g = σ(W_e · e)
        g = torch.sigmoid(self.W_e(e))                  # (B, output_dim)

        # Step 3 — gated activation  u' = g ⊙ u
        u_prime = g * u                                  # (B, output_dim)

        # Step 4 — memory injection  u'' = u' + λ · proj(m)
        if memory is not None:
            m_proj = self.memory_proj(memory)            # (B, output_dim)
            u_double_prime = u_prime + self.memory_lambda * m_proj
        else:
            u_double_prime = u_prime

        # Step 5 — output non-linearity
        y = self._activation_fn(u_double_prime)          # (B, output_dim)
        return y

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def gate_values(self, e: torch.Tensor) -> torch.Tensor:
        """Return the gate vector g for a given epigenetic state (for analysis)."""
        with torch.no_grad():
            return torch.sigmoid(self.W_e(e))

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"epigenetic_dim={self.epigenetic_dim}, "
            f"memory_lambda={self.memory_lambda}"
        )
