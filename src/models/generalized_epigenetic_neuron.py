"""
GeneralizedEpigeneticNeuron
============================

Extends EpigeneticNeuron by making the gate *input-aware*:

    Original gate:    g = σ(W_e · e_t)
                      Blind to the current token — pure context routing.

    Generalized gate: q = W_q · x          (query from current input)
                      k = W_k · e_t        (key from persistent state)
                      g = σ( q·k/√D + W_e·e_t )

    The dot-product term answers: "given what e_t knows about context,
    how relevant is *this specific input* right now?"

    When q·k → 0 (orthogonal), the gate collapses to the original EpiNeuron.
    The generalization is therefore a strict superset.

5-step forward pass (compare with epigenetic_neuron.py):
─────────────────────────────────────────────────────────
  ① u   = W·x + b                          linear projection
  ② q   = W_q·x,  k = W_k·e_t
     g   = σ( (q·k)/√D_out + W_e·e_t )    input-aware gate
  ③ u'  = g ⊙ u                            element-wise modulation
  ④ u'' = u' + λ·proj(m)                   memory injection  (unchanged)
  ⑤ y   = act(u'') + β·W_skip·x            residual to input (new)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GeneralizedEpigeneticNeuron(nn.Module):
    """
    Input-aware epigenetic gating unit.

    Parameters
    ----------
    input_dim      : dimension of input x
    output_dim     : dimension of output y
    epigenetic_dim : dimension of e_t
    memory_lambda  : λ — weight of memory injection (0 = disable)
    activation     : one of 'relu', 'gelu', 'tanh', 'silu'
    skip_weight    : β — weight of input residual (0 = no skip)
    """

    _ACTIVATIONS = {
        "relu":  F.relu,
        "gelu":  F.gelu,
        "tanh":  torch.tanh,
        "silu":  F.silu,
    }

    def __init__(
        self,
        input_dim:      int,
        output_dim:     int,
        epigenetic_dim: int,
        memory_lambda:  float = 0.1,
        activation:     str   = "gelu",
        skip_weight:    float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.epigenetic_dim = epigenetic_dim
        self.memory_lambda  = memory_lambda
        self.skip_weight    = skip_weight
        self._act           = self._ACTIVATIONS[activation]

        # ── ① Linear projection ──────────────────────────────────────────
        self.W   = nn.Linear(input_dim, output_dim)

        # ── ② Input-aware gate ───────────────────────────────────────────
        self.W_q = nn.Linear(input_dim,      output_dim, bias=False)  # query
        self.W_k = nn.Linear(epigenetic_dim, output_dim, bias=False)  # key
        self.W_e = nn.Linear(epigenetic_dim, output_dim, bias=True)   # bias term

        self._scale = output_dim ** -0.5

        # ── ④ Memory projection ──────────────────────────────────────────
        if memory_lambda > 0.0:
            self.memory_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.memory_proj = None

        # ── ⑤ Input residual (skip connection) ──────────────────────────
        if skip_weight > 0.0 and input_dim != output_dim:
            self.W_skip = nn.Linear(input_dim, output_dim, bias=False)
        elif skip_weight > 0.0:
            self.W_skip = nn.Identity()
        else:
            self.W_skip = None

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:   torch.Tensor,                    # (B, input_dim)
        e_t: torch.Tensor,                    # (B, epigenetic_dim)
        m:   Optional[torch.Tensor] = None,   # (B, input_dim) memory context
    ) -> tuple:
        """
        Returns
        -------
        y    : (B, output_dim)
        gate : (B, output_dim)  gate values in (0, 1) — for inspection
        """
        # ① Linear
        u = self.W(x)                                         # (B, D_out)

        # ② Input-aware gate
        q     = self.W_q(x)                                   # (B, D_out)
        k     = self.W_k(e_t)                                 # (B, D_out)
        score = (q * k) * self._scale                         # (B, D_out)
        gate  = torch.sigmoid(score + self.W_e(e_t))          # (B, D_out)

        # ③ Modulate
        u_prime = gate * u

        # ④ Memory injection
        if self.memory_proj is not None and m is not None:
            u_prime = u_prime + self.memory_lambda * self.memory_proj(m)

        # ⑤ Activate + residual
        y = self._act(u_prime)
        if self.W_skip is not None:
            y = y + self.skip_weight * self.W_skip(x)

        return y, gate
