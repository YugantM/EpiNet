"""
EpigeneticLayer — a full hidden layer composed of EpigeneticNeurons.

Rather than a single wide linear projection, the layer groups neurons into
*heads* (analogous to multi-head attention), each with its own epigenetic
gate.  Outputs are concatenated and projected to the target dimension.

This design allows different neuron groups to specialise under different
epigenetic states (e.g., one head may be silenced by high-stress signals
while another remains active).
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from .epigenetic_neuron import EpigeneticNeuron


class EpigeneticLayer(nn.Module):
    """
    A multi-head epigenetically-modulated hidden layer.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    epigenetic_dim : int
    num_heads : int
        Number of neuron groups.  Each head has input_dim → output_dim//num_heads
        neurons; outputs are concatenated.
    memory_lambda : float
    activation : str
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        epigenetic_dim: int,
        num_heads: int = 4,
        memory_lambda: float = 0.1,
        activation: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if output_dim % num_heads != 0:
            # Adjust num_heads to cleanly divide output_dim
            num_heads = 1

        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.epigenetic_dim = epigenetic_dim
        self.num_heads      = num_heads
        self.head_dim       = output_dim // num_heads

        # One EpigeneticNeuron per head
        self.heads = nn.ModuleList([
            EpigeneticNeuron(
                input_dim      = input_dim,
                output_dim     = self.head_dim,
                epigenetic_dim = epigenetic_dim,
                memory_lambda  = memory_lambda,
                activation     = activation,
            )
            for _ in range(num_heads)
        ])

        # Projection from concatenated head outputs → output_dim
        # (identity when num_heads == 1 and head_dim == output_dim)
        concat_dim = self.head_dim * num_heads
        self.output_proj = (
            nn.Linear(concat_dim, output_dim, bias=False)
            if concat_dim != output_dim
            else nn.Identity()
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout    = nn.Dropout(dropout)

        # Residual projection when input_dim != output_dim
        self.residual_proj = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        x : (batch, input_dim)
        e : (batch, epigenetic_dim)
        memory : (batch, input_dim) or None

        Returns
        -------
        out : (batch, output_dim)
            Layer output after residual + layer-norm.
        gate_values : list of (batch, head_dim) tensors
            Gate activations from each head (useful for interpretability).
        """
        head_outputs: List[torch.Tensor] = []
        gate_values:  List[torch.Tensor] = []

        for head in self.heads:
            h = head(x, e, memory)           # (B, head_dim)
            head_outputs.append(h)
            gate_values.append(head.gate_values(e))

        # Aggregate: concatenate then project
        concat = torch.cat(head_outputs, dim=-1)   # (B, head_dim * num_heads)
        projected = self.output_proj(concat)        # (B, output_dim)
        projected = self.dropout(projected)

        # Pre-norm residual connection
        residual = self.residual_proj(x)            # (B, output_dim)
        out = self.layer_norm(projected + residual)

        return out, gate_values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mean_gate_activation(self, e: torch.Tensor) -> float:
        """
        Return average gate activation across all heads (scalar).
        Useful for homeostasis monitoring.
        """
        total = 0.0
        with torch.no_grad():
            for head in self.heads:
                g = torch.sigmoid(head.W_e(e))
                total += g.mean().item()
        return total / self.num_heads

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}"
        )
