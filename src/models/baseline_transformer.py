"""
Baseline Transformer model for comparison with EpigeneticNetwork.

This is an intentionally small model (matching EpiNet's parameter budget)
so comparisons are fair.  It uses:
  - Token + positional embeddings
  - N stacked Transformer encoder layers (multi-head self-attention + FFN)
  - Mean-pool → classification head

The architecture deliberately avoids any dynamic-state mechanism so we can
isolate the contribution of the epigenetic / memory components.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        self.qkv_proj   = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x    : (B, T, D)
        mask : (B, T) boolean — True = pad token to ignore
        """
        B, T, D = x.shape

        qkv = self.qkv_proj(x)                               # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                       # each (B, T, D)

        # Reshape to (B, heads, T, head_dim)
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T) — expand to (B, 1, 1, T)
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out  = torch.matmul(attn, v)                          # (B, H, T, head_dim)
        out  = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → MHSA → residual, LN → FFN → residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    """
    Small Transformer encoder for classification tasks.

    Parameters
    ----------
    vocab_size  : int
    embed_dim   : int
    num_heads   : int
    num_layers  : int
    ffn_dim     : int   (FFN hidden dimension; defaults to 4 * embed_dim)
    num_classes : int
    max_seq_len : int
    dropout     : float
    """

    def __init__(
        self,
        vocab_size:  int = 5000,
        embed_dim:   int = 64,
        num_heads:   int = 4,
        num_layers:  int = 2,
        ffn_dim:     Optional[int] = None,
        num_classes: int = 2,
        max_seq_len: int = 256,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        ffn_dim = ffn_dim or (4 * embed_dim)

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed   = nn.Embedding(max_seq_len, embed_dim)
        self.embed_drop  = nn.Dropout(dropout)
        self.embed_norm  = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        pad_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (B, T)   long tensor
        pad_mask  : (B, T)   bool tensor — True where padding

        Returns
        -------
        logits : (B, num_classes)
        """
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)

        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.embed_drop(self.embed_norm(x))             # (B, T, D)

        for layer in self.layers:
            x = layer(x, pad_mask)

        x = self.output_norm(x)

        # Mean-pool (ignoring padding)
        if pad_mask is not None:
            lengths = (~pad_mask).sum(dim=1, keepdim=True).unsqueeze(-1).float()
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            x = x.sum(dim=1) / lengths.squeeze(-1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.classifier(x)

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
