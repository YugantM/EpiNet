"""
Additional baseline models for fair benchmarking against EpigeneticNetwork.

Models:
  - MLPClassifier       : 2-layer MLP (mean-pooled bag-of-embeddings)
  - BiLSTMClassifier    : Bidirectional LSTM with mean-pool classifier

Both share the same embedding layer convention (vocab_size, embed_dim) so they
are directly comparable to EpigeneticNetwork and BaselineTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPClassifier(nn.Module):
    """
    Bag-of-embeddings MLP baseline.

    Pipeline:
      token_ids → Embedding → mean-pool → LayerNorm → MLP → logits

    No positional encoding, no sequence order awareness.
    This is the weakest baseline — any sequence-aware model should beat it.
    """

    def __init__(
        self,
        vocab_size:  int = 5000,
        embed_dim:   int = 64,
        hidden_dim:  int = 128,
        num_classes: int = 2,
        dropout:     float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.norm    = nn.LayerNorm(embed_dim)
        self.mlp     = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """token_ids : (B, T)  →  logits : (B, num_classes)"""
        x = self.embed(token_ids)           # (B, T, D)
        # Mean pool (ignore padding index 0)
        mask = (token_ids != 0).float().unsqueeze(-1)  # (B, T, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)
        x = self.norm(x)
        return self.mlp(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier.

    Pipeline:
      token_ids → Embedding → BiLSTM → mean-pool → LayerNorm → Linear → logits

    Captures sequential order; closer in spirit to the Transformer but uses
    recurrence instead of attention.
    """

    def __init__(
        self,
        vocab_size:   int = 5000,
        embed_dim:    int = 64,
        hidden_dim:   int = 64,
        num_layers:   int = 2,
        num_classes:  int = 2,
        dropout:      float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2   # bidirectional
        self.norm       = nn.LayerNorm(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """token_ids : (B, T)  →  logits : (B, num_classes)"""
        x, _ = self.lstm(self.embed(token_ids))   # (B, T, 2H)
        # Mean pool over non-padding positions
        mask = (token_ids != 0).float().unsqueeze(-1)  # (B, T, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 2H)
        x = self.norm(x)
        return self.classifier(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
