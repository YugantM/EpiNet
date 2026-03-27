"""
Configuration management for EpiNet experiments.

Provides a dataclass-based config system with sensible defaults
for the Epigenetic Neural Network architecture.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
import os


@dataclass
class ModelConfig:
    """Core architecture hyperparameters."""

    # Input / embedding
    vocab_size: int = 5000          # vocabulary size for text tasks
    embed_dim: int = 64             # embedding dimension
    input_dim: int = 64             # input feature dimension

    # Epigenetic layer stack
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    epigenetic_dim: int = 32        # dimension of epigenetic state vector e_t
    num_epigenetic_layers: int = 2

    # Gating / modulation
    memory_lambda: float = 0.1      # λ: weight of memory contribution
    epigenetic_alpha: float = 0.3   # α: epigenetic update rate (EMA coefficient)

    # Output
    num_classes: int = 2            # binary classification default
    dropout: float = 0.1


@dataclass
class MemoryConfig:
    """Memory system hyperparameters."""

    memory_size: int = 64           # number of memory slots
    memory_dim: int = 64            # dimension of each memory vector
    decay: float = 0.99             # importance decay per step
    top_k: int = 4                  # number of top-k slots retrieved per query


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    batch_size: int = 32
    max_seq_len: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    seed: int = 42

    # Homeostasis
    homeostasis_enabled: bool = True
    stress_threshold: float = 0.5   # activation above this triggers adaptation
    lr_adaptation_factor: float = 0.5  # shrink LR when stressed


@dataclass
class ExperimentConfig:
    """Top-level experiment config combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    experiment_name: str = "epinet_default"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"             # always CPU for this prototype

    def save(self, path: str) -> None:
        """Serialize config to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        cfg.model = ModelConfig(**data.get("model", {}))
        cfg.memory = MemoryConfig(**data.get("memory", {}))
        cfg.training = TrainingConfig(**data.get("training", {}))
        for k, v in data.items():
            if k not in ("model", "memory", "training"):
                setattr(cfg, k, v)
        return cfg

    def _to_dict(self) -> dict:
        import dataclasses
        return {
            "model": dataclasses.asdict(self.model),
            "memory": dataclasses.asdict(self.memory),
            "training": dataclasses.asdict(self.training),
            "experiment_name": self.experiment_name,
            "log_dir": self.log_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "device": self.device,
        }


def get_default_config() -> ExperimentConfig:
    """Return default experiment configuration."""
    return ExperimentConfig()
