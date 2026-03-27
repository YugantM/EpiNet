"""
Training loop for EpigeneticNetwork and BaselineTransformer.

Supports:
  - Standard single-task training
  - Continual learning (sequential task training with optional memory reset)
  - Mixed-precision friendly (float32 on CPU — no AMP needed)

Usage (CLI)
-----------
    python -m src.training.train \\
        --task sentiment \\
        --model epinet \\
        --epochs 5 \\
        --batch_size 32 \\
        --lr 1e-3
"""

import argparse
import logging
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.epigenetic_network import EpigeneticNetwork
from ..models.baseline_transformer import BaselineTransformer
from ..models.epi_transformer import EpiTransformer
from ..models.contextual_transformer import ContextualTransformer
from ..controllers.homeostasis import HomeostasisModule
from ..utils.metrics import MetricsTracker, accuracy
from ..utils.config import ExperimentConfig, get_default_config
from ..data.dataset_loader import build_dataloaders

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Generic trainer that works with both EpigeneticNetwork and BaselineTransformer.

    Parameters
    ----------
    model      : nn.Module
    config     : ExperimentConfig
    device     : str
    """

    def __init__(
        self,
        model:      nn.Module,
        config:     ExperimentConfig,
        device:     str = "cpu",
    ) -> None:
        self.model   = model.to(device)
        self.config  = config
        self.device  = device
        self.is_epi  = isinstance(model, (EpigeneticNetwork, EpiTransformer, ContextualTransformer))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = config.training.learning_rate,
            weight_decay = config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = config.training.epochs,
            eta_min = config.training.learning_rate * 0.1,
        )

        self.homeostasis = HomeostasisModule(
            stress_threshold  = config.training.stress_threshold,
            adaptation_factor = config.training.lr_adaptation_factor,
            enabled           = config.training.homeostasis_enabled,
        )

        self.metrics = MetricsTracker()
        self._base_lr = config.training.learning_rate

        # Per-step epigenetic state (only used for EpigeneticNetwork)
        self._e_t: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one full training epoch. Returns metric summary dict."""
        self.model.train()
        self._e_t = None    # reset state at start of each epoch

        for batch_idx, (token_ids, labels) in enumerate(loader):
            token_ids = token_ids.to(self.device)
            labels    = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.is_epi:
                # Reset state if batch size changed (last smaller batch)
                if self._e_t is not None and self._e_t.size(0) != token_ids.size(0):
                    self._e_t = None
                logits, self._e_t, info = self.model(
                    token_ids, e_t=self._e_t, update_memory=True
                )
                self._e_t = self._e_t.detach()  # detach to prevent BPTT across batches

                # Homeostasis: compute mean gate activation
                gate_vals = info["gate_values"]
                if gate_vals:
                    flat_parts = []
                    for layer_gates in gate_vals:
                        if isinstance(layer_gates, torch.Tensor):
                            flat_parts.append(layer_gates.flatten())
                        else:
                            for g_head in layer_gates:
                                flat_parts.append(g_head.flatten())
                    all_gates = torch.cat(flat_parts)
                    mean_gate = all_gates.mean().item()
                    lr_scale  = self.homeostasis.update(mean_gate)
                    self.homeostasis.apply_to_optimizer(
                        self.optimizer, self._base_lr, lr_scale
                    )
            else:
                logits = self.model(token_ids)

            loss = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.clip_grad_norm,
            )
            self.optimizer.step()

            acc = accuracy(logits.detach(), labels)
            self.metrics.update(loss=loss.item(), accuracy=acc)

        self.scheduler.step()
        return self.metrics.epoch_end()

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Run evaluation on a DataLoader. Returns metric summary dict."""
        self.model.eval()
        eval_metrics = MetricsTracker()
        e_t = None

        for token_ids, labels in loader:
            token_ids = token_ids.to(self.device)
            labels    = labels.to(self.device)

            if self.is_epi:
                if e_t is not None and e_t.size(0) != token_ids.size(0):
                    e_t = None
                logits, e_t, _ = self.model(token_ids, e_t=e_t, update_memory=False)
                e_t = e_t.detach()
            else:
                logits = self.model(token_ids)

            loss = self.criterion(logits, labels)
            acc  = accuracy(logits, labels)
            eval_metrics.update(loss=loss.item(), accuracy=acc)

        return eval_metrics.epoch_end()

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        task_name:    str = "task",
        verbose:      bool = True,
    ) -> Dict[str, list]:
        """
        Train for config.training.epochs epochs.

        Returns
        -------
        history : dict with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        """
        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        for epoch in range(1, self.config.training.epochs + 1):
            t0 = time.time()
            train_stats = self.train_epoch(train_loader)
            val_stats   = self.evaluate(val_loader)
            elapsed     = time.time() - t0

            history["train_loss"].append(train_stats["loss"])
            history["train_acc"].append(train_stats["accuracy"])
            history["val_loss"].append(val_stats["loss"])
            history["val_acc"].append(val_stats["accuracy"])

            if verbose:
                print(
                    f"[{task_name}] Epoch {epoch:02d}/{self.config.training.epochs} | "
                    f"train_loss={train_stats['loss']:.4f} | "
                    f"train_acc={train_stats['accuracy']:.4f} | "
                    f"val_loss={val_stats['loss']:.4f} | "
                    f"val_acc={val_stats['accuracy']:.4f} | "
                    f"t={elapsed:.1f}s"
                )

        return history

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """Save model weights + optimizer state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "homeostasis_summary":  self.homeostasis.summary(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_model(args, vocab_size: int) -> nn.Module:
    """Construct model based on CLI args."""
    if args.model == "epinet":
        return EpigeneticNetwork(
            vocab_size       = vocab_size,
            embed_dim        = 64,
            hidden_dims      = [128, 64],
            epigenetic_dim   = 32,
            num_classes      = 2,
            memory_size      = 64,
            memory_dim       = 64,
            memory_lambda    = 0.1,
            epigenetic_alpha = 0.3,
            num_heads        = 4,
            dropout          = 0.1,
            max_seq_len      = args.max_seq_len,
        )
    elif args.model == "transformer":
        return BaselineTransformer(
            vocab_size  = vocab_size,
            embed_dim   = 64,
            num_heads   = 4,
            num_layers  = 2,
            num_classes = 2,
            max_seq_len = args.max_seq_len,
            dropout     = 0.1,
        )
    else:
        raise ValueError(f"Unknown model '{args.model}'")


def main():
    parser = argparse.ArgumentParser(description="Train EpiNet or Baseline Transformer")
    parser.add_argument("--model",       default="epinet",     choices=["epinet", "transformer"])
    parser.add_argument("--task",        default="sentiment",  choices=["sentiment", "continual_a", "continual_b"])
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--n_samples",   type=int,   default=2000)
    parser.add_argument("--max_seq_len", type=int,   default=64)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--save_dir",    default="checkpoints")
    parser.add_argument("--verbose",     action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"\n{'='*60}")
    print(f"  Training: {args.model.upper()} on task={args.task}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, vocab = build_dataloaders(
        task       = args.task,
        n_samples  = args.n_samples,
        max_len    = args.max_seq_len,
        batch_size = args.batch_size,
        seed       = args.seed,
    )
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

    # Model
    model = build_model(args, len(vocab))
    print(f"Model parameters: {model.num_parameters():,}\n")

    # Config
    cfg = get_default_config()
    cfg.training.epochs        = args.epochs
    cfg.training.learning_rate = args.lr
    cfg.training.batch_size    = args.batch_size

    # Train
    trainer = Trainer(model, cfg)
    history = trainer.fit(train_loader, val_loader, task_name=args.task)

    print(f"\nFinal val accuracy: {history['val_acc'][-1]:.4f}")

    # Save
    ckpt_path = os.path.join(args.save_dir, f"{args.model}_{args.task}.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
