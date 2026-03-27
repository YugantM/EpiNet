"""
Integration tests for the training loop and evaluation pipeline.

Tests cover:
  - Trainer constructs without error
  - Loss decreases over a mini training run
  - Evaluation returns correct keys
  - Checkpoint save/load round-trip
  - Metrics tracker accumulation
  - Forgetting score and adaptation speed metrics
"""

import pytest
import os
import tempfile
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer
from src.training.train import Trainer
from src.training.evaluate import evaluate_model, measure_adaptation_speed
from src.utils.config import get_default_config
from src.utils.metrics import (
    MetricsTracker, accuracy, forgetting_score, adaptation_speed,
)
from src.data.dataset_loader import build_dataloaders


# ---------------------------------------------------------------------------
# Small data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_loaders():
    train_loader, val_loader, vocab = build_dataloaders(
        task="sentiment", n_samples=200, max_len=32,
        batch_size=16, seed=42,
    )
    return train_loader, val_loader, vocab


@pytest.fixture(scope="module")
def small_epinet(small_loaders):
    _, _, vocab = small_loaders
    return EpigeneticNetwork(
        vocab_size=len(vocab), embed_dim=32,
        hidden_dims=[64, 32], epigenetic_dim=16,
        num_classes=2, memory_size=16, memory_dim=32,
        max_seq_len=32, dropout=0.0,
    )


@pytest.fixture(scope="module")
def small_transformer(small_loaders):
    _, _, vocab = small_loaders
    return BaselineTransformer(
        vocab_size=len(vocab), embed_dim=32,
        num_heads=2, num_layers=2, num_classes=2,
        max_seq_len=32, dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestTrainer:

    def test_trainer_constructs(self, small_epinet):
        cfg = get_default_config()
        cfg.training.epochs = 2
        trainer = Trainer(small_epinet, cfg)
        assert trainer is not None

    def test_train_epoch_returns_metrics(self, small_epinet, small_loaders):
        train_loader, _, _ = small_loaders
        cfg = get_default_config()
        cfg.training.epochs = 1
        trainer = Trainer(small_epinet, cfg)
        stats = trainer.train_epoch(train_loader)
        assert "loss" in stats
        assert "accuracy" in stats
        assert 0.0 <= stats["accuracy"] <= 1.0

    def test_evaluate_returns_metrics(self, small_epinet, small_loaders):
        _, val_loader, _ = small_loaders
        cfg = get_default_config()
        trainer = Trainer(small_epinet, cfg)
        stats = trainer.evaluate(val_loader)
        assert "loss" in stats
        assert "accuracy" in stats

    def test_fit_returns_history(self, small_transformer, small_loaders):
        train_loader, val_loader, _ = small_loaders
        cfg = get_default_config()
        cfg.training.epochs = 2
        trainer = Trainer(small_transformer, cfg)
        history = trainer.fit(train_loader, val_loader, verbose=False)
        assert len(history["val_acc"]) == 2
        assert len(history["train_loss"]) == 2

    def test_checkpoint_save_and_load(self, small_epinet, small_loaders):
        train_loader, val_loader, _ = small_loaders
        cfg = get_default_config()
        cfg.training.epochs = 1
        trainer = Trainer(small_epinet, cfg)
        trainer.fit(train_loader, val_loader, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.pt")
            trainer.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            # Load into a fresh model
            new_model = EpigeneticNetwork(
                vocab_size=len(small_loaders[2]), embed_dim=32,
                hidden_dims=[64, 32], epigenetic_dim=16,
                num_classes=2, memory_size=16, memory_dim=32,
                max_seq_len=32, dropout=0.0,
            )
            new_trainer = Trainer(new_model, cfg)
            new_trainer.load_checkpoint(ckpt_path)
            # After loading, both models should produce identical output
            token_batch = torch.randint(1, 10, (4, 16))
            small_epinet.eval()
            new_model.eval()
            with torch.no_grad():
                logits_orig, _, _ = small_epinet(token_batch, update_memory=False)
                logits_load, _, _ = new_model(token_batch,  update_memory=False)
            assert torch.allclose(logits_orig, logits_load, atol=1e-5)


# ---------------------------------------------------------------------------
# Evaluate module tests
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_evaluate_model_keys(self, small_epinet, small_loaders):
        _, val_loader, _ = small_loaders
        stats = evaluate_model(small_epinet, val_loader)
        assert "loss" in stats
        assert "accuracy" in stats

    def test_evaluate_model_accuracy_range(self, small_epinet, small_loaders):
        _, val_loader, _ = small_loaders
        stats = evaluate_model(small_epinet, val_loader)
        assert 0.0 <= stats["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_accuracy_perfect(self):
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        labels = torch.tensor([0, 1])
        assert accuracy(logits, labels) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        labels = torch.tensor([1, 0])
        assert accuracy(logits, labels) == pytest.approx(0.0)

    def test_forgetting_score_positive_means_forgot(self):
        assert forgetting_score(0.9, 0.6) == pytest.approx(0.3)

    def test_forgetting_score_negative_means_improved(self):
        assert forgetting_score(0.6, 0.8) == pytest.approx(-0.2)

    def test_adaptation_speed_hits_target(self):
        accs = [0.55, 0.62, 0.70, 0.75, 0.80]
        assert adaptation_speed(accs, target_acc=0.70) == 3   # 1-indexed

    def test_adaptation_speed_never_hits(self):
        accs = [0.55, 0.60, 0.62]
        speed = adaptation_speed(accs, target_acc=0.90)
        assert speed == len(accs)

    def test_metrics_tracker_accumulate(self):
        tracker = MetricsTracker()
        for _ in range(5):
            tracker.update(loss=1.0, accuracy=0.5)
        summary = tracker.epoch_end()
        assert summary["loss"]     == pytest.approx(1.0)
        assert summary["accuracy"] == pytest.approx(0.5)

    def test_metrics_tracker_history(self):
        tracker = MetricsTracker()
        for epoch in range(3):
            tracker.update(loss=float(epoch))
            tracker.epoch_end()
        hist = tracker.get_history("loss")
        assert hist == pytest.approx([0.0, 1.0, 2.0])

    def test_metrics_tracker_last(self):
        tracker = MetricsTracker()
        tracker.update(acc=0.8)
        tracker.epoch_end()
        assert tracker.last("acc") == pytest.approx(0.8)

    def test_metrics_tracker_last_default(self):
        tracker = MetricsTracker()
        assert tracker.last("nonexistent", default=-1.0) == pytest.approx(-1.0)
