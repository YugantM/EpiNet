"""
Q&A Answer-Selection Benchmark
================================

Evaluates all four models on the tiny Q&A dataset (80 questions, 320 binary
answer-selection examples).  Mirrors the protocol in benchmark.py so numbers
are directly comparable.

Usage
-----
    python -m experiments.benchmark_qa
    python -m experiments.benchmark_qa --epochs 10 --batch_size 8
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer
from src.models.baselines import MLPClassifier, BiLSTMClassifier
from src.data.qa_dataset import build_qa_loaders
from src.utils.metrics import accuracy, adaptation_speed
from src.utils.config import get_default_config
from src.training.train import Trainer
from src.training.evaluate import evaluate_model


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def make_models(vocab_size: int, max_seq_len: int) -> Dict[str, nn.Module]:
    return {
        "EpigeneticNetwork": EpigeneticNetwork(
            vocab_size=vocab_size, embed_dim=64, hidden_dims=[128, 64],
            epigenetic_dim=32, num_classes=2, memory_size=64, memory_dim=64,
            memory_lambda=0.1, epigenetic_alpha=0.3, num_heads=4,
            dropout=0.1, max_seq_len=max_seq_len,
        ),
        "Transformer": BaselineTransformer(
            vocab_size=vocab_size, embed_dim=64, num_heads=4,
            num_layers=2, num_classes=2, max_seq_len=max_seq_len, dropout=0.1,
        ),
        "BiLSTM": BiLSTMClassifier(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=64,
            num_layers=2, num_classes=2, dropout=0.1,
        ),
        "MLP": MLPClassifier(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=128,
            num_classes=2, dropout=0.1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

def train_and_track(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> Tuple[List[float], List[float]]:
    """Return (epoch_times, val_acc_per_epoch)."""
    cfg = get_default_config()
    cfg.training.epochs = epochs
    cfg.training.learning_rate = lr
    trainer = Trainer(model, cfg)

    epoch_times, val_acc_hist = [], []
    for _ in range(epochs):
        t0 = time.time()
        trainer.train_epoch(train_loader)
        epoch_times.append(time.time() - t0)
        val_acc_hist.append(trainer.evaluate(val_loader)["accuracy"])
    return epoch_times, val_acc_hist


# ─────────────────────────────────────────────────────────────────────────────
# Per-topic breakdown
# ─────────────────────────────────────────────────────────────────────────────

def per_topic_accuracy(
    model: nn.Module,
    val_loader: DataLoader,
) -> Dict[str, float]:
    """Accuracy broken down by topic using the dataset's .topics attribute."""
    model.eval()
    dataset = val_loader.dataset  # QADataset
    topic_correct: Dict[str, int] = {}
    topic_total:   Dict[str, int] = {}

    with torch.no_grad():
        for i in range(len(dataset)):
            token_ids, label = dataset[i]
            topic = dataset.topics[i]
            token_ids = token_ids.unsqueeze(0)
            if isinstance(model, EpigeneticNetwork):
                logits, _, _ = model(token_ids, e_t=None, update_memory=False)
            else:
                logits = model(token_ids)
            pred = logits.argmax(-1).item()
            correct = int(pred == label.item())
            topic_correct[topic] = topic_correct.get(topic, 0) + correct
            topic_total[topic]   = topic_total.get(topic, 0) + 1

    return {
        t: topic_correct[t] / topic_total[t]
        for t in topic_total
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context sensitivity (EpiNet only)
# ─────────────────────────────────────────────────────────────────────────────

def measure_context_sensitivity(model: nn.Module, val_loader: DataLoader) -> float:
    if not isinstance(model, EpigeneticNetwork):
        return 0.0
    model.eval()
    flip, total = 0, 0
    with torch.no_grad():
        for token_ids, _ in val_loader:
            B = token_ids.size(0)
            e_pos = torch.ones(B, model.epigenetic_dim)
            e_neg = torch.full((B, model.epigenetic_dim), -1.0)
            p1 = model(token_ids, e_t=e_pos, update_memory=False)[0].argmax(-1)
            p2 = model(token_ids, e_t=e_neg, update_memory=False)[0].argmax(-1)
            flip  += (p1 != p2).sum().item()
            total += B
    return flip / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_qa_benchmark(args) -> Dict:
    torch.manual_seed(args.seed)

    print("\n" + "=" * 68)
    print("  EPINET Q&A BENCHMARK — Answer Selection")
    print("=" * 68)

    # ── Data ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading Q&A dataset...")
    train_loader, val_loader, vocab, meta = build_qa_loaders(
        batch_size=args.batch_size,
        max_len=args.max_seq_len,
        val_split=0.2,
        seed=args.seed,
    )
    vocab_size = len(vocab)
    print(f"  Questions: 80  |  Examples: {meta['n_total']}  "
          f"(train {meta['n_train']}, val {meta['n_val']})")
    print(f"  Vocab size: {vocab_size}  |  Positive: {meta['n_positive']}  "
          f"Negative: {meta['n_negative']}")
    print(f"  Topics: {', '.join(meta['topics'].keys())}")

    # ── Models ────────────────────────────────────────────────────────
    print("\n[2/3] Building models...")
    all_models = make_models(vocab_size, args.max_seq_len)
    for name, m in all_models.items():
        print(f"  {name:<22} {m.num_parameters():>8,} params")

    # ── Benchmark loop ────────────────────────────────────────────────
    print(f"\n[3/3] Training ({args.epochs} epochs each)...")
    results = {}

    for model_name, model in all_models.items():
        print(f"\n{'─' * 50}")
        print(f"  {model_name}")
        print(f"{'─' * 50}")

        epoch_times, val_acc_hist = train_and_track(
            model, train_loader, val_loader, args.epochs, args.lr,
        )
        final_acc  = val_acc_hist[-1]
        adapt_ep   = adaptation_speed(val_acc_hist, target_acc=0.70)
        t_per_ep   = sum(epoch_times) / len(epoch_times)
        ctx_sens   = measure_context_sensitivity(model, val_loader)
        topic_accs = per_topic_accuracy(model, val_loader)

        print(f"  Val accuracy: {final_acc:.4f}  "
              f"adapt: {adapt_ep} ep  {t_per_ep:.2f}s/ep  ctx_sens: {ctx_sens:.4f}")
        print(f"  Per-topic: " +
              "  ".join(f"{t}={v:.2f}" for t, v in topic_accs.items()))

        results[model_name] = {
            "params":           model.num_parameters(),
            "val_accuracy":     final_acc,
            "adapt_speed":      adapt_ep,
            "time_per_epoch":   t_per_ep,
            "context_sensitivity": ctx_sens,
            "per_topic_accuracy":  topic_accs,
            "val_acc_history":  val_acc_hist,
        }

    # ── Print results table ───────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  Q&A BENCHMARK RESULTS")
    print("=" * 68)

    header = (
        f"\n{'Model':<22} {'Params':>8} {'Val Acc':>8} "
        f"{'Adapt':>6} {'CtxSens':>8} {'s/epoch':>8}"
    )
    print(header)
    print("-" * len(header.strip()))

    for name, r in results.items():
        print(
            f"{name:<22} {r['params']:>8,} "
            f"{r['val_accuracy']:>8.4f} "
            f"{r['adapt_speed']:>6} "
            f"{r['context_sensitivity']:>8.4f} "
            f"{r['time_per_epoch']:>8.2f}s"
        )

    # Baseline accuracy at 25% (random) vs 50% (always-positive)
    print("\n  Baselines: random=0.25  majority=0.50")
    print("  Legend:")
    print("    Val Acc  : answer-selection accuracy on held-out val set")
    print("    Adapt    : epochs to first reach 70% val accuracy")
    print("    CtxSens  : fraction of val samples with flipped prediction")
    print("               under different epigenetic initialisation (EpiNet only)")
    print("    s/epoch  : mean CPU wall-clock seconds per training epoch")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    saveable = {
        k: {kk: vv for kk, vv in v.items() if kk != "val_acc_history"}
        for k, v in results.items()
    }
    out_path = "results/benchmark_qa.json"
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="EpiNet Q&A benchmark")
    p.add_argument("--epochs",      type=int,   default=12)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max_seq_len", type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_qa_benchmark(parse_args())
