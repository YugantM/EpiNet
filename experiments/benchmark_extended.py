"""
Extended benchmark: adds EpiNet+Boundary (memory reset at task boundary)
and per-epoch accuracy curves for all models.

Run:
    python -m experiments.benchmark_extended
"""

import json, os, sys, time, copy, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer
from src.models.baselines import MLPClassifier, BiLSTMClassifier
from src.data.dataset_loader import build_dataloaders
from src.utils.metrics import accuracy, forgetting_score, adaptation_speed
from src.utils.config import get_default_config
from src.training.train import Trainer
from src.training.evaluate import evaluate_model


def make_epinet(vocab_size, max_seq_len):
    return EpigeneticNetwork(
        vocab_size=vocab_size, embed_dim=64, hidden_dims=[128, 64],
        epigenetic_dim=32, num_classes=2, memory_size=64, memory_dim=64,
        memory_lambda=0.1, epigenetic_alpha=0.3, num_heads=4,
        dropout=0.1, max_seq_len=max_seq_len,
    )


def train_track(model, train_loader, val_loader, epochs, lr):
    """Return (val_acc_history, mean_seconds_per_epoch)."""
    cfg = get_default_config()
    cfg.training.epochs = epochs
    cfg.training.learning_rate = lr
    trainer = Trainer(model, cfg)
    times, accs = [], []
    for _ in range(epochs):
        t0 = time.time()
        trainer.train_epoch(train_loader)
        times.append(time.time() - t0)
        accs.append(trainer.evaluate(val_loader)["accuracy"])
    return accs, sum(times) / len(times)


def run_extended(args):
    torch.manual_seed(args.seed)
    print("\n" + "=" * 70)
    print("  EXTENDED BENCHMARK")
    print("=" * 70)

    train_a, val_a, vocab = build_dataloaders(
        task="continual_a", n_samples=args.n_samples,
        max_len=args.max_seq_len, batch_size=args.batch_size, seed=args.seed,
    )
    train_b, val_b, _ = build_dataloaders(
        task="continual_b", n_samples=args.n_samples,
        max_len=args.max_seq_len, batch_size=args.batch_size, seed=args.seed,
    )
    vocab_size = len(vocab)

    results = {}

    configs = {
        "EpiNet (no boundary)": make_epinet(vocab_size, args.max_seq_len),
        "EpiNet+Boundary":      make_epinet(vocab_size, args.max_seq_len),
        "Transformer":          BaselineTransformer(
            vocab_size=vocab_size, embed_dim=64, num_heads=4,
            num_layers=2, num_classes=2, max_seq_len=args.max_seq_len, dropout=0.1),
        "BiLSTM":               BiLSTMClassifier(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=64,
            num_layers=2, num_classes=2, dropout=0.1),
        "MLP":                  MLPClassifier(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=128,
            num_classes=2, dropout=0.1),
    }

    for name, model in configs.items():
        print(f"\n── {name} ({model.num_parameters():,} params) ──")

        # Task A
        val_acc_a, t_per_ep = train_track(model, train_a, val_a, args.epochs, args.lr)
        acc_a_before = val_acc_a[-1]
        adapt_ep = adaptation_speed(val_acc_a, target_acc=0.70)
        print(f"  Task-A: {acc_a_before:.4f}  adapt: {adapt_ep} ep  "
              f"{t_per_ep:.2f}s/ep")

        # Task boundary handling for EpiNet+Boundary
        if name == "EpiNet+Boundary":
            model.reset_memory()
            print("  [memory reset at boundary]")

        # Task B (continual)
        cfg = get_default_config()
        cfg.training.epochs = args.epochs
        cfg.training.learning_rate = args.lr
        trainer_b = Trainer(model, cfg)
        for _ in range(args.epochs):
            trainer_b.train_epoch(train_b)

        acc_a_after = evaluate_model(model, val_a)["accuracy"]
        acc_b_final = evaluate_model(model, val_b)["accuracy"]
        forget = forgetting_score(acc_a_before, acc_a_after)
        print(f"  Task-B: {acc_b_final:.4f}  "
              f"Task-A after: {acc_a_after:.4f}  F={forget:.4f}")

        results[name] = {
            "params":         model.num_parameters(),
            "task_a_acc":     acc_a_before,
            "task_b_acc":     acc_b_final,
            "acc_a_after":    acc_a_after,
            "forgetting":     forget,
            "adapt_speed":    adapt_ep,
            "time_per_epoch": t_per_ep,
            "val_acc_hist_a": val_acc_a,
        }

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    hdr = f"\n{'Model':<24} {'Params':>8} {'Task-A':>7} {'Task-B':>7} {'Forget':>7} {'Adapt':>6} {'s/ep':>6}"
    print(hdr)
    print("─" * len(hdr.strip()))
    for name, r in results.items():
        marker = " ★" if name == "EpiNet+Boundary" else ""
        print(f"{name:<24} {r['params']:>8,} {r['task_a_acc']:>7.4f} "
              f"{r['task_b_acc']:>7.4f} {r['forgetting']:>7.4f} "
              f"{r['adapt_speed']:>6} {r['time_per_epoch']:>6.2f}s{marker}")

    os.makedirs("results", exist_ok=True)
    saveable = {k: {kk: vv for kk, vv in v.items() if kk != "val_acc_hist_a"}
                for k, v in results.items()}
    with open("results/benchmark_extended.json", "w") as f:
        json.dump(saveable, f, indent=2)
    print("\nSaved to results/benchmark_extended.json")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=8)
    p.add_argument("--n_samples",   type=int,   default=3000)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max_seq_len", type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_extended(parse_args())
