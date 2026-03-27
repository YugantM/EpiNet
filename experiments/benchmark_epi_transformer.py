"""
EpiTransformer Benchmark
=========================

Runs EpiTransformer head-to-head against:
  - EpigeneticNetwork  (original)
  - BaselineTransformer (standard, no e_t)
  - BiLSTM
  - MLP

on BOTH datasets:
  1. Synthetic sentiment / topic classification (dataset_loader.py)
  2. Q&A answer selection (qa_dataset.py)

Usage
-----
    python -m experiments.benchmark_epi_transformer
    python -m experiments.benchmark_epi_transformer --epochs 15 --dataset both
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epi_transformer import EpiTransformer
from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer
from src.models.baselines import BiLSTMClassifier, MLPClassifier
from src.data.dataset_loader import build_dataloaders
from src.data.qa_dataset import build_qa_loaders
from src.utils.config import get_default_config
from src.training.train import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

def make_all_models(vocab_size: int, max_seq_len: int) -> Dict[str, nn.Module]:
    return {
        "EpiTransformer": EpiTransformer(
            vocab_size=vocab_size, embed_dim=64, num_heads=4,
            num_layers=2, epigenetic_dim=32, num_classes=2,
            inner_dim=256, memory_size=64, memory_dim=64,
            epigenetic_alpha=0.3, memory_lambda=0.1,
            skip_weight=0.1, dropout=0.1, max_seq_len=max_seq_len,
        ),
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
# Training + evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:       int,
    lr:           float,
) -> Dict:
    cfg = get_default_config()
    cfg.training.epochs = epochs
    cfg.training.learning_rate = lr
    trainer = Trainer(model, cfg)

    val_acc_hist, times = [], []
    for _ in range(epochs):
        t0 = time.time()
        trainer.train_epoch(train_loader)
        times.append(time.time() - t0)
        val_acc_hist.append(trainer.evaluate(val_loader)["accuracy"])

    return {
        "val_accuracy":   val_acc_hist[-1],
        "best_accuracy":  max(val_acc_hist),
        "time_per_epoch": sum(times) / len(times),
        "val_acc_history": val_acc_hist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context sensitivity (EpiTransformer and EpiNet only)
# ─────────────────────────────────────────────────────────────────────────────

def context_sensitivity(model: nn.Module, loader: DataLoader) -> float:
    if not hasattr(model, "epigenetic_dim"):
        return 0.0
    model.eval()
    flip, total = 0, 0
    with torch.no_grad():
        for token_ids, _ in loader:
            B = token_ids.size(0)
            e_pos = torch.ones(B, model.epigenetic_dim)
            e_neg = torch.full((B, model.epigenetic_dim), -1.0)
            p1 = model(token_ids, e_t=e_pos, update_memory=False)[0].argmax(-1)
            p2 = model(token_ids, e_t=e_neg, update_memory=False)[0].argmax(-1)
            flip  += (p1 != p2).sum().item()
            total += B
    return flip / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Run one dataset
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(
    name:         str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    vocab_size:   int,
    max_seq_len:  int,
    epochs:       int,
    lr:           float,
    seed:         int,
) -> Dict:
    torch.manual_seed(seed)
    models = make_all_models(vocab_size, max_seq_len)

    print(f"\n  {'Model':<22} {'Params':>8} {'ValAcc':>7} {'BestAcc':>8} "
          f"{'CtxSens':>8} {'s/ep':>6}")
    print("  " + "─" * 64)

    results = {}
    for model_name, model in models.items():
        r = train_and_eval(model, train_loader, val_loader, epochs, lr)
        r["params"]             = model.num_parameters()
        r["context_sensitivity"] = context_sensitivity(model, val_loader)

        print(
            f"  {model_name:<22} {r['params']:>8,} "
            f"{r['val_accuracy']:>7.4f} {r['best_accuracy']:>8.4f} "
            f"{r['context_sensitivity']:>8.4f} {r['time_per_epoch']:>5.2f}s"
        )
        results[model_name] = r

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    all_results = {}

    # ── Dataset 1: Sentiment / Topic ──────────────────────────────────────
    if args.dataset in ("sentiment", "both"):
        print("\n" + "=" * 68)
        print("  DATASET 1: Synthetic Sentiment/Topic Classification")
        print("=" * 68)

        cfg = get_default_config()
        train_loader, val_loader, vocab = build_dataloaders(
            task="sentiment", n_samples=1000, max_len=128,
            batch_size=32, vocab_size=5000, seed=args.seed,
        )
        vocab_size  = len(vocab)
        max_seq_len = 128

        print(f"  Vocab: {vocab_size}  |  Max seq len: {max_seq_len}  "
              f"|  Epochs: {args.epochs}")

        all_results["sentiment"] = run_dataset(
            "sentiment", train_loader, val_loader,
            vocab_size, max_seq_len, args.epochs, args.lr, args.seed,
        )

    # ── Dataset 2: Q&A Answer Selection ──────────────────────────────────
    if args.dataset in ("qa", "both"):
        print("\n" + "=" * 68)
        print("  DATASET 2: Q&A Answer Selection (80 questions, 320 examples)")
        print("=" * 68)

        train_loader, val_loader, vocab, meta = build_qa_loaders(
            batch_size=16, max_len=64, val_split=0.2, seed=args.seed,
        )
        vocab_size  = len(vocab)
        max_seq_len = 64

        print(f"  Vocab: {vocab_size}  |  Train: {meta['n_train']}  "
              f"|  Val: {meta['n_val']}  |  Epochs: {args.epochs}")

        all_results["qa"] = run_dataset(
            "qa", train_loader, val_loader,
            vocab_size, max_seq_len, args.epochs, args.lr, args.seed,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SUMMARY — EpiTransformer vs Baselines")
    print("=" * 68)

    for ds_name, results in all_results.items():
        print(f"\n  [{ds_name.upper()}]")
        et = results.get("EpiTransformer", {})
        for model_name, r in results.items():
            delta = r["val_accuracy"] - et["val_accuracy"]
            marker = " ◄ NEW" if model_name == "EpiTransformer" else (
                f"  Δ={delta:+.4f} vs EpiTransformer"
            )
            print(f"    {model_name:<22} {r['val_accuracy']:.4f}{marker}")

    print("\n  Columns: ValAcc=final-epoch  BestAcc=best-epoch  "
          "CtxSens=prediction flip under e_t ±1")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    saveable = {}
    for ds, res in all_results.items():
        saveable[ds] = {
            k: {kk: vv for kk, vv in v.items() if kk != "val_acc_history"}
            for k, v in res.items()
        }
    out_path = "results/benchmark_epi_transformer.json"
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    return all_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=12)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--dataset", choices=["sentiment", "qa", "both"], default="both")
    p.add_argument("--seed",    type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
