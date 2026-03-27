"""
Comprehensive Benchmark
=======================

Evaluates four models side-by-side on three tasks using a single synthetic
dataset so all numbers are directly comparable.

Models
------
  1. EpigeneticNetwork   (ours)
  2. BaselineTransformer
  3. BiLSTMClassifier
  4. MLPClassifier

Metrics collected per model
---------------------------
  Task-A accuracy      : sentiment classification val accuracy after training
  Task-B accuracy      : topic classification val accuracy after training
  Forgetting (F)       : acc_A_before_B − acc_A_after_B  (lower = better)
  Adaptation speed     : epochs to reach 70% val accuracy on Task A
  Context sensitivity  : fraction of samples that flip under different e_0
                         (EpiNet only; 0 for stateless models)
  Parameters           : total trainable parameter count
  Time/epoch (s)       : wall-clock seconds per training epoch on CPU

Usage
-----
    python -m experiments.benchmark
    python -m experiments.benchmark --epochs 5 --n_samples 2000 --seed 42
"""

import argparse
import json
import os
import sys
import time
import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer
from src.models.baselines import MLPClassifier, BiLSTMClassifier
from src.data.dataset_loader import build_dataloaders
from src.utils.metrics import accuracy, forgetting_score, adaptation_speed
from src.utils.config import get_default_config
from src.training.train import Trainer
from src.training.evaluate import evaluate_model


# ---------------------------------------------------------------------------
# Model factory — all models matched to a similar parameter budget (~85-115k)
# ---------------------------------------------------------------------------

def make_models(vocab_size: int, max_seq_len: int) -> Dict[str, nn.Module]:
    return {
        "EpigeneticNetwork": EpigeneticNetwork(
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
            max_seq_len      = max_seq_len,
        ),
        "Transformer": BaselineTransformer(
            vocab_size  = vocab_size,
            embed_dim   = 64,
            num_heads   = 4,
            num_layers  = 2,
            num_classes = 2,
            max_seq_len = max_seq_len,
            dropout     = 0.1,
        ),
        "BiLSTM": BiLSTMClassifier(
            vocab_size  = vocab_size,
            embed_dim   = 64,
            hidden_dim  = 64,
            num_layers  = 2,
            num_classes = 2,
            dropout     = 0.1,
        ),
        "MLP": MLPClassifier(
            vocab_size  = vocab_size,
            embed_dim   = 64,
            hidden_dim  = 128,
            num_classes = 2,
            dropout     = 0.1,
        ),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    verbose: bool = False,
) -> Tuple[List[float], float]:
    """
    Train model for `epochs` epochs.

    Returns
    -------
    val_acc_per_epoch : list of float
    seconds_per_epoch : float  (mean wall-clock time)
    """
    cfg = get_default_config()
    cfg.training.epochs        = epochs
    cfg.training.learning_rate = lr

    trainer = Trainer(model, cfg)
    epoch_times = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        trainer.train_epoch(train_loader)
        epoch_times.append(time.time() - t0)
        if verbose:
            stats = trainer.evaluate(val_loader)
            print(f"  epoch {epoch}/{epochs}  val_acc={stats['accuracy']:.4f}")

    # Collect per-epoch val accuracies
    # (we re-run eval once per epoch for adaptation speed; collect all)
    cfg2 = get_default_config()
    cfg2.training.epochs = epochs
    trainer2 = Trainer(copy.deepcopy(model.__class__.__new__(model.__class__)), cfg2)
    # Easier: just re-evaluate the trained model on val loader
    final_stats = evaluate_model(model, val_loader)

    # For adaptation speed we need per-epoch accuracy — retrain fresh model
    # tracking each epoch
    return epoch_times, final_stats["accuracy"]


def train_and_track(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
) -> Tuple[List[float], List[float]]:
    """
    Train and return (epoch_times, val_acc_per_epoch).
    """
    cfg = get_default_config()
    cfg.training.epochs        = epochs
    cfg.training.learning_rate = lr
    trainer = Trainer(model, cfg)

    epoch_times  = []
    val_acc_hist = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        trainer.train_epoch(train_loader)
        epoch_times.append(time.time() - t0)
        stats = trainer.evaluate(val_loader)
        val_acc_hist.append(stats["accuracy"])

    return epoch_times, val_acc_hist


# ---------------------------------------------------------------------------
# Continual learning benchmark
# ---------------------------------------------------------------------------

def run_continual(
    model_name: str,
    model: nn.Module,
    train_a: DataLoader,
    val_a: DataLoader,
    train_b: DataLoader,
    val_b: DataLoader,
    epochs: int,
    lr: float,
) -> Dict:
    print(f"  [{model_name}] Task A training...")
    epoch_times_a, val_acc_a = train_and_track(model, train_a, val_a, epochs, lr)

    acc_a_before = val_acc_a[-1]
    adapt_ep     = adaptation_speed(val_acc_a, target_acc=0.70)

    print(f"    Task-A final acc: {acc_a_before:.4f}  "
          f"adapt_speed: {adapt_ep} epoch(s)")

    print(f"  [{model_name}] Task B training (continual)...")
    cfg = get_default_config()
    cfg.training.epochs        = epochs
    cfg.training.learning_rate = lr
    trainer_b = Trainer(model, cfg)
    for _ in range(epochs):
        trainer_b.train_epoch(train_b)

    acc_a_after = evaluate_model(model, val_a)["accuracy"]
    acc_b_final = evaluate_model(model, val_b)["accuracy"]
    forget      = forgetting_score(acc_a_before, acc_a_after)

    print(f"    Task-A after B: {acc_a_after:.4f}  "
          f"Task-B: {acc_b_final:.4f}  F={forget:.4f}")

    return {
        "task_a_acc":     acc_a_before,
        "task_b_acc":     acc_b_final,
        "acc_a_after":    acc_a_after,
        "forgetting":     forget,
        "adapt_speed":    adapt_ep,
        "time_per_epoch": sum(epoch_times_a) / len(epoch_times_a),
        "params":         model.num_parameters(),
        "val_acc_hist_a": val_acc_a,
    }


# ---------------------------------------------------------------------------
# Context sensitivity (EpiNet only)
# ---------------------------------------------------------------------------

def measure_context_sensitivity(model: nn.Module, val_loader: DataLoader) -> float:
    if not isinstance(model, EpigeneticNetwork):
        return 0.0

    model.eval()
    flip_count, total = 0, 0
    with torch.no_grad():
        for token_ids, _ in val_loader:
            B = token_ids.size(0)
            # Two epigenetic states: all-positive vs all-negative
            e_pos = torch.ones(B, model.epigenetic_dim)
            e_neg = torch.full((B, model.epigenetic_dim), -1.0)
            preds_pos = model(token_ids, e_t=e_pos, update_memory=False)[0].argmax(-1)
            preds_neg = model(token_ids, e_t=e_neg, update_memory=False)[0].argmax(-1)
            flip_count += (preds_pos != preds_neg).sum().item()
            total      += B
    return flip_count / max(total, 1)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args) -> Dict:
    torch.manual_seed(args.seed)

    print("\n" + "=" * 68)
    print("  EPINET BENCHMARK — All models on synthetic dataset")
    print("=" * 68)

    # ── Data ──────────────────────────────────────────────────────────
    print("\n[1/3] Preparing datasets...")
    train_a, val_a, vocab = build_dataloaders(
        task="continual_a", n_samples=args.n_samples,
        max_len=args.max_seq_len, batch_size=args.batch_size, seed=args.seed,
    )
    train_b, val_b, _ = build_dataloaders(
        task="continual_b", n_samples=args.n_samples,
        max_len=args.max_seq_len, batch_size=args.batch_size, seed=args.seed,
    )
    vocab_size = len(vocab)
    print(f"  Vocab: {vocab_size} | Train: {len(train_a.dataset)} | "
          f"Val: {len(val_a.dataset)} | Seq len: {args.max_seq_len}")

    # ── Models ────────────────────────────────────────────────────────
    print("\n[2/3] Building models...")
    all_models = make_models(vocab_size, args.max_seq_len)
    for name, m in all_models.items():
        print(f"  {name:<22} {m.num_parameters():>8,} params")

    # ── Benchmark loop ────────────────────────────────────────────────
    print(f"\n[3/3] Running benchmark ({args.epochs} epochs × 2 tasks each)...")
    results = {}

    for model_name, model in all_models.items():
        print(f"\n{'─' * 50}")
        print(f"  {model_name}")
        print(f"{'─' * 50}")

        res = run_continual(
            model_name, model,
            train_a, val_a, train_b, val_b,
            epochs=args.epochs, lr=args.lr,
        )
        res["context_sensitivity"] = measure_context_sensitivity(model, val_a)
        results[model_name] = res

    # ── Print results table ───────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BENCHMARK RESULTS")
    print("=" * 68)

    header = (
        f"\n{'Model':<22} {'Params':>8} {'Task-A':>8} {'Task-B':>8} "
        f"{'Forget':>8} {'Adapt':>6} {'CtxSens':>8} {'s/epoch':>8}"
    )
    print(header)
    print("-" * len(header.strip()))

    for name, r in results.items():
        print(
            f"{name:<22} {r['params']:>8,} "
            f"{r['task_a_acc']:>8.4f} "
            f"{r['task_b_acc']:>8.4f} "
            f"{r['forgetting']:>8.4f} "
            f"{r['adapt_speed']:>6} "
            f"{r['context_sensitivity']:>8.4f} "
            f"{r['time_per_epoch']:>8.2f}s"
        )

    print("\nLegend:")
    print("  Task-A  : val accuracy on sentiment task after training")
    print("  Task-B  : val accuracy on topic task after continual training")
    print("  Forget  : acc_A_before - acc_A_after  (lower = better retention)")
    print("  Adapt   : epochs to first reach 70% val accuracy (lower = faster)")
    print("  CtxSens : fraction of samples with flipped prediction under")
    print("            different epigenetic state (EpiNet only)")
    print("  s/epoch : mean CPU wall-clock seconds per training epoch")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    saveable = {
        k: {kk: vv for kk, vv in v.items() if kk != "val_acc_hist_a"}
        for k, v in results.items()
    }
    with open("results/benchmark.json", "w") as f:
        json.dump(saveable, f, indent=2)
    print("\nResults saved to results/benchmark.json")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="EpiNet benchmark")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--n_samples",   type=int,   default=2000)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max_seq_len", type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
