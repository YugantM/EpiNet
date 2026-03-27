"""
Continual Learning Experiment
==============================

Goal: Evaluate how well EpigeneticNetwork retains knowledge of Task A
after being trained on Task B, compared to a Baseline Transformer.

Protocol
--------
1. Train both models on Task A (sentiment classification).
2. Evaluate both models on Task A  → record acc_A_before.
3. Continue training both models on Task B (topic classification).
4. Evaluate both models on Task A  → record acc_A_after.
5. Compute:
       forgetting = acc_A_before - acc_A_after
       (lower = better)

Hypothesis
----------
EpigeneticNetwork should exhibit lower forgetting because:
  a) The memory store retains Task A representations.
  b) The epigenetic gate can partially suppress Task-B-specific pathways,
     preserving Task-A-trained pathways.
  c) Homeostasis prevents catastrophic weight update magnitudes.

Usage
-----
    python -m experiments.continual_learning
    python -m experiments.continual_learning --epochs 5 --n_samples 2000
"""

import argparse
import sys
import os
import json
import time
import torch

# Allow running as a script from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer
from src.training.train import Trainer
from src.training.evaluate import evaluate_model, comparative_summary
from src.utils.config import get_default_config
from src.utils.metrics import forgetting_score
from src.data.dataset_loader import build_dataloaders, make_continual_tasks


def build_epinet(vocab_size: int, max_seq_len: int) -> EpigeneticNetwork:
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
        max_seq_len      = max_seq_len,
    )


def build_transformer(vocab_size: int, max_seq_len: int) -> BaselineTransformer:
    return BaselineTransformer(
        vocab_size  = vocab_size,
        embed_dim   = 64,
        num_heads   = 4,
        num_layers  = 2,
        num_classes = 2,
        max_seq_len = max_seq_len,
        dropout     = 0.1,
    )


def run_continual_experiment(args) -> dict:
    torch.manual_seed(args.seed)

    print("\n" + "=" * 65)
    print("  CONTINUAL LEARNING EXPERIMENT")
    print("=" * 65)

    # ── Data ─────────────────────────────────────────────────────────
    print("\n[1/6] Preparing datasets...")
    train_a, val_a, vocab = build_dataloaders(
        task="continual_a", n_samples=args.n_samples,
        max_len=args.max_seq_len, batch_size=args.batch_size, seed=args.seed,
    )
    train_b, val_b, _     = build_dataloaders(
        task="continual_b", n_samples=args.n_samples,
        max_len=args.max_seq_len, batch_size=args.batch_size, seed=args.seed,
    )
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Task A — train: {len(train_a.dataset)} | val: {len(val_a.dataset)}")
    print(f"  Task B — train: {len(train_b.dataset)} | val: {len(val_b.dataset)}")

    # ── Build models ─────────────────────────────────────────────────
    print("\n[2/6] Building models...")
    epinet      = build_epinet(vocab_size, args.max_seq_len)
    transformer = build_transformer(vocab_size, args.max_seq_len)
    print(f"  EpigeneticNetwork params: {epinet.num_parameters():,}")
    print(f"  BaselineTransformer params: {transformer.num_parameters():,}")

    cfg = get_default_config()
    cfg.training.epochs        = args.epochs
    cfg.training.learning_rate = args.lr
    cfg.training.batch_size    = args.batch_size

    results = {}

    for model_name, model in [("EpigeneticNetwork", epinet), ("BaselineTransformer", transformer)]:
        print(f"\n{'─'*65}")
        print(f"  Model: {model_name}")
        print(f"{'─'*65}")

        trainer = Trainer(model, cfg)

        # Step 1: Train on Task A
        print(f"\n[3/6] Training on Task A ({args.epochs} epochs)...")
        t0 = time.time()
        history_a = trainer.fit(train_a, val_a, task_name="TaskA", verbose=True)
        print(f"  Task A training: {time.time()-t0:.1f}s")

        # Step 2: Evaluate on Task A  (before Task B)
        print(f"\n[4/6] Evaluating on Task A (before Task B)...")
        stats_a_before = evaluate_model(model, val_a)
        acc_a_before   = stats_a_before["accuracy"]
        print(f"  Task A accuracy (before B): {acc_a_before:.4f}")

        # Step 3: Train on Task B (continual; no weight reset)
        print(f"\n[5/6] Continual training on Task B ({args.epochs} epochs)...")
        # Reset epigenetic state / optionally reset memory for ablation
        if isinstance(model, EpigeneticNetwork) and not args.reset_memory:
            print("  [EpiNet] Memory retained across tasks.")
        elif isinstance(model, EpigeneticNetwork) and args.reset_memory:
            model.reset_memory()
            print("  [EpiNet] Memory cleared before Task B.")

        # Create a new trainer to avoid LR scheduler state carry-over
        trainer_b = Trainer(model, cfg)
        history_b = trainer_b.fit(train_b, val_b, task_name="TaskB", verbose=True)

        # Step 4: Evaluate on Task A  (after Task B)
        print(f"\n[6/6] Evaluating on Task A (after Task B)...")
        stats_a_after = evaluate_model(model, val_a)
        acc_a_after   = stats_a_after["accuracy"]
        print(f"  Task A accuracy (after  B): {acc_a_after:.4f}")

        stats_b_after = evaluate_model(model, val_b)
        acc_b_final   = stats_b_after["accuracy"]
        print(f"  Task B accuracy (final)   : {acc_b_final:.4f}")

        # Metrics
        forget_score = forgetting_score(acc_a_before, acc_a_after)
        print(f"\n  ► Forgetting score: {forget_score:.4f}  "
              f"({'↓ less forgetting' if forget_score < 0.05 else '↑ significant forgetting'})")

        results[model_name] = {
            "task_a_acc_before": acc_a_before,
            "task_a_acc_after":  acc_a_after,
            "task_b_acc":        acc_b_final,
            "forgetting":        forget_score,
            "stability":         1.0 - abs(forget_score),   # proxy
            "params":            model.num_parameters(),
            "history_a":         history_a,
            "history_b":         history_b,
        }

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    summary_data = {
        name: {
            "task_a_acc":  v["task_a_acc_after"],
            "task_b_acc":  v["task_b_acc"],
            "forgetting":  v["forgetting"],
            "stability":   v["stability"],
            "params":      v["params"],
        }
        for name, v in results.items()
    }
    table = comparative_summary(summary_data)
    print(table)

    # Print forgetting comparison
    epi_forget = results["EpigeneticNetwork"]["forgetting"]
    tf_forget  = results["BaselineTransformer"]["forgetting"]
    improvement = tf_forget - epi_forget
    print(f"\nEpiNet forgetting reduction vs Transformer: {improvement:+.4f}")
    if improvement > 0:
        print("  → EpigeneticNetwork forgot LESS (as hypothesised).")
    else:
        print("  → Transformer forgot less (epigenetic advantage not observed on this run).")

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/continual_learning.json"
    # Remove non-serialisable history tensors for JSON
    saveable = {k: {kk: vv for kk, vv in v.items() if kk not in ("history_a", "history_b")}
                for k, v in results.items()}
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Continual Learning Experiment")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--n_samples",   type=int,   default=2000)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max_seq_len", type=int,   default=64)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--reset_memory", action="store_true",
                        help="Reset EpiNet memory before Task B (ablation)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_continual_experiment(args)
