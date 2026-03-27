"""
Evaluation utilities for EpiNet experiments.

Provides:
  - per-task accuracy evaluation
  - forgetting score computation
  - adaptation speed measurement
  - stability assessment under perturbation
  - comparative summary table
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import json

from ..utils.metrics import (
    accuracy,
    forgetting_score,
    adaptation_speed,
    stability_index,
    MetricsTracker,
)
from ..models.epigenetic_network import EpigeneticNetwork


@torch.no_grad()
def evaluate_model(
    model:  nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Run full evaluation on a DataLoader.

    Returns
    -------
    {'loss': float, 'accuracy': float}
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    tracker   = MetricsTracker()
    is_epi    = isinstance(model, EpigeneticNetwork)
    e_t: Optional[torch.Tensor] = None

    for token_ids, labels in loader:
        token_ids = token_ids.to(device)
        labels    = labels.to(device)

        if is_epi:
            # Reset e_t if batch size changed (e.g. last smaller batch)
            if e_t is not None and e_t.size(0) != token_ids.size(0):
                e_t = None
            logits, e_t, _ = model(token_ids, e_t=e_t, update_memory=False)
            e_t = e_t.detach()
        else:
            logits = model(token_ids)

        loss = criterion(logits, labels)
        acc  = accuracy(logits, labels)
        tracker.update(loss=loss.item(), accuracy=acc)

    return tracker.epoch_end()


def evaluate_forgetting(
    model:        nn.Module,
    task_a_loader: DataLoader,
    task_b_loader: DataLoader,
    device:       str = "cpu",
) -> Dict[str, float]:
    """
    Compute the forgetting score for continual learning:

    1. Evaluate model on Task A (before Task B training)  → acc_A_before
    2. [External: train on Task B]
    3. Evaluate model on Task A (after Task B training)   → acc_A_after
    4. forgetting = acc_A_before - acc_A_after

    This function only evaluates — the training step must be done separately.

    Returns
    -------
    {
        'task_a_acc': float,
        'task_b_acc': float,
    }
    """
    stats_a = evaluate_model(model, task_a_loader, device)
    stats_b = evaluate_model(model, task_b_loader, device)
    return {
        "task_a_acc": stats_a["accuracy"],
        "task_b_acc": stats_b["accuracy"],
    }


def measure_adaptation_speed(
    model:       nn.Module,
    loader:      DataLoader,
    n_epochs:    int    = 10,
    lr:          float  = 1e-3,
    target_acc:  float  = 0.70,
    device:      str    = "cpu",
    reset_memory: bool  = False,
) -> Dict:
    """
    Measure how quickly the model adapts to a new task by fine-tuning
    and recording per-epoch accuracy.

    Returns
    -------
    {
        'acc_per_epoch': list of float,
        'adaptation_epoch': int,
        'final_acc': float,
    }
    """
    if reset_memory and isinstance(model, EpigeneticNetwork):
        model.reset_memory()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    is_epi    = isinstance(model, EpigeneticNetwork)

    acc_per_epoch = []

    for epoch in range(n_epochs):
        model.train()
        e_t = None
        for token_ids, labels in loader:
            token_ids = token_ids.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            if is_epi:
                if e_t is not None and e_t.size(0) != token_ids.size(0):
                    e_t = None
                logits, e_t, _ = model(token_ids, e_t=e_t, update_memory=True)
                e_t = e_t.detach()
            else:
                logits = model(token_ids)

            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval at end of epoch
        stats = evaluate_model(model, loader, device)
        acc_per_epoch.append(stats["accuracy"])

    adapt_epoch = adaptation_speed(acc_per_epoch, target_acc)

    return {
        "acc_per_epoch":    acc_per_epoch,
        "adaptation_epoch": adapt_epoch,
        "final_acc":        acc_per_epoch[-1],
    }


def run_stability_test(
    model:     nn.Module,
    loader:    DataLoader,
    noise_std: float = 0.01,
    n_trials:  int   = 20,
    device:    str   = "cpu",
) -> float:
    """
    Measure output stability by adding Gaussian noise to the *embedded*
    representation and checking consistency of predictions.

    Returns
    -------
    stability : float in [0, 1]
    """
    model.eval()
    # Collect a single batch for the test
    token_ids, _ = next(iter(loader))
    token_ids = token_ids.to(device)

    # For stability we perturb the embedding layer weights temporarily
    # but here we use the EpigeneticNetwork's encode step differently.
    # Simpler: add noise to the input token IDs by flipping random tokens.
    results = []
    with torch.no_grad():
        for _ in range(n_trials):
            # Small token perturbation: randomly replace ~2% of tokens with UNK
            noisy_ids = token_ids.clone()
            mask = torch.rand_like(token_ids.float()) < 0.02
            noisy_ids[mask] = 1   # UNK token

            is_epi = isinstance(model, EpigeneticNetwork)
            if is_epi:
                logits, _, _ = model(noisy_ids, update_memory=False)
            else:
                logits = model(noisy_ids)

            probs = torch.softmax(logits, dim=-1)
            results.append(probs.cpu())

    import torch as _torch
    stacked = _torch.stack(results, dim=0)      # (n_trials, B, C)
    std_val = stacked.std(dim=0).mean().item()  # mean std over trials
    return max(0.0, 1.0 - std_val * 10)         # normalise to [0, 1]


# ---------------------------------------------------------------------------
# Comparative summary
# ---------------------------------------------------------------------------

def comparative_summary(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> str:
    """
    Format a comparison table from experiment results.

    Parameters
    ----------
    results : {
        'EpiNet': {'task_a_acc': ..., 'task_b_acc': ..., 'forgetting': ..., ...},
        'Transformer': {...},
        ...
    }

    Returns
    -------
    Formatted string table.
    """
    header  = f"\n{'Model':<20} {'Task-A Acc':>12} {'Task-B Acc':>12} {'Forgetting':>12} {'Stability':>12} {'Params':>10}"
    sep     = "-" * len(header)
    lines   = [header, sep]

    for model_name, stats in results.items():
        line = (
            f"{model_name:<20} "
            f"{stats.get('task_a_acc', 0.0):>12.4f} "
            f"{stats.get('task_b_acc', 0.0):>12.4f} "
            f"{stats.get('forgetting', 0.0):>12.4f} "
            f"{stats.get('stability', 0.0):>12.4f} "
            f"{stats.get('params', 0):>10,}"
        )
        lines.append(line)

    table = "\n".join(lines) + "\n"

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(table)
            if save_path.endswith(".json"):
                pass  # handled below

        json_path = save_path.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

    return table
