"""
Metrics for evaluating Epigenetic Neural Networks.

Beyond standard accuracy, we track:
  - Adaptation speed  : how quickly performance recovers on a new task
  - Forgetting score  : performance drop on old task after learning new task
  - Stability index   : variance of outputs under small input perturbations
"""

from typing import Dict, List, Optional
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Basic classification metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy from raw logits."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary classification accuracy (logits → sigmoid threshold at 0.5)."""
    if logits.dim() == 2 and logits.size(1) == 2:
        return accuracy(logits, labels)
    preds = (torch.sigmoid(logits.squeeze()) >= 0.5).long()
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Continual learning metrics
# ---------------------------------------------------------------------------

def forgetting_score(
    acc_before: float,
    acc_after: float,
) -> float:
    """
    Backward-transfer / forgetting measure.

    Positive value → the model forgot (performance dropped).
    Negative value → the model improved (rare but possible via transfer).

    F = acc_before - acc_after
    """
    return acc_before - acc_after


def adaptation_speed(
    acc_per_epoch: List[float],
    target_acc: float = 0.70,
) -> int:
    """
    Number of epochs until accuracy first surpasses *target_acc*.

    Returns len(acc_per_epoch) if the target is never reached
    (indicating slow / failed adaptation).
    """
    for epoch, acc in enumerate(acc_per_epoch):
        if acc >= target_acc:
            return epoch + 1          # 1-indexed
    return len(acc_per_epoch)


def backward_transfer(
    task_accs: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Compute backward transfer for each task.

    task_accs: {'taskA': [acc_epoch0, acc_epoch1, ...], ...}
    Returns: {'taskA': BWT_score, ...}
    """
    results = {}
    for task, accs in task_accs.items():
        if len(accs) >= 2:
            results[task] = accs[-1] - accs[0]   # final - initial
        else:
            results[task] = 0.0
    return results


# ---------------------------------------------------------------------------
# Stability metric
# ---------------------------------------------------------------------------

def stability_index(
    model,
    inputs: torch.Tensor,
    noise_std: float = 0.01,
    n_trials: int = 20,
) -> float:
    """
    Measure output stability under small Gaussian input perturbations.

    stability = 1 - mean(std of softmax distributions across trials)

    A perfectly stable model returns 1.0; a highly unstable model returns ~0.
    """
    model.eval()
    all_probs = []
    with torch.no_grad():
        for _ in range(n_trials):
            noisy = inputs + torch.randn_like(inputs) * noise_std
            out = model(noisy)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=-1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.stack(all_probs, axis=0)   # (n_trials, batch, classes)
    std_per_class = all_probs.std(axis=0)     # (batch, classes)
    mean_std = std_per_class.mean().item()
    return max(0.0, 1.0 - mean_std * 100)     # scale to [0, 1]


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Lightweight tracker that accumulates per-batch metrics
    and reports epoch-level summaries.
    """

    def __init__(self):
        self._history: Dict[str, List[float]] = {}
        self._batch_buffer: Dict[str, List[float]] = {}

    def update(self, **kwargs: float) -> None:
        """Record one batch worth of scalar metrics."""
        for k, v in kwargs.items():
            self._batch_buffer.setdefault(k, []).append(v)

    def epoch_end(self) -> Dict[str, float]:
        """
        Compute epoch averages, store in history, clear buffer.

        Returns the epoch summary dict.
        """
        summary = {}
        for k, vals in self._batch_buffer.items():
            avg = float(np.mean(vals))
            self._history.setdefault(k, []).append(avg)
            summary[k] = avg
        self._batch_buffer.clear()
        return summary

    def get_history(self, key: str) -> List[float]:
        """Return full history for a metric key."""
        return self._history.get(key, [])

    def last(self, key: str, default: float = 0.0) -> float:
        """Return most recent epoch value for a metric."""
        hist = self._history.get(key, [])
        return hist[-1] if hist else default
