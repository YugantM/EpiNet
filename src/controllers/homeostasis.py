"""
Homeostasis Module — biological-inspired adaptive regulation.

In biology, homeostasis is the process by which a cell maintains a stable
internal environment despite external perturbations.  Neurons under chronic
high activation ("stress") downscale their synaptic weights; neurons that
are chronically under-active upscale (synaptic scaling / intrinsic plasticity).

This module implements a computational analogue:
  - Tracks a running "stress" signal (mean gate activation per layer).
  - When stress exceeds a threshold, reduces the effective learning rate
    and/or scales down gate activations to prevent runaway excitation.
  - When stress drops below a low threshold, nudges learning rate upward
    to promote plasticity in quiet phases.

The HomeostasisModule is decoupled from the network and acts as a
*training-time callback* as well as an *inference-time regulator*.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class HomeostasisModule(nn.Module):
    """
    Adaptive homeostatic regulator.

    Parameters
    ----------
    stress_threshold : float
        Gate-activation mean above which "stress" is declared.
    low_threshold : float
        Gate-activation mean below which "inactivity" is declared.
    adaptation_factor : float
        Multiplicative factor applied to LR or gate scales when stressed.
    window : int
        Size of the exponential moving average window for stress estimation.
    enabled : bool
        If False, the module acts as a no-op (useful for ablation studies).
    """

    def __init__(
        self,
        stress_threshold:  float = 0.7,
        low_threshold:     float = 0.2,
        adaptation_factor: float = 0.5,
        window:            int   = 10,
        enabled:           bool  = True,
    ) -> None:
        super().__init__()

        self.stress_threshold  = stress_threshold
        self.low_threshold     = low_threshold
        self.adaptation_factor = adaptation_factor
        self.window            = window
        self.enabled           = enabled

        # EMA stress estimate
        self.register_buffer("_stress_ema", torch.tensor(0.0))
        self._ema_alpha = 2.0 / (window + 1)    # standard EMA decay

        # History
        self._stress_history:   List[float] = []
        self._lr_scale_history: List[float] = []
        self._adaptation_events: int = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def update(self, gate_activation_mean: float) -> float:
        """
        Record one step's worth of gate activation and return the current
        recommended LR scale factor.

        Parameters
        ----------
        gate_activation_mean : float
            Mean sigmoid gate value across all heads/layers this step.

        Returns
        -------
        lr_scale : float  (1.0 = no change, <1.0 = reduce LR, >1.0 = increase)
        """
        if not self.enabled:
            return 1.0

        # Update EMA stress
        self._stress_ema = (
            self._ema_alpha * gate_activation_mean
            + (1 - self._ema_alpha) * self._stress_ema
        )
        stress = self._stress_ema.item()
        self._stress_history.append(stress)

        # Determine adaptation
        lr_scale = 1.0
        if stress > self.stress_threshold:
            lr_scale = self.adaptation_factor
            self._adaptation_events += 1
            logger.debug(
                f"Homeostasis: high stress ({stress:.3f}) → lr_scale={lr_scale}"
            )
        elif stress < self.low_threshold:
            # Gentle upward nudge to encourage plasticity
            lr_scale = 1.0 + (1.0 - self.adaptation_factor) * 0.5
            logger.debug(
                f"Homeostasis: low activity ({stress:.3f}) → lr_scale={lr_scale}"
            )

        self._lr_scale_history.append(lr_scale)
        return lr_scale

    def apply_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        lr_scale: float,
    ) -> None:
        """
        Directly adjust the learning rate of an optimizer.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        base_lr   : float   — the reference learning rate
        lr_scale  : float   — output of self.update(...)
        """
        new_lr = base_lr * lr_scale
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def gate_scale_factor(self) -> float:
        """
        Return a multiplicative scale to apply to gate activations when stressed.
        (Applied inside EpigeneticLayer during inference — soft inhibition.)
        """
        if not self.enabled:
            return 1.0
        stress = self._stress_ema.item()
        if stress > self.stress_threshold:
            # Linearly scale between 1.0 (at threshold) and adaptation_factor (at 1.0)
            excess = (stress - self.stress_threshold) / (1.0 - self.stress_threshold + 1e-9)
            return 1.0 - (1.0 - self.adaptation_factor) * min(excess, 1.0)
        return 1.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def current_stress(self) -> float:
        return self._stress_ema.item()

    def adaptation_events(self) -> int:
        return self._adaptation_events

    def summary(self) -> Dict[str, float]:
        return {
            "stress_ema":         self.current_stress(),
            "adaptation_events":  float(self._adaptation_events),
            "mean_stress":        float(
                sum(self._stress_history) / max(len(self._stress_history), 1)
            ),
        }

    def reset_history(self) -> None:
        self._stress_history.clear()
        self._lr_scale_history.clear()
        self._adaptation_events = 0
        self._stress_ema.zero_()
