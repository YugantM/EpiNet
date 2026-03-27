"""
EpigeneticController — manages the evolution of epigenetic state across
inference steps / time steps.

The controller is responsible for:
  1. Tracking the current epigenetic state e_t per sample (or per task).
  2. Applying the EMA update rule:
         e_{t+1} = (1 - α) · e_t  +  α · f(input, hidden, memory)
  3. Detecting *epigenetic events* (large shifts in state) and logging them.
  4. Supporting *context switching* — resetting or blending states when
     the task context changes (analogous to epigenetic reprogramming).

Design note
-----------
The controller intentionally does NOT own the network parameters; it only
manages state vectors.  This separation allows the same controller to work
with different network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class EpigeneticController(nn.Module):
    """
    Manages the epigenetic state e_t and provides update / reset utilities.

    Parameters
    ----------
    epigenetic_dim : int
        Dimension of the state vector.
    alpha : float
        EMA coefficient — how quickly state responds to new inputs.
        alpha=0 → state never changes; alpha=1 → state forgets past entirely.
    event_threshold : float
        Cosine-distance threshold above which a "state shift event" is logged.
    """

    def __init__(
        self,
        epigenetic_dim: int = 32,
        alpha: float = 0.3,
        event_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.epigenetic_dim  = epigenetic_dim
        self.alpha           = alpha
        self.event_threshold = event_threshold

        # Learnable baseline epigenetic state (analogous to the "ground state"
        # of a cell type before any environmental input)
        self.baseline_state = nn.Parameter(torch.zeros(1, epigenetic_dim))

        # History for analysis
        self._state_history: List[torch.Tensor] = []
        self._event_count: int = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def step(
        self,
        e_t: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply one EMA update step.

            e_{t+1} = (1 - α) · e_t  +  α · delta

        Parameters
        ----------
        e_t   : (batch, epigenetic_dim)
        delta : (batch, epigenetic_dim)   — target state from f(·)

        Returns
        -------
        e_next : (batch, epigenetic_dim)
        """
        e_next = (1 - self.alpha) * e_t + self.alpha * delta

        # Detect large state shifts (epigenetic events)
        dist = self._cosine_distance(e_t, e_next)
        if dist.mean().item() > self.event_threshold:
            self._event_count += 1
            logger.debug(
                f"Epigenetic event detected (shift={dist.mean().item():.3f}, "
                f"total_events={self._event_count})"
            )

        return e_next

    def reset(self, batch_size: int = 1) -> torch.Tensor:
        """
        Return the baseline (ground) epigenetic state.

        Call this when starting a completely new sequence or task.
        """
        return self.baseline_state.expand(batch_size, -1).clone()

    def blend(
        self,
        e_a: torch.Tensor,
        e_b: torch.Tensor,
        weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Blend two epigenetic states (partial context switch).

            e_blend = (1 - w) · e_a  +  w · e_b

        Useful when a new task partially overlaps with the previous one.
        """
        return (1 - weight) * e_a + weight * e_b

    # ------------------------------------------------------------------
    # Contextual adaptation
    # ------------------------------------------------------------------

    def save_context(self, e_t: torch.Tensor, label: str) -> None:
        """Save the current state under a named label (for context replay)."""
        if not hasattr(self, "_saved_contexts"):
            self._saved_contexts: Dict[str, torch.Tensor] = {}
        self._saved_contexts[label] = e_t.detach().clone()

    def restore_context(self, label: str) -> Optional[torch.Tensor]:
        """Restore a previously saved context state."""
        if not hasattr(self, "_saved_contexts"):
            return None
        return self._saved_contexts.get(label, None)

    def list_contexts(self) -> List[str]:
        if not hasattr(self, "_saved_contexts"):
            return []
        return list(self._saved_contexts.keys())

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def record_state(self, e_t: torch.Tensor) -> None:
        """Append current state to history (call once per time step)."""
        self._state_history.append(e_t.detach().mean(dim=0).cpu())

    def state_trajectory(self) -> torch.Tensor:
        """
        Return the recorded trajectory of mean epigenetic states.

        Returns
        -------
        trajectory : (T, epigenetic_dim)  or empty tensor if no history.
        """
        if not self._state_history:
            return torch.empty(0, self.epigenetic_dim)
        return torch.stack(self._state_history, dim=0)

    def state_entropy(self, e_t: torch.Tensor) -> float:
        """
        Proxy for the "diversity" of the epigenetic state:
        entropy of the softmax-normalised absolute values.

        Higher entropy ≈ more dimensions are active simultaneously.
        """
        with torch.no_grad():
            probs = F.softmax(e_t.abs().mean(dim=0), dim=0)
            entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
        return entropy

    def clear_history(self) -> None:
        self._state_history.clear()
        self._event_count = 0

    @property
    def event_count(self) -> int:
        return self._event_count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise cosine distance between two batches of vectors."""
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        cosine_sim = (a_norm * b_norm).sum(dim=-1)   # (B,)
        return 1.0 - cosine_sim

    def extra_repr(self) -> str:
        return (
            f"epigenetic_dim={self.epigenetic_dim}, alpha={self.alpha}, "
            f"event_threshold={self.event_threshold}"
        )
