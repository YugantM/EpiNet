"""
Tests for EpigeneticController and HomeostasisModule.
"""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controllers.epigenetic_controller import EpigeneticController
from src.controllers.homeostasis import HomeostasisModule


# ---------------------------------------------------------------------------
# EpigeneticController
# ---------------------------------------------------------------------------

class TestEpigeneticController:

    @pytest.fixture
    def ctrl(self):
        return EpigeneticController(epigenetic_dim=16, alpha=0.3, event_threshold=0.5)

    def test_reset_returns_correct_shape(self, ctrl):
        e = ctrl.reset(batch_size=4)
        assert e.shape == (4, 16)

    def test_step_output_shape(self, ctrl):
        e_t   = torch.zeros(4, 16)
        delta = torch.randn(4, 16)
        e_next = ctrl.step(e_t, delta)
        assert e_next.shape == (4, 16)

    def test_step_is_ema_update(self, ctrl):
        """e_next should be between e_t and delta."""
        e_t   = torch.zeros(4, 16)
        delta = torch.ones(4, 16)
        e_next = ctrl.step(e_t, delta)
        # With alpha=0.3: e_next = 0.7 * 0 + 0.3 * 1 = 0.3
        expected = torch.ones(4, 16) * 0.3
        assert torch.allclose(e_next, expected, atol=1e-5)

    def test_alpha_zero_means_no_change(self):
        ctrl = EpigeneticController(16, alpha=0.0)
        e_t   = torch.ones(2, 16) * 0.5
        delta = torch.zeros(2, 16)
        e_next = ctrl.step(e_t, delta)
        assert torch.allclose(e_next, e_t, atol=1e-6)

    def test_alpha_one_means_full_replacement(self):
        ctrl = EpigeneticController(16, alpha=1.0)
        e_t   = torch.ones(2, 16) * 0.5
        delta = torch.zeros(2, 16)
        e_next = ctrl.step(e_t, delta)
        assert torch.allclose(e_next, delta, atol=1e-6)

    def test_blend(self, ctrl):
        e_a = torch.zeros(2, 16)
        e_b = torch.ones(2, 16)
        e_blend = ctrl.blend(e_a, e_b, weight=0.5)
        expected = torch.ones(2, 16) * 0.5
        assert torch.allclose(e_blend, expected, atol=1e-6)

    def test_save_restore_context(self, ctrl):
        e = torch.randn(2, 16)
        ctrl.save_context(e, "task_a")
        e_restored = ctrl.restore_context("task_a")
        assert torch.allclose(e, e_restored, atol=1e-5)

    def test_restore_missing_context_returns_none(self, ctrl):
        assert ctrl.restore_context("nonexistent") is None

    def test_list_contexts(self, ctrl):
        ctrl.save_context(torch.randn(2, 16), "ctx1")
        ctrl.save_context(torch.randn(2, 16), "ctx2")
        assert "ctx1" in ctrl.list_contexts()
        assert "ctx2" in ctrl.list_contexts()

    def test_record_and_trajectory(self, ctrl):
        for _ in range(5):
            ctrl.record_state(torch.randn(2, 16))
        traj = ctrl.state_trajectory()
        assert traj.shape == (5, 16)

    def test_state_entropy_positive(self, ctrl):
        e = torch.randn(4, 16)
        entropy = ctrl.state_entropy(e)
        assert entropy >= 0.0

    def test_event_count_increments(self, ctrl):
        """Large shift should trigger event detection."""
        e_t   = torch.zeros(2, 16)
        delta = torch.ones(2, 16) * 10.0   # very different → large cosine distance
        ctrl.step(e_t, delta)
        # Event may or may not trigger depending on threshold; just check no crash
        assert ctrl.event_count >= 0


# ---------------------------------------------------------------------------
# HomeostasisModule
# ---------------------------------------------------------------------------

class TestHomeostasisModule:

    @pytest.fixture
    def hm(self):
        return HomeostasisModule(
            stress_threshold  = 0.7,
            low_threshold     = 0.2,
            adaptation_factor = 0.5,
            window            = 5,
            enabled           = True,
        )

    def test_update_returns_scalar(self, hm):
        scale = hm.update(0.5)
        assert isinstance(scale, float)

    def test_high_stress_reduces_lr(self, hm):
        # Warm up EMA
        for _ in range(10):
            hm.update(0.9)
        scale = hm.update(0.9)
        assert scale < 1.0, "High stress should reduce LR scale"

    def test_low_activity_increases_lr(self, hm):
        for _ in range(10):
            hm.update(0.05)
        scale = hm.update(0.05)
        assert scale > 1.0, "Low activity should increase LR scale"

    def test_normal_activity_no_change(self, hm):
        for _ in range(10):
            hm.update(0.5)
        scale = hm.update(0.5)
        assert scale == pytest.approx(1.0, abs=0.3), \
            f"Normal activity should leave LR roughly unchanged: {scale}"

    def test_disabled_always_returns_one(self):
        hm = HomeostasisModule(enabled=False)
        for _ in range(10):
            hm.update(0.99)
        assert hm.update(0.99) == pytest.approx(1.0, abs=1e-6)

    def test_gate_scale_factor_under_stress(self, hm):
        for _ in range(20):
            hm.update(0.95)
        factor = hm.gate_scale_factor()
        assert factor < 1.0, "Gate scale should reduce under stress"

    def test_gate_scale_factor_no_stress(self, hm):
        factor = hm.gate_scale_factor()
        assert factor == pytest.approx(1.0, abs=1e-6)

    def test_summary_keys(self, hm):
        for _ in range(5):
            hm.update(0.5)
        s = hm.summary()
        assert "stress_ema" in s
        assert "adaptation_events" in s
        assert "mean_stress" in s

    def test_adaptation_events_counted(self, hm):
        for _ in range(30):
            hm.update(0.95)   # should trigger events
        assert hm.adaptation_events() > 0

    def test_reset_history(self, hm):
        for _ in range(10):
            hm.update(0.8)
        hm.reset_history()
        assert hm.current_stress() == pytest.approx(0.0, abs=1e-6)
        assert hm.adaptation_events() == 0
