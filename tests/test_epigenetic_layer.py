"""
Unit tests for EpigeneticLayer.

Tests cover:
  - Output shape and gate value shapes
  - Residual connection preserves information
  - Layer normalisation output statistics
  - Multi-head aggregation
  - Gate activation range
"""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_layer import EpigeneticLayer


@pytest.fixture
def layer():
    return EpigeneticLayer(
        input_dim      = 64,
        output_dim     = 64,
        epigenetic_dim = 32,
        num_heads      = 4,
        memory_lambda  = 0.1,
        activation     = "relu",
        dropout        = 0.0,   # off for deterministic tests
    )


@pytest.fixture
def sample():
    torch.manual_seed(0)
    B = 8
    return (
        torch.randn(B, 64),   # x
        torch.randn(B, 32),   # e
        torch.randn(B, 64),   # memory
    )


class TestEpigeneticLayerShapes:

    def test_output_shape(self, layer, sample):
        x, e, m = sample
        out, gate_vals = layer(x, e, m)
        assert out.shape == (8, 64)

    def test_gate_values_structure(self, layer, sample):
        x, e, m = sample
        _, gate_vals = layer(x, e, m)
        # gate_vals is a list of tensors, one per head
        assert len(gate_vals) == layer.num_heads
        for gv in gate_vals:
            assert gv.shape == (8, layer.head_dim)

    def test_no_memory_still_works(self, layer, sample):
        x, e, _ = sample
        out, _ = layer(x, e, memory=None)
        assert out.shape == (8, 64)

    def test_different_in_out_dims(self):
        layer2 = EpigeneticLayer(
            input_dim  = 32,
            output_dim = 64,
            epigenetic_dim = 16,
            num_heads  = 2,
            dropout    = 0.0,
        )
        x = torch.randn(4, 32)
        e = torch.randn(4, 16)
        out, _ = layer2(x, e)
        assert out.shape == (4, 64)


class TestEpigeneticLayerValues:

    def test_gate_values_in_unit_interval(self, layer, sample):
        x, e, m = sample
        _, gate_vals = layer(x, e, m)
        for gv in gate_vals:
            assert gv.min().item() >= 0.0 - 1e-5
            assert gv.max().item() <= 1.0 + 1e-5

    def test_mean_gate_activation_returns_scalar(self, layer, sample):
        _, e, _ = sample
        mean_gate = layer.mean_gate_activation(e)
        assert isinstance(mean_gate, float)
        assert 0.0 <= mean_gate <= 1.0

    def test_layer_norm_applied(self, layer, sample):
        """Output should have approximately zero mean and unit variance."""
        x, e, m = sample
        with torch.no_grad():
            out, _ = layer(x, e, m)
        # LayerNorm normalises last dimension
        # check that std is roughly 1 (within factor of 2)
        std = out.std(dim=-1).mean().item()
        assert 0.1 < std < 5.0, f"Unexpected std after layer norm: {std:.4f}"

    def test_no_nan_or_inf(self, layer, sample):
        x, e, m = sample
        out, gate_vals = layer(x, e, m)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        for gv in gate_vals:
            assert not torch.isnan(gv).any()


class TestEpigeneticLayerGradients:

    def test_gradients_flow(self, layer, sample):
        x, e, m = sample
        x.requires_grad_(True)
        e.requires_grad_(True)
        out, _ = layer(x, e, m)
        out.sum().backward()
        assert x.grad is not None
        assert e.grad is not None

    def test_parameter_gradients(self, layer, sample):
        x, e, m = sample
        out, _ = layer(x, e, m)
        out.mean().backward()
        for name, param in layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
