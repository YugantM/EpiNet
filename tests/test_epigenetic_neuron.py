"""
Unit tests for EpigeneticNeuron.

Tests cover:
  - Output shape correctness
  - Gate values are in (0, 1)
  - Memory injection changes output
  - Epigenetic state changes output (modulation works)
  - Gradient flow through the neuron
  - Activation function variants
"""

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_neuron import EpigeneticNeuron


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neuron():
    return EpigeneticNeuron(
        input_dim      = 32,
        output_dim     = 16,
        epigenetic_dim = 8,
        memory_lambda  = 0.1,
        activation     = "relu",
    )


@pytest.fixture
def sample_inputs():
    torch.manual_seed(0)
    B = 4
    x = torch.randn(B, 32)
    e = torch.randn(B, 8)
    m = torch.randn(B, 32)
    return x, e, m


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestEpigeneticNeuronShapes:

    def test_output_shape_no_memory(self, neuron, sample_inputs):
        x, e, _ = sample_inputs
        y = neuron(x, e, memory=None)
        assert y.shape == (4, 16), f"Expected (4,16), got {y.shape}"

    def test_output_shape_with_memory(self, neuron, sample_inputs):
        x, e, m = sample_inputs
        y = neuron(x, e, m)
        assert y.shape == (4, 16)

    def test_single_sample(self, neuron):
        x = torch.randn(1, 32)
        e = torch.randn(1, 8)
        y = neuron(x, e)
        assert y.shape == (1, 16)

    def test_gate_values_shape(self, neuron, sample_inputs):
        _, e, _ = sample_inputs
        g = neuron.gate_values(e)
        assert g.shape == (4, 16)


# ---------------------------------------------------------------------------
# Value tests
# ---------------------------------------------------------------------------

class TestEpigeneticNeuronValues:

    def test_gate_values_in_unit_interval(self, neuron, sample_inputs):
        _, e, _ = sample_inputs
        g = neuron.gate_values(e)
        assert g.min().item() >= 0.0 - 1e-6
        assert g.max().item() <= 1.0 + 1e-6

    def test_memory_changes_output(self, neuron, sample_inputs):
        x, e, m = sample_inputs
        y_no_mem = neuron(x, e, memory=None)
        y_mem    = neuron(x, e, memory=m)
        assert not torch.allclose(y_no_mem, y_mem), \
            "Memory should change the output"

    def test_epigenetic_state_modulates_output(self, neuron, sample_inputs):
        x, e, _ = sample_inputs
        e_zeros = torch.zeros_like(e)
        y1 = neuron(x, e)
        y2 = neuron(x, e_zeros)
        assert not torch.allclose(y1, y2), \
            "Different epigenetic states should produce different outputs"

    def test_relu_non_negative(self, sample_inputs):
        neuron_relu = EpigeneticNeuron(32, 16, 8, activation="relu")
        x, e, _ = sample_inputs
        y = neuron_relu(x, e)
        assert y.min().item() >= 0.0 - 1e-6, "ReLU output must be non-negative"

    def test_tanh_bounded(self, sample_inputs):
        neuron_tanh = EpigeneticNeuron(32, 16, 8, activation="tanh")
        x, e, _ = sample_inputs
        y = neuron_tanh(x, e)
        assert y.min().item() >= -1.0 - 1e-6
        assert y.max().item() <=  1.0 + 1e-6

    def test_sigmoid_bounded(self, sample_inputs):
        neuron_sig = EpigeneticNeuron(32, 16, 8, activation="sigmoid")
        x, e, _ = sample_inputs
        y = neuron_sig(x, e)
        assert y.min().item() >= 0.0 - 1e-6
        assert y.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestEpigeneticNeuronGradients:

    def test_gradients_flow_through_x(self, neuron, sample_inputs):
        x, e, _ = sample_inputs
        x.requires_grad_(True)
        y = neuron(x, e)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradients_flow_through_e(self, neuron, sample_inputs):
        x, e, _ = sample_inputs
        e.requires_grad_(True)
        y = neuron(x, e)
        loss = y.sum()
        loss.backward()
        assert e.grad is not None
        assert e.grad.shape == e.shape

    def test_weight_gradients_exist(self, neuron, sample_inputs):
        x, e, m = sample_inputs
        y = neuron(x, e, m)
        loss = y.mean()
        loss.backward()
        assert neuron.W.weight.grad is not None
        assert neuron.W_e.weight.grad is not None
        assert neuron.memory_proj.weight.grad is not None

    def test_no_nan_in_output(self, neuron, sample_inputs):
        x, e, m = sample_inputs
        y = neuron(x, e, m)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_no_inf_in_output(self, neuron, sample_inputs):
        x, e, m = sample_inputs
        y = neuron(x, e, m)
        assert not torch.isinf(y).any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestEpigeneticNeuronErrors:

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            EpigeneticNeuron(32, 16, 8, activation="swiglu")

    def test_mismatched_dims_raise(self, neuron):
        x = torch.randn(4, 16)   # wrong input_dim
        e = torch.randn(4, 8)
        with pytest.raises(RuntimeError):
            neuron(x, e)


# ---------------------------------------------------------------------------
# Memory lambda scaling test
# ---------------------------------------------------------------------------

def test_memory_lambda_scaling():
    """Higher lambda should increase the impact of memory."""
    torch.manual_seed(1)
    x = torch.randn(4, 32)
    e = torch.randn(4, 8)
    m = torch.randn(4, 32)

    neuron_lo = EpigeneticNeuron(32, 16, 8, memory_lambda=0.01)
    neuron_hi = EpigeneticNeuron(32, 16, 8, memory_lambda=1.0)

    # Copy weights so only lambda differs
    with torch.no_grad():
        neuron_hi.W.weight.copy_(neuron_lo.W.weight)
        neuron_hi.W.bias.copy_(neuron_lo.W.bias)
        neuron_hi.W_e.weight.copy_(neuron_lo.W_e.weight)
        neuron_hi.memory_proj.weight.copy_(neuron_lo.memory_proj.weight)

    y_no_mem = neuron_lo(x, e, memory=None)
    y_lo     = neuron_lo(x, e, memory=m)
    y_hi     = neuron_hi(x, e, memory=m)

    diff_lo = (y_lo - y_no_mem).abs().mean().item()
    diff_hi = (y_hi - y_no_mem).abs().mean().item()

    # Higher lambda → larger memory contribution
    assert diff_hi > diff_lo, \
        f"Higher lambda should increase memory impact: {diff_hi:.4f} vs {diff_lo:.4f}"
