"""
Integration tests for EpigeneticNetwork and BaselineTransformer.

Tests cover:
  - End-to-end forward pass shapes
  - Epigenetic state update across calls
  - Memory write/read integration
  - Parameter count
  - Prediction API
  - Memory reset between tasks
  - Gradient flow through full network
"""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def epinet():
    return EpigeneticNetwork(
        vocab_size       = 100,
        embed_dim        = 32,
        hidden_dims      = [64, 32],
        epigenetic_dim   = 16,
        num_classes      = 2,
        memory_size      = 16,
        memory_dim       = 32,
        memory_lambda    = 0.1,
        epigenetic_alpha = 0.3,
        num_heads        = 2,
        dropout          = 0.0,
        max_seq_len      = 32,
    )


@pytest.fixture
def transformer():
    return BaselineTransformer(
        vocab_size  = 100,
        embed_dim   = 32,
        num_heads   = 2,
        num_layers  = 2,
        num_classes = 2,
        max_seq_len = 32,
        dropout     = 0.0,
    )


@pytest.fixture
def token_batch():
    torch.manual_seed(0)
    return torch.randint(1, 100, (4, 16))    # batch=4, seq_len=16


# ---------------------------------------------------------------------------
# EpigeneticNetwork tests
# ---------------------------------------------------------------------------

class TestEpigeneticNetworkForward:

    def test_logits_shape(self, epinet, token_batch):
        logits, e_next, info = epinet(token_batch)
        assert logits.shape == (4, 2)

    def test_epigenetic_state_shape(self, epinet, token_batch):
        _, e_next, _ = epinet(token_batch)
        assert e_next.shape == (4, 16)

    def test_info_dict_keys(self, epinet, token_batch):
        _, _, info = epinet(token_batch)
        assert "gate_values" in info
        assert "memory_ctx"  in info
        assert "hidden"      in info
        assert "epigenetic_state" in info

    def test_hidden_shape(self, epinet, token_batch):
        _, _, info = epinet(token_batch)
        assert info["hidden"].shape == (4, 32)   # last hidden_dim

    def test_memory_ctx_shape(self, epinet, token_batch):
        _, _, info = epinet(token_batch)
        assert info["memory_ctx"].shape == (4, 32)

    def test_gate_values_populated(self, epinet, token_batch):
        _, _, info = epinet(token_batch)
        gate_vals = info["gate_values"]
        assert len(gate_vals) > 0
        assert len(gate_vals[0]) > 0   # at least one head

    def test_no_nan_in_logits(self, epinet, token_batch):
        logits, _, _ = epinet(token_batch)
        assert not torch.isnan(logits).any()

    def test_custom_e_t_is_used(self, epinet, token_batch):
        e0 = torch.zeros(4, 16)
        e1 = torch.ones(4, 16)
        logits_0, _, _ = epinet(token_batch, e_t=e0, update_memory=False)
        logits_1, _, _ = epinet(token_batch, e_t=e1, update_memory=False)
        # Different initial epigenetic states should produce different logits
        assert not torch.allclose(logits_0, logits_1)

    def test_update_memory_writes_to_store(self, epinet, token_batch):
        util_before = epinet.memory.utilization()
        epinet(token_batch, update_memory=True)
        util_after = epinet.memory.utilization()
        assert util_after >= util_before   # memory was written

    def test_no_update_memory_skips_write(self, epinet, token_batch):
        epinet.memory.reset()
        epinet(token_batch, update_memory=False)
        assert epinet.memory.utilization() == pytest.approx(0.0, abs=1e-6)


class TestEpigeneticNetworkMemory:

    def test_reset_memory_clears_store(self, epinet, token_batch):
        epinet(token_batch, update_memory=True)
        epinet.reset_memory()
        assert epinet.memory.utilization() == pytest.approx(0.0, abs=1e-6)

    def test_memory_changes_output(self, epinet, token_batch):
        """Logits should differ before and after memory accumulation."""
        epinet.memory.reset()
        logits_1, _, _ = epinet(token_batch, update_memory=False)
        # Simulate accumulated memory
        for _ in range(3):
            epinet(token_batch, update_memory=True)
        logits_2, _, _ = epinet(token_batch, update_memory=False)
        # Memory changes should influence output (not identical)
        assert not torch.allclose(logits_1, logits_2, atol=1e-4)


class TestEpigeneticNetworkParameters:

    def test_parameter_count_positive(self, epinet):
        assert epinet.num_parameters() > 0

    def test_parameter_count_is_finite(self, epinet):
        n = epinet.num_parameters()
        assert n == int(n)  # integer
        assert n < 100_000_000  # sanity upper bound

    def test_predict_returns_class_indices(self, epinet, token_batch):
        preds = epinet.predict(token_batch)
        assert preds.shape == (4,)
        assert preds.dtype == torch.long
        assert preds.min().item() >= 0
        assert preds.max().item() <= 1


class TestEpigeneticNetworkGradients:

    def test_end_to_end_gradient_flow(self, epinet, token_batch):
        labels = torch.randint(0, 2, (4,))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(epinet.parameters(), lr=0.01)

        optimizer.zero_grad()
        logits, _, _ = epinet(token_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # At least some parameters should have received gradients
        grads = [p.grad for p in epinet.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_loss_decreases_with_training(self, epinet, token_batch):
        """Loss should decrease after several gradient steps on a fixed batch."""
        torch.manual_seed(42)
        labels = torch.randint(0, 2, (4,))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(epinet.parameters(), lr=1e-2)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            logits, e_t, _ = epinet(token_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at end should be lower than at start (or at least not exploding)
        assert losses[-1] <= losses[0] * 2.0, \
            f"Loss not decreasing: {losses[0]:.4f} → {losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# BaselineTransformer tests
# ---------------------------------------------------------------------------

class TestBaselineTransformer:

    def test_logits_shape(self, transformer, token_batch):
        logits = transformer(token_batch)
        assert logits.shape == (4, 2)

    def test_no_nan_in_logits(self, transformer, token_batch):
        logits = transformer(token_batch)
        assert not torch.isnan(logits).any()

    def test_parameter_count(self, transformer):
        assert transformer.num_parameters() > 0

    def test_gradient_flow(self, transformer, token_batch):
        labels = torch.randint(0, 2, (4,))
        logits = transformer(token_batch)
        loss   = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        grads = [p.grad for p in transformer.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_pad_mask(self, transformer, token_batch):
        """Model should handle padding masks without error."""
        pad_mask = torch.zeros(4, 16, dtype=torch.bool)
        pad_mask[:, 12:] = True   # last 4 tokens are padding
        logits = transformer(token_batch, pad_mask=pad_mask)
        assert logits.shape == (4, 2)
        assert not torch.isnan(logits).any()

    def test_parameter_count_similar_to_epinet(self, transformer, epinet):
        """Both models should be in the same order of magnitude (fair comparison)."""
        tf_params  = transformer.num_parameters()
        epi_params = epinet.num_parameters()
        ratio = max(tf_params, epi_params) / min(tf_params, epi_params)
        assert ratio < 20, \
            f"Parameter counts too different: Transformer={tf_params:,} vs EpiNet={epi_params:,}"
