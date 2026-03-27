"""
Tests for GeneralizedEpigeneticNeuron and EpiTransformer.
"""

import pytest
import torch
import torch.nn as nn
from src.models.generalized_epigenetic_neuron import GeneralizedEpigeneticNeuron
from src.models.epi_transformer import (
    GeneralizedEpigeneticFFN,
    EpiTransformerBlock,
    EpiTransformer,
)


# ─────────────────────────────────────────────────────────────────────────────
# GeneralizedEpigeneticNeuron
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneralizedEpigeneticNeuron:

    def _make(self, input_dim=16, output_dim=32, epi_dim=8):
        return GeneralizedEpigeneticNeuron(
            input_dim=input_dim, output_dim=output_dim,
            epigenetic_dim=epi_dim, memory_lambda=0.1,
            activation="gelu", skip_weight=0.1,
        )

    def test_output_shape(self):
        B, D_in, D_out, D_e = 4, 16, 32, 8
        m = self._make(D_in, D_out, D_e)
        x   = torch.randn(B, D_in)
        e_t = torch.randn(B, D_e)
        y, gate = m(x, e_t)
        assert y.shape   == (B, D_out)
        assert gate.shape == (B, D_out)

    def test_gate_in_unit_interval(self):
        m = self._make()
        x   = torch.randn(8, 16)
        e_t = torch.randn(8, 8)
        _, gate = m(x, e_t)
        assert gate.min() >= 0.0 - 1e-6
        assert gate.max() <= 1.0 + 1e-6

    def test_memory_injection_changes_output(self):
        m = self._make()
        x   = torch.randn(4, 16)
        e_t = torch.randn(4, 8)
        mem = torch.randn(4, 16)
        y_no_mem, _ = m(x, e_t, m=None)
        y_with_mem, _ = m(x, e_t, m=mem)
        assert not torch.allclose(y_no_mem, y_with_mem)

    def test_different_e_t_gives_different_output(self):
        """Gate is a function of e_t — different state → different output."""
        m = self._make()
        x    = torch.randn(4, 16)
        e_t1 = torch.ones(4, 8)
        e_t2 = torch.full((4, 8), -1.0)
        y1, _ = m(x, e_t1)
        y2, _ = m(x, e_t2)
        assert not torch.allclose(y1, y2)

    def test_same_input_different_e_t_gate_differs(self):
        m = self._make()
        x    = torch.randn(1, 16)
        e1   = torch.ones(1, 8)
        e2   = -torch.ones(1, 8)
        _, g1 = m(x, e1)
        _, g2 = m(x, e2)
        assert not torch.allclose(g1, g2)

    def test_gradient_flows(self):
        m = self._make()
        x   = torch.randn(4, 16, requires_grad=True)
        e_t = torch.randn(4, 8,  requires_grad=True)
        y, _ = m(x, e_t)
        y.sum().backward()
        assert x.grad   is not None
        assert e_t.grad is not None

    def test_no_memory_proj_when_lambda_zero(self):
        m = GeneralizedEpigeneticNeuron(
            input_dim=16, output_dim=32, epigenetic_dim=8,
            memory_lambda=0.0,
        )
        assert m.memory_proj is None

    def test_skip_connection_active(self):
        """With skip_weight > 0 and input != output dim, W_skip exists."""
        m = self._make(input_dim=16, output_dim=32)
        assert m.W_skip is not None

    def test_no_skip_when_weight_zero(self):
        m = GeneralizedEpigeneticNeuron(
            input_dim=16, output_dim=32, epigenetic_dim=8, skip_weight=0.0
        )
        assert m.W_skip is None

    def test_activations(self):
        for act in ("relu", "gelu", "tanh", "silu"):
            m = GeneralizedEpigeneticNeuron(
                input_dim=8, output_dim=16, epigenetic_dim=4, activation=act
            )
            x   = torch.randn(2, 8)
            e_t = torch.randn(2, 4)
            y, _ = m(x, e_t)
            assert y.shape == (2, 16), f"activation {act} failed"


# ─────────────────────────────────────────────────────────────────────────────
# GeneralizedEpigeneticFFN
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneralizedEpigeneticFFN:

    def _make(self, embed_dim=32, epi_dim=16):
        return GeneralizedEpigeneticFFN(
            embed_dim=embed_dim, epigenetic_dim=epi_dim,
            inner_dim=64, memory_lambda=0.1, dropout=0.0,
        )

    def test_output_shape(self):
        B, T, D, D_e = 2, 10, 32, 16
        ffn = self._make(D, D_e)
        x   = torch.randn(B, T, D)
        e_t = torch.randn(B, D_e)
        out, gates = ffn(x, e_t, m=None)
        assert out.shape   == (B, T, D)
        assert gates.shape == (B, T, 64)   # inner_dim

    def test_e_t_modulates_output(self):
        ffn = self._make()
        x    = torch.randn(2, 5, 32)
        e1   = torch.ones(2, 16)
        e2   = -torch.ones(2, 16)
        o1, _ = ffn(x, e1, m=None)
        o2, _ = ffn(x, e2, m=None)
        assert not torch.allclose(o1, o2)

    def test_gradient_flows_through_e_t(self):
        ffn = self._make()
        x   = torch.randn(2, 5, 32)
        e_t = torch.randn(2, 16, requires_grad=True)
        out, _ = ffn(x, e_t, m=None)
        out.sum().backward()
        assert e_t.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# EpiTransformerBlock
# ─────────────────────────────────────────────────────────────────────────────

class TestEpiTransformerBlock:

    def _make(self):
        return EpiTransformerBlock(
            embed_dim=32, num_heads=4, epigenetic_dim=16,
            inner_dim=64, dropout=0.0,
        )

    def test_output_shape(self):
        B, T, D = 2, 8, 32
        block = self._make()
        x   = torch.randn(B, T, D)
        e_t = torch.randn(B, 16)
        out, gates = block(x, e_t, memory_ctx=None)
        assert out.shape == (B, T, D)

    def test_padding_mask_accepted(self):
        B, T, D = 2, 8, 32
        block = self._make()
        x    = torch.randn(B, T, D)
        e_t  = torch.randn(B, 16)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -2:] = True   # last 2 tokens are padding
        out, _ = block(x, e_t, memory_ctx=None, key_padding_mask=mask)
        assert out.shape == (B, T, D)

    def test_gradient_flows(self):
        block = self._make()
        x   = torch.randn(2, 6, 32, requires_grad=True)
        e_t = torch.randn(2, 16, requires_grad=True)
        out, _ = block(x, e_t, memory_ctx=None)
        out.sum().backward()
        assert x.grad   is not None
        assert e_t.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# EpiTransformer (full model)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_model():
    return EpiTransformer(
        vocab_size=200, embed_dim=32, num_heads=4, num_layers=2,
        epigenetic_dim=16, num_classes=2, inner_dim=64,
        memory_size=16, memory_dim=32, dropout=0.0, max_seq_len=64,
    )


class TestEpiTransformer:

    def test_output_shapes(self, small_model):
        B, T = 4, 20
        token_ids = torch.randint(1, 200, (B, T))
        logits, e_next, info = small_model(token_ids)
        assert logits.shape == (B, 2)
        assert e_next.shape == (B, 16)

    def test_e_t_persists_and_updates(self, small_model):
        B, T = 3, 10
        ids = torch.randint(1, 200, (B, T))
        _, e1, _ = small_model(ids, e_t=None)
        _, e2, _ = small_model(ids, e_t=e1)
        assert not torch.allclose(e1, e2)

    def test_different_e_t_different_logits(self, small_model):
        B, T = 2, 10
        ids  = torch.randint(1, 200, (B, T))
        e_a  = torch.ones(B, 16)
        e_b  = -torch.ones(B, 16)
        l1, _, _ = small_model(ids, e_t=e_a, update_memory=False)
        l2, _, _ = small_model(ids, e_t=e_b, update_memory=False)
        assert not torch.allclose(l1, l2)

    def test_batch_size_guard(self, small_model):
        """Stale e_t from batch-32 should not crash on batch-7."""
        ids_big   = torch.randint(1, 200, (32, 15))
        _, e_big, _ = small_model(ids_big)
        ids_small = torch.randint(1, 200, (7, 15))
        # Should not raise even though e_big has size 32
        logits, _, _ = small_model(ids_small, e_t=e_big)
        assert logits.shape == (7, 2)

    def test_no_nan_in_output(self, small_model):
        ids = torch.randint(1, 200, (8, 30))
        logits, e_next, _ = small_model(ids)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(e_next).any()

    def test_gradient_flows_through_e_t(self, small_model):
        ids = torch.randint(1, 200, (4, 10))
        e_t = torch.randn(4, 16, requires_grad=True)
        logits, _, _ = small_model(ids, e_t=e_t, update_memory=False)
        logits.sum().backward()
        assert e_t.grad is not None

    def test_reset_memory(self, small_model):
        ids = torch.randint(1, 200, (4, 10))
        small_model(ids, update_memory=True)
        small_model.reset_memory()
        # After reset, memory store should be zeroed
        assert (small_model.memory.keys == 0).all()

    def test_num_parameters_positive(self, small_model):
        assert small_model.num_parameters() > 0

    def test_loss_decreases_with_training(self):
        """Sanity: model should be able to overfit a tiny batch."""
        model = EpiTransformer(
            vocab_size=50, embed_dim=32, num_heads=4, num_layers=2,
            epigenetic_dim=16, num_classes=2, inner_dim=64,
            memory_size=8, memory_dim=32, dropout=0.0, max_seq_len=16,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        ids    = torch.randint(1, 50, (8, 10))
        labels = torch.randint(0, 2, (8,))

        losses = []
        e_t = None
        for _ in range(20):
            optimizer.zero_grad()
            logits, e_t, _ = model(ids, e_t=e_t.detach() if e_t is not None else None)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease on tiny batch"

    def test_info_keys(self, small_model):
        ids = torch.randint(1, 200, (2, 8))
        _, _, info = small_model(ids)
        for key in ("gate_values", "memory_ctx", "hidden", "epigenetic_state"):
            assert key in info
