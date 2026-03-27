"""
Tests for the data loading utilities.
"""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import (
    Vocabulary,
    TextDataset,
    build_dataloaders,
    make_synthetic_sentiment,
    make_continual_tasks,
    make_context_adaptation_data,
)


class TestVocabulary:

    def test_special_tokens(self):
        v = Vocabulary()
        assert v.PAD == 0
        assert v.UNK == 1
        assert v.BOS == 2
        assert v.EOS == 3

    def test_build_increases_size(self):
        v = Vocabulary(min_freq=1)
        v.build(["hello world", "hello again"])
        assert len(v) > 4   # 4 special + some words

    def test_encode_produces_list(self):
        v = Vocabulary(min_freq=1)
        v.build(["hello world"])
        ids = v.encode("hello world")
        assert isinstance(ids, list)
        assert v.BOS in ids
        assert v.EOS in ids

    def test_encode_unknown_word_uses_unk(self):
        v = Vocabulary(min_freq=1)
        v.build(["hello world"])
        ids = v.encode("xyz123_unknown")
        assert v.UNK in ids

    def test_pad_pads_to_length(self):
        v = Vocabulary()
        ids = [1, 2, 3]
        padded = v.pad(ids, max_len=10)
        assert len(padded) == 10
        assert padded[3:] == [0] * 7

    def test_pad_truncates(self):
        v = Vocabulary()
        ids = list(range(20))
        truncated = v.pad(ids, max_len=5)
        assert len(truncated) == 5


class TestSyntheticData:

    def test_make_synthetic_sentiment_shape(self):
        texts, labels = make_synthetic_sentiment(100, seed=0)
        assert len(texts) == 100
        assert len(labels) == 100

    def test_make_synthetic_sentiment_balanced(self):
        texts, labels = make_synthetic_sentiment(200, seed=0)
        pos = sum(labels)
        neg = len(labels) - pos
        assert abs(pos - neg) <= 2   # perfectly balanced

    def test_labels_binary(self):
        _, labels = make_synthetic_sentiment(100, seed=0)
        assert all(l in (0, 1) for l in labels)

    def test_make_continual_tasks_returns_two_tasks(self):
        (texts_a, labels_a), (texts_b, labels_b) = make_continual_tasks(100)
        assert len(texts_a) == 100
        assert len(texts_b) == 100

    def test_make_context_adaptation_shape(self):
        texts, labels, contexts = make_context_adaptation_data(100, seed=0)
        assert len(texts) == 100
        assert len(labels) == 100
        assert len(contexts) == 100

    def test_context_adaptation_has_both_contexts(self):
        _, _, contexts = make_context_adaptation_data(100, seed=0)
        assert "formal" in contexts
        assert "casual" in contexts


class TestTextDataset:

    @pytest.fixture
    def vocab_and_dataset(self):
        texts  = ["good film", "bad movie", "great story", "terrible plot"]
        labels = [1, 0, 1, 0]
        vocab  = Vocabulary(min_freq=1)
        vocab.build(texts)
        ds = TextDataset(texts, labels, vocab, max_len=10)
        return vocab, ds

    def test_dataset_length(self, vocab_and_dataset):
        _, ds = vocab_and_dataset
        assert len(ds) == 4

    def test_getitem_returns_tensors(self, vocab_and_dataset):
        _, ds = vocab_and_dataset
        ids, label = ds[0]
        assert isinstance(ids,   torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_token_ids_correct_length(self, vocab_and_dataset):
        _, ds = vocab_and_dataset
        ids, _ = ds[0]
        assert ids.shape == (10,)

    def test_labels_correct(self, vocab_and_dataset):
        _, ds = vocab_and_dataset
        _, label = ds[1]
        assert label.item() == 0


class TestBuildDataloaders:

    def test_returns_three_items(self):
        train_l, val_l, vocab = build_dataloaders(
            task="sentiment", n_samples=100, max_len=16,
            batch_size=8, seed=0,
        )
        assert train_l is not None
        assert val_l   is not None
        assert vocab   is not None

    def test_train_loader_batch_shape(self):
        train_l, _, _ = build_dataloaders(
            task="sentiment", n_samples=100, max_len=16, batch_size=8, seed=0,
        )
        ids, labels = next(iter(train_l))
        assert ids.shape[1] == 16
        assert labels.shape[0] <= 8

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError):
            build_dataloaders(task="nonexistent", n_samples=100)
