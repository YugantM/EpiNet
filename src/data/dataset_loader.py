"""
Dataset loading utilities for EpiNet experiments.

Supported datasets
------------------
1. Synthetic sentiment   — fast, no download; always available
2. IMDB (small subset)   — downloaded via torchtext if available; falls back
                           to synthetic on import failure
3. Synthetic continual   — two-task sequence for continual learning experiments
4. Context adaptation    — same inputs, different context labels

All datasets return (token_ids, label) pairs with a shared vocabulary.
"""

import random
import string
from typing import List, Tuple, Dict, Optional, Iterator
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Simple vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Minimal word-level vocabulary with special tokens.

    Special tokens:
      <PAD> = 0
      <UNK> = 1
      <BOS> = 2
      <EOS> = 3
    """

    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, max_size: int = 5000, min_freq: int = 2) -> None:
        self.max_size = max_size
        self.min_freq = min_freq
        self._word2idx: Dict[str, int] = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3
        }
        self._idx2word: Dict[int, str] = {v: k for k, v in self._word2idx.items()}

    def build(self, texts: List[str]) -> None:
        """Build vocabulary from a list of raw text strings."""
        counts: Counter = Counter()
        for text in texts:
            for word in text.lower().split():
                counts[word] += 1

        vocab_words = [
            word for word, cnt in counts.most_common(self.max_size - 4)
            if cnt >= self.min_freq
        ]
        for word in vocab_words:
            idx = len(self._word2idx)
            self._word2idx[word] = idx
            self._idx2word[idx]  = word

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        tokens = [self.BOS]
        for word in text.lower().split()[:max_len - 2]:
            tokens.append(self._word2idx.get(word, self.UNK))
        tokens.append(self.EOS)
        return tokens

    def pad(self, ids: List[int], max_len: int) -> List[int]:
        if len(ids) < max_len:
            ids = ids + [self.PAD] * (max_len - len(ids))
        return ids[:max_len]

    def __len__(self) -> int:
        return len(self._word2idx)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

# Word banks for synthetic sentiment generation
_POS_WORDS = [
    "great", "excellent", "amazing", "wonderful", "fantastic", "good",
    "love", "enjoy", "brilliant", "superb", "perfect", "best", "happy",
    "beautiful", "positive", "pleasure", "outstanding", "impressive",
]
_NEG_WORDS = [
    "terrible", "awful", "horrible", "bad", "worst", "hate", "boring",
    "disappointing", "poor", "dreadful", "mediocre", "waste", "slow",
    "stupid", "negative", "ugly", "useless", "broken", "failure",
]
_NEUTRAL_WORDS = [
    "the", "a", "an", "is", "was", "this", "film", "movie", "book",
    "story", "plot", "character", "scene", "time", "day", "think",
    "very", "quite", "really", "just", "also", "but", "and", "or",
]


def _make_sentence(label: int, length: int = 15) -> str:
    """Generate a synthetic sentence with the given sentiment label."""
    words = random.choices(_NEUTRAL_WORDS, k=length)
    sentiment_words = _POS_WORDS if label == 1 else _NEG_WORDS
    # Replace a few neutral words with sentiment words
    n_sentiment = max(2, length // 4)
    indices = random.sample(range(length), n_sentiment)
    for i in indices:
        words[i] = random.choice(sentiment_words)
    return " ".join(words)


def make_synthetic_sentiment(
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Generate a balanced synthetic binary sentiment dataset.

    Returns
    -------
    texts  : list of strings
    labels : list of 0/1 integers
    """
    random.seed(seed)
    texts, labels = [], []
    for _ in range(n_samples // 2):
        texts.append(_make_sentence(1, length=random.randint(10, 20)))
        labels.append(1)
        texts.append(_make_sentence(0, length=random.randint(10, 20)))
        labels.append(0)
    return texts, labels


# ---------------------------------------------------------------------------
# Task-pair for continual learning
# ---------------------------------------------------------------------------

def make_continual_tasks(
    n_per_task: int = 800,
    seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """
    Create two distinct classification tasks for continual learning.

    Task A: sentiment classification (positive vs negative)
    Task B: topic classification (tech vs sports) — synthetic

    Both tasks use the same input format (token sequences) but test
    different classification boundaries.
    """
    random.seed(seed)

    # Task A — sentiment (same as make_synthetic_sentiment)
    texts_a, labels_a = make_synthetic_sentiment(n_per_task, seed=seed)

    # Task B — topic: tech (1) vs sports (0)
    tech_words   = ["computer", "software", "algorithm", "data", "network",
                    "neural", "model", "train", "predict", "code", "system",
                    "ai", "machine", "learning", "python", "pytorch"]
    sports_words = ["game", "team", "score", "win", "lose", "play", "match",
                    "run", "ball", "player", "coach", "goal", "field", "race",
                    "sport", "championship"]

    texts_b, labels_b = [], []
    for _ in range(n_per_task // 2):
        # Tech sample
        words = random.choices(_NEUTRAL_WORDS, k=12)
        tech_count = random.randint(3, 6)
        for i in random.sample(range(len(words)), tech_count):
            words[i] = random.choice(tech_words)
        texts_b.append(" ".join(words))
        labels_b.append(1)
        # Sports sample
        words = random.choices(_NEUTRAL_WORDS, k=12)
        sports_count = random.randint(3, 6)
        for i in random.sample(range(len(words)), sports_count):
            words[i] = random.choice(sports_words)
        texts_b.append(" ".join(words))
        labels_b.append(0)

    return (texts_a, labels_a), (texts_b, labels_b)


# ---------------------------------------------------------------------------
# Context adaptation dataset
# ---------------------------------------------------------------------------

def make_context_adaptation_data(
    n_samples: int = 400,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Create a dataset where the *same* input maps to different outputs
    depending on a discrete context string.

    Returns
    -------
    texts    : list of text strings  (same text repeated across contexts)
    labels   : list of int labels    (vary by context)
    contexts : list of context tags  ('formal' | 'casual')
    """
    random.seed(seed)
    base_texts = [_make_sentence(random.randint(0, 1), 15) for _ in range(n_samples // 2)]

    texts, labels, contexts = [], [], []
    for text in base_texts:
        # Formal context: predicts positive sentiment
        texts.append(text)
        labels.append(1)
        contexts.append("formal")
        # Casual context: predicts negative sentiment (inverted label)
        texts.append(text)
        labels.append(0)
        contexts.append("casual")

    return texts, labels, contexts


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Wraps (texts, labels) into a PyTorch Dataset of padded token tensors.

    Parameters
    ----------
    texts   : list of raw strings
    labels  : list of int labels
    vocab   : Vocabulary (must already be built)
    max_len : int — pad/truncate to this length
    """

    def __init__(
        self,
        texts:   List[str],
        labels:  List[int],
        vocab:   Vocabulary,
        max_len: int = 128,
    ) -> None:
        self.max_len = max_len
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.token_ids = torch.tensor(
            [vocab.pad(vocab.encode(t, max_len), max_len) for t in texts],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_ids[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# High-level factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    task: str = "sentiment",
    n_samples: int = 1000,
    max_len: int = 128,
    batch_size: int = 32,
    val_split: float = 0.2,
    vocab_size: int = 5000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Vocabulary]:
    """
    Build train/val DataLoaders for the requested task.

    Parameters
    ----------
    task      : 'sentiment' | 'continual_a' | 'continual_b'
    n_samples : total number of samples
    ...

    Returns
    -------
    train_loader, val_loader, vocab
    """
    if task in ("sentiment", "continual_a"):
        texts, labels = make_synthetic_sentiment(n_samples, seed)
    elif task == "continual_b":
        _, (texts, labels) = make_continual_tasks(n_samples, seed)
    else:
        raise ValueError(f"Unknown task '{task}'")

    # Build vocab on all texts
    vocab = Vocabulary(max_size=vocab_size)
    vocab.build(texts)

    # Train/val split
    n_val   = int(len(texts) * val_split)
    n_train = len(texts) - n_val

    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_texts  = [texts[i]  for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts    = [texts[i]  for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    train_ds = TextDataset(train_texts, train_labels, vocab, max_len)
    val_ds   = TextDataset(val_texts,   val_labels,   vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, vocab
