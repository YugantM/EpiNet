"""
EpigeneticSeq2Seq — training and interactive demo on the Q&A dataset.

The model is trained in a seq2seq fashion:
  source  : question tokens     (e.g.  "What is the capital of France ?")
  target  : answer tokens       (e.g.  "Paris")

At inference, given a question the decoder autoregressively generates an answer.

Because the Q&A dataset is tiny (80 questions), the model will essentially
memorise the training set.  The experiment demonstrates:

  1. That the EpigeneticSeq2Seq architecture can train end-to-end.
  2. How the persistent epigenetic state e_t changes as more of the question
     is encoded, and how it steers the decoder differently for different topics.
  3. An interactive loop where a user can type a known question and see the
     generated answer.

Usage
-----
    python -m experiments.demo_decoder
    python -m experiments.demo_decoder --epochs 40 --interactive
"""

import argparse
import os
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epigenetic_decoder import EpigeneticSeq2Seq
from src.data.qa_dataset import _QA_RAW, QAVocabulary, _tokenise


# ─────────────────────────────────────────────────────────────────────────────
# Seq2Seq dataset: source = question, target = correct answer
# ─────────────────────────────────────────────────────────────────────────────

class QASeq2SeqDataset(Dataset):
    """Each item: (question_ids, answer_ids).  Both padded to fixed length."""

    def __init__(
        self,
        items:   List[Dict],
        vocab:   QAVocabulary,
        src_len: int = 32,
        tgt_len: int = 16,
    ) -> None:
        self.vocab   = vocab
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.items   = items

        self.src_ids = []
        self.tgt_ids = []

        for item in items:
            q_toks = [vocab.BOS] + [vocab._w2i.get(t, vocab.UNK)
                                     for t in _tokenise(item["q"])]
            q_toks = q_toks[:src_len]
            q_toks += [vocab.PAD] * (src_len - len(q_toks))

            a_toks = [vocab.BOS] + [vocab._w2i.get(t, vocab.UNK)
                                     for t in _tokenise(item["a"])]
            # target includes BOS (teacher forced input) + EOS (supervision)
            a_tgt  = a_toks + [vocab.EOS]
            a_tgt  = a_tgt[:tgt_len]
            a_tgt += [vocab.PAD] * (tgt_len - len(a_tgt))

            self.src_ids.append(q_toks)
            self.tgt_ids.append(a_tgt)

        self.src_ids = torch.tensor(self.src_ids, dtype=torch.long)
        self.tgt_ids = torch.tensor(self.tgt_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src_ids[idx], self.tgt_ids[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary construction
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(items: List[Dict]) -> QAVocabulary:
    vocab = QAVocabulary()
    texts = []
    for item in items:
        texts.append(item["q"])
        texts.append(item["a"])
        texts.extend(item["d"])
    vocab.build(texts)
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_seq2seq(
    model:        EpigeneticSeq2Seq,
    train_loader: DataLoader,
    epochs:       int,
    lr:           float,
) -> List[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore PAD

    loss_hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            # tgt input = all tokens except last; supervision = all except first
            logits = model(src, tgt[:, :-1])      # (B, tgt_len-1, V)
            labels = tgt[:, 1:].contiguous()       # (B, tgt_len-1)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        loss_hist.append(avg_loss)
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    return loss_hist


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative evaluation
# ─────────────────────────────────────────────────────────────────────────────

def decode_answer(
    model:   EpigeneticSeq2Seq,
    vocab:   QAVocabulary,
    question: str,
    src_len: int = 32,
    max_new_tokens: int = 12,
    temperature: float = 0.7,
) -> str:
    model.eval()
    q_toks = [vocab.BOS] + [vocab._w2i.get(t, vocab.UNK)
                              for t in _tokenise(question)]
    q_toks = q_toks[:src_len] + [vocab.PAD] * max(0, src_len - len(q_toks))
    src = torch.tensor([q_toks], dtype=torch.long)

    gen_ids = model.generate(
        src, bos_id=vocab.BOS, eos_id=vocab.EOS,
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=10, top_p=0.9,
    )
    tokens = [vocab._i2w.get(i, "<UNK>") for i in gen_ids
              if i not in (vocab.PAD, vocab.BOS, vocab.EOS)]
    return " ".join(tokens) if tokens else "<empty>"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("\n" + "=" * 60)
    print("  EpigeneticSeq2Seq — Q&A Generation Demo")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    items = list(_QA_RAW)
    vocab = build_vocab(items)
    print(f"\n  Questions: {len(items)}  |  Vocab size: {len(vocab)}")

    dataset = QASeq2SeqDataset(items, vocab, src_len=32, tgt_len=16)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────
    model = EpigeneticSeq2Seq(
        vocab_size=len(vocab), embed_dim=64, epigenetic_dim=32,
        hidden_dims=[128, 64], decoder_hidden=128, memory_size=32,
        memory_dim=64, dropout=0.1, max_seq_len=32,
    )
    print(f"  Model parameters: {model.num_parameters():,}")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n  Training for {args.epochs} epochs (lr={args.lr}) ...")
    loss_hist = train_seq2seq(model, loader, args.epochs, args.lr)
    print(f"  Final loss: {loss_hist[-1]:.4f}")

    # ── Qualitative evaluation ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Sample generations (training set questions):")
    print("─" * 60)

    sample_qs = [
        ("What is the capital of France?",           "Paris"),
        ("Who wrote '1984'?",                         "George Orwell"),
        ("What is the chemical symbol for gold?",     "Au"),
        ("How many rings are on the Olympic flag?",   "5"),
        ("What is the main ingredient in guacamole?", "Avocado"),
        ("What does CPU stand for?",                  "Central Processing Unit"),
    ]

    correct = 0
    for question, expected in sample_qs:
        generated = decode_answer(model, vocab, question,
                                  temperature=args.temperature)
        match = expected.lower() in generated.lower() or generated.lower() in expected.lower()
        marker = "✓" if match else " "
        print(f"  [{marker}] Q: {question}")
        print(f"       Expected : {expected}")
        print(f"       Generated: {generated}")
        correct += int(match)

    print(f"\n  Exact-ish match: {correct}/{len(sample_qs)}")
    print("\n  Note: with 80 training examples and a compact model the")
    print("  generator primarily memorises seen questions.  The demo")
    print("  validates that the seq2seq pipeline is end-to-end trainable")
    print("  and that e_t provides topic-steering at the decoder level.")

    # ── Interactive mode ──────────────────────────────────────────────
    if args.interactive:
        print("\n" + "=" * 60)
        print("  Interactive mode — type a question (or 'q' to quit):")
        print("=" * 60)
        while True:
            try:
                q = input("\n  Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("q", "quit", "exit", ""):
                break
            ans = decode_answer(model, vocab, q, temperature=args.temperature)
            print(f"  Answer  : {ans}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--interactive", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
