"""
Multi-Model Interactive Scorer
================================

Trains all 5 models on the Q&A dataset, then probes them at 4 levels:

  Level 1 — MEMORISATION   : Exact questions from training data
  Level 2 — PARAPHRASE     : Same question, words reordered / synonym phrasing
  Level 3 — TOPIC TRANSFER : New questions in the same topic cluster,
                             using vocabulary the model has seen
  Level 4 — ADVERSARIAL    : Misleading distractors, partial-match traps,
                             negation, and out-of-distribution questions

For each probe the scorer shows:
  • Each model's raw softmax score for every candidate
  • Which model picks correctly
  • Confidence margin (correct score − best wrong score)

Then enters an interactive loop where you type any question + candidates
and see all model scores side-by-side.

Usage
-----
    python -m experiments.interactive_scorer
    python -m experiments.interactive_scorer --epochs 20 --no-interactive
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.epi_transformer import EpiTransformer
from src.models.epigenetic_network import EpigeneticNetwork
from src.models.baseline_transformer import BaselineTransformer
from src.models.baselines import BiLSTMClassifier, MLPClassifier
from src.data.qa_dataset import build_qa_loaders, QAVocabulary, _tokenise
from src.utils.config import get_default_config
from src.training.train import Trainer


# ─────────────────────────────────────────────────────────────────────────────
# Probe bank — 4 difficulty levels
# ─────────────────────────────────────────────────────────────────────────────

PROBES = {

    # ── Level 1: Exact training questions ────────────────────────────────────
    1: [
        {
            "q": "What is the capital of France?",
            "correct": "Paris",
            "distractors": ["London", "Berlin", "Madrid"],
            "note": "Exact training example — geography",
        },
        {
            "q": "Who wrote '1984'?",
            "correct": "George Orwell",
            "distractors": ["Aldous Huxley", "Ray Bradbury", "H.G. Wells"],
            "note": "Exact training example — literature",
        },
        {
            "q": "What is the chemical symbol for gold?",
            "correct": "Au",
            "distractors": ["Ag", "Fe", "Gd"],
            "note": "Exact training example — science",
        },
        {
            "q": "How many rings are on the Olympic flag?",
            "correct": "5",
            "distractors": ["4", "6", "7"],
            "note": "Exact training example — sport",
        },
        {
            "q": "What is the main ingredient in guacamole?",
            "correct": "Avocado",
            "distractors": ["Tomato", "Lime", "Onion"],
            "note": "Exact training example — food",
        },
    ],

    # ── Level 2: Paraphrase (same facts, different phrasing) ─────────────────
    2: [
        {
            "q": "Paris is the capital of which country?",
            "correct": "France",
            "distractors": ["England", "Germany", "Spain"],
            "note": "Inverted geography question",
        },
        {
            "q": "George Orwell is the author of which dystopian novel?",
            "correct": "1984",
            "distractors": ["Brave New World", "Fahrenheit 451", "Animal Farm"],
            "note": "Inverted literature question",
        },
        {
            "q": "Au is the chemical symbol for which element?",
            "correct": "Gold",
            "distractors": ["Silver", "Iron", "Copper"],
            "note": "Inverted science question — answer uses training vocab",
        },
        {
            "q": "The Olympic flag has how many rings?",
            "correct": "Five",
            "distractors": ["Four", "Six", "Seven"],
            "note": "Paraphrase with written number instead of digit",
        },
        {
            "q": "Avocado is the main ingredient in which dip?",
            "correct": "Guacamole",
            "distractors": ["Hummus", "Tzatziki", "Salsa"],
            "note": "Inverted food question",
        },
    ],

    # ── Level 3: Topic transfer (new questions, seen vocabulary) ─────────────
    3: [
        {
            "q": "What is the capital of Japan?",
            "correct": "Tokyo",
            "distractors": ["Osaka", "Kyoto", "Hiroshima"],
            "note": "Geography — different country, answer seen in training",
        },
        {
            "q": "Who wrote 'Brave New World'?",
            "correct": "Aldous Huxley",
            "distractors": ["George Orwell", "H.G. Wells", "Arthur C. Clarke"],
            "note": "Literature — different book, correct answer was a distractor",
        },
        {
            "q": "What is the chemical symbol for silver?",
            "correct": "Ag",
            "distractors": ["Au", "Fe", "Cu"],
            "note": "Science — different element, correct answer was a distractor",
        },
        {
            "q": "What sport is played at Wimbledon?",
            "correct": "Tennis",
            "distractors": ["Cricket", "Croquet", "Badminton"],
            "note": "Sport — exact training question (sanity check at level 3)",
        },
        {
            "q": "What is the primary ingredient in hummus?",
            "correct": "Chickpeas",
            "distractors": ["Lentils", "White beans", "Tofu"],
            "note": "Food — seen in training, correct answer was a distractor",
        },
    ],

    # ── Level 4: Adversarial ─────────────────────────────────────────────────
    4: [
        {
            "q": "What is NOT the capital of France?",
            "correct": "Berlin",
            "distractors": ["Paris", "Rome", "Brussels"],
            "note": "Negation trap — Paris is the obvious wrong choice",
        },
        {
            "q": "Who wrote '1984' and 'Animal Farm'?",
            "correct": "George Orwell",
            "distractors": ["Aldous Huxley", "Ray Bradbury", "H.G. Wells"],
            "note": "Extended question — extra info should not confuse",
        },
        {
            "q": "What is the capital of Australia?",
            "correct": "Canberra",
            "distractors": ["Sydney", "Melbourne", "Brisbane"],
            "note": "Common misconception trap — most people say Sydney",
        },
        {
            "q": "What does the chemical symbol Fe stand for?",
            "correct": "Iron",
            "distractors": ["Gold", "Silver", "Copper"],
            "note": "Fe seen as distractor in training, never as correct",
        },
        {
            "q": "Who invented the telephone?",
            "correct": "Alexander Graham Bell",
            "distractors": ["Thomas Edison", "Nikola Tesla", "Guglielmo Marconi"],
            "note": "Out-of-distribution — not in training set at all",
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_models(vocab_size: int, max_seq_len: int = 64) -> Dict[str, nn.Module]:
    return {
        "EpiTransformer    ": EpiTransformer(
            vocab_size=vocab_size, embed_dim=64, num_heads=4,
            num_layers=2, epigenetic_dim=32, num_classes=2,
            inner_dim=256, memory_size=64, memory_dim=64,
            epigenetic_alpha=0.3, memory_lambda=0.1,
            skip_weight=0.1, dropout=0.1, max_seq_len=max_seq_len,
        ),
        "EpigeneticNetwork ": EpigeneticNetwork(
            vocab_size=vocab_size, embed_dim=64, hidden_dims=[128, 64],
            epigenetic_dim=32, num_classes=2, memory_size=64, memory_dim=64,
            memory_lambda=0.1, epigenetic_alpha=0.3, num_heads=4,
            dropout=0.1, max_seq_len=max_seq_len,
        ),
        "Transformer       ": BaselineTransformer(
            vocab_size=vocab_size, embed_dim=64, num_heads=4,
            num_layers=2, num_classes=2, max_seq_len=max_seq_len, dropout=0.1,
        ),
        "BiLSTM            ": BiLSTMClassifier(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=64,
            num_layers=2, num_classes=2, dropout=0.1,
        ),
        "MLP               ": MLPClassifier(
            vocab_size=vocab_size, embed_dim=64, hidden_dim=128,
            num_classes=2, dropout=0.1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    models:       Dict[str, nn.Module],
    train_loader, val_loader,
    epochs: int,
    lr:     float,
) -> None:
    cfg = get_default_config()
    cfg.training.epochs        = epochs
    cfg.training.learning_rate = lr

    for name, model in models.items():
        trainer = Trainer(model, cfg)
        for ep in range(epochs):
            trainer.train_epoch(train_loader)
        final = trainer.evaluate(val_loader)
        print(f"  {name.strip():<22}  val_acc={final['accuracy']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(
    model: nn.Module,
    vocab: QAVocabulary,
    question:  str,
    candidate: str,
    max_len:   int = 64,
) -> float:
    """Return P(label=1) — probability the candidate is the correct answer."""
    model.eval()
    ids = vocab.encode(question, candidate, max_len)
    token_ids = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        if hasattr(model, "epigenetic_dim"):
            logits, _, _ = model(token_ids, e_t=None, update_memory=False)
        else:
            logits = model(token_ids)

    probs = F.softmax(logits[0], dim=-1)
    return probs[1].item()   # class-1 = correct


def score_all_models(
    models:    Dict[str, nn.Module],
    vocab:     QAVocabulary,
    question:  str,
    candidates: List[str],
    correct:   Optional[str] = None,
    max_len:   int = 64,
) -> Dict[str, Dict[str, float]]:
    """
    Returns { model_name: { candidate: score } } for every model.
    """
    return {
        name: {
            c: score_candidate(model, vocab, question, c, max_len)
            for c in candidates
        }
        for name, model in models.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

BAR_WIDTH = 20

def _bar(score: float) -> str:
    filled = int(round(score * BAR_WIDTH))
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def display_probe(
    question:    str,
    candidates:  List[str],
    correct:     Optional[str],
    scores:      Dict[str, Dict[str, float]],
    note:        str = "",
) -> None:
    print(f"\n  Q: {question}")
    if note:
        print(f"     [{note}]")
    if correct:
        print(f"     Correct answer: {correct}")

    # Header
    print()
    max_c = max(len(c) for c in candidates)

    for model_name, cand_scores in scores.items():
        # Find model's pick
        pick = max(cand_scores, key=cand_scores.get)
        is_right = (pick == correct) if correct else None
        marker = "✓" if is_right else ("✗" if is_right is False else " ")
        print(f"  [{marker}] {model_name.strip():<22}  pick: {pick:<{max_c}}  "
              f"margin: {_margin(cand_scores, correct):+.3f}")

        for cand, sc in sorted(cand_scores.items(), key=lambda x: -x[1]):
            indicator = "►" if cand == pick else " "
            star      = "✓" if cand == correct else " "
            print(f"          {indicator}{star} {cand:<{max_c}}  "
                  f"{_bar(sc)}  {sc:.4f}")
        print()


def _margin(scores: Dict[str, float], correct: Optional[str]) -> float:
    """correct_score − best_distractor_score"""
    if correct is None or correct not in scores:
        return 0.0
    correct_score = scores[correct]
    best_wrong    = max(v for k, v in scores.items() if k != correct)
    return correct_score - best_wrong


def level_summary(
    level:   int,
    probes:  List[dict],
    models:  Dict[str, nn.Module],
    vocab:   QAVocabulary,
) -> None:
    """Print per-model accuracy for one difficulty level."""
    counts  = {n: 0 for n in models}
    correct = {n: 0 for n in models}

    for probe in probes:
        all_c = [probe["correct"]] + probe["distractors"]
        sc = score_all_models(models, vocab, probe["q"], all_c, probe["correct"])
        for name, cand_scores in sc.items():
            pick = max(cand_scores, key=cand_scores.get)
            counts[name]  += 1
            correct[name] += int(pick == probe["correct"])

    print(f"\n  Level {level} accuracy:")
    for name in models:
        acc = correct[name] / max(counts[name], 1)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {name.strip():<22}  {bar}  {acc:.2f}  "
              f"({correct[name]}/{counts[name]})")


# ─────────────────────────────────────────────────────────────────────────────
# Run structured probes
# ─────────────────────────────────────────────────────────────────────────────

def run_structured_probes(models: Dict, vocab: QAVocabulary, verbose: bool = True) -> None:
    level_names = {
        1: "MEMORISATION   (exact training questions)",
        2: "PARAPHRASE     (same fact, different phrasing)",
        3: "TOPIC TRANSFER (new question, seen vocab)",
        4: "ADVERSARIAL    (traps, negation, OOD)",
    }

    for level, probes in PROBES.items():
        print("\n" + "=" * 70)
        print(f"  LEVEL {level} — {level_names[level]}")
        print("=" * 70)

        for probe in probes:
            all_c = [probe["correct"]] + probe["distractors"]
            sc = score_all_models(models, vocab, probe["q"], all_c, probe["correct"])
            if verbose:
                display_probe(probe["q"], all_c, probe["correct"], sc, probe["note"])

        level_summary(level, probes, models, vocab)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive loop
# ─────────────────────────────────────────────────────────────────────────────

def interactive_loop(models: Dict, vocab: QAVocabulary) -> None:
    print("\n" + "=" * 70)
    print("  INTERACTIVE SCORER")
    print("=" * 70)
    print("  Enter a question, then candidate answers one per line.")
    print("  Blank line after candidates = score them.")
    print("  Mark the correct answer with * e.g.  *Paris")
    print("  Type 'q' to quit.\n")

    while True:
        try:
            q = input("  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit", ""):
            break

        candidates, correct = [], None
        print("  Candidates (blank line when done, prefix correct with *):")
        while True:
            try:
                c = input("    > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if c == "":
                break
            if c.startswith("*"):
                correct = c[1:].strip()
                candidates.append(correct)
            else:
                candidates.append(c)

        if not candidates:
            print("  (no candidates entered)\n")
            continue

        sc = score_all_models(models, vocab, q, candidates, correct)
        display_probe(q, candidates, correct, sc)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print("  MULTI-MODEL SCORER — Training on Q&A dataset")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, vocab, meta = build_qa_loaders(
        batch_size=16, max_len=64, val_split=0.2, seed=args.seed,
    )
    print(f"\n  Dataset: {meta['n_total']} examples  |  vocab: {len(vocab)}")

    # ── Models ────────────────────────────────────────────────────────
    models = build_models(vocab_size=len(vocab), max_seq_len=64)
    print(f"\n  Building {len(models)} models...")
    for name, m in models.items():
        print(f"    {name.strip():<22}  {m.num_parameters():>8,} params")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n  Training ({args.epochs} epochs each)...")
    train_all(models, train_loader, val_loader, args.epochs, args.lr)

    # ── Structured probes ─────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STRUCTURED PROBE RESULTS")
    print("=" * 70)
    run_structured_probes(models, vocab, verbose=not args.summary_only)

    # ── Level summary table ───────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  OVERALL SUMMARY  (accuracy by level)")
    print("=" * 70)
    header = f"\n  {'Model':<22}  {'L1':>5}  {'L2':>5}  {'L3':>5}  {'L4':>5}  {'Total':>6}"
    print(header)
    print("  " + "─" * 55)

    level_accs = {name: [] for name in models}
    for level, probes in PROBES.items():
        for name, model in models.items():
            hits = sum(
                int(
                    max(
                        score_all_models(
                            {name: model}, vocab,
                            p["q"], [p["correct"]] + p["distractors"], p["correct"]
                        )[name],
                        key=lambda x: score_all_models(
                            {name: model}, vocab,
                            p["q"], [p["correct"]] + p["distractors"], p["correct"]
                        )[name][x],
                    ) == p["correct"]
                )
                for p in probes
            )
            level_accs[name].append(hits / len(probes))

    for name in models:
        accs   = level_accs[name]
        total  = sum(accs) / len(accs)
        bars   = "  ".join(f"{a:.2f}" for a in accs)
        print(f"  {name.strip():<22}  {bars}  {total:.2f}")

    print("\n  L1=Memorisation  L2=Paraphrase  L3=TopicTransfer  L4=Adversarial")

    # ── Interactive ───────────────────────────────────────────────────
    if not args.no_interactive:
        interactive_loop(models, vocab)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no-interactive", action="store_true",
                   help="Skip the interactive loop (useful for CI)")
    p.add_argument("--summary-only",   action="store_true",
                   help="Print only level summaries, not per-question detail")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
