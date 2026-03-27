# EpiNet — Epigenetic Neural Network

> *"Same DNA. Different expression. Context is everything."*

A research prototype implementing a novel neural architecture where neurons
**evolve their behaviour during inference**, inspired by epigenetic regulation
of gene expression in biology.

```
pip install -e ".[dev]"
python -m experiments.continual_learning
python -m experiments.context_adaptation
python -m pytest tests/ -v          # 122 tests, all green
```

---

## What Makes EpiNet Different

Standard neural networks are static at inference time — given the same input
they always produce the same output.  EpiNet neurons have two layers of control:

| Layer | Biology | EpiNet |
|-------|---------|--------|
| Stable | DNA sequence | Base weights **W, b** |
| Dynamic | Epigenetic marks (methylation) | Epigenetic state **e_t** |

The epigenetic state acts as a **soft switch bank** that can silence or amplify
individual neurons without changing any weights — exactly as chemical marks
toggle gene expression without altering the DNA sequence.

**Core capabilities enabled by this:**
- Same weights → different outputs for different contexts (zero extra params)
- Adapts during inference via a running state update (no gradient descent needed)
- Episodic memory accumulates context across batches
- Homeostasis prevents runaway activations under stress

---

## Architecture

```
  Input tokens / features
         │
         ▼
  ┌──────────────┐
  │   Encoder    │  Token embed + positional encoding  (or MLP for vectors)
  └──────┬───────┘
         │  x_summary ∈ R^(B × D)
         │
  ┌──────▼─────────────────────────────────────────────┐
  │  MemoryStore.read(x_summary)  →  memory_ctx        │
  │  Soft attention over M key-value slots             │
  └──────┬─────────────────────────────────────────────┘
         │                            ▲
         │                            │  write(hidden, importance)
         ▼                            │
  ┌──────────────────────────────────┐│
  │  EpigeneticLayer ×N             ││◄── e_t  (evolving state)
  │  ┌────────┐  ┌────────┐        ││
  │  │ Head 1 │  │ Head 2 │  ...   ││
  │  │        │  │        │        ││
  │  │  u=Wx  │  │  u=Wx  │        ││   Step 1: linear pre-activation
  │  │  g=σ(We│  │  g=σ(We│        ││   Step 2: epigenetic gate
  │  │  u'=g⊙u│  │  u'=g⊙u│        ││   Step 3: gated activation
  │  │  +λ·m  │  │  +λ·m  │        ││   Step 4: memory injection
  │  │  act() │  │  act() │        ││   Step 5: non-linearity
  │  └────────┘  └────────┘        ││
  │  concat → proj → LN + residual  │
  └──────────────────┬──────────────┘
                     │  h  (final hidden state)
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Epigenetic State Update                            │
  │  e_{t+1} = (1−α)·e_t  +  α · f(x, h, memory_ctx)  │
  └──────────────────┬──────────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────────┐
  │  Output Head  →  logits                             │
  └─────────────────────────────────────────────────────┘
```

---

## Mathematics

### Epigenetic Neuron Forward Pass

```
Given:
  x  ∈ R^d_in    input vector
  e  ∈ R^d_e     epigenetic state
  m  ∈ R^d_in    memory context

Step 1 — Linear pre-activation
    u   = W·x + b

Step 2 — Epigenetic gate  (the key innovation)
    g   = σ(W_e · e)           g ∈ (0,1)^d_out

Step 3 — Gated activation
    u'  = g ⊙ u                element-wise product

Step 4 — Memory injection
    u'' = u' + λ · proj(m)

Step 5 — Non-linearity
    y   = act(u'')
```

### State Dynamics (EMA)

```
e_{t+1} = (1 − α) · e_t  +  α · f(x_summary, h, memory_ctx)

α = 0  → rigid (state never changes)
α = 1  → reactive (no past memory)
α = 0.3 (default) → 70% persistence, 30% new signal
```

### Memory Read (Differentiable Attention)

```
q = W_q · x
k = W_k · keys

score_m = cosine(q, k_m) + softmax(importance_m)
attn    = softmax(score / τ)
ctx     = Σ_m  attn_m · values_m
```

---

## Repository Structure

```
EpiNet/
├── src/
│   ├── models/
│   │   ├── epigenetic_neuron.py       # Atomic unit: gate + memory + activation
│   │   ├── epigenetic_layer.py        # Multi-head layer with residual + LN
│   │   ├── epigenetic_network.py      # Full end-to-end network
│   │   └── baseline_transformer.py   # Standard transformer for comparison
│   ├── memory/
│   │   └── memory_store.py            # Key-value episodic memory (read/write/decay)
│   ├── controllers/
│   │   ├── epigenetic_controller.py   # State update, context save/restore
│   │   └── homeostasis.py             # Stress-adaptive learning rate regulation
│   ├── training/
│   │   ├── train.py                   # Trainer class + CLI entry point
│   │   └── evaluate.py                # Evaluation utilities + comparison table
│   ├── data/
│   │   └── dataset_loader.py          # Synthetic + continual + context datasets
│   └── utils/
│       ├── config.py                  # Dataclass config (model / memory / training)
│       └── metrics.py                 # Accuracy, forgetting, adaptation speed
├── experiments/
│   ├── continual_learning.py          # Task A → Task B forgetting experiment
│   └── context_adaptation.py          # Same input / different context routing
├── tests/                             # 122 unit + integration tests (all green)
│   ├── test_epigenetic_neuron.py
│   ├── test_epigenetic_layer.py
│   ├── test_memory_store.py
│   ├── test_epigenetic_network.py
│   ├── test_controllers.py
│   ├── test_training.py
│   └── test_data.py
├── reports/
│   └── performance_report.md          # Measured results + mathematical analysis
├── notebooks/
│   └── visualization.ipynb            # Gate heatmaps, state trajectories, memory util
├── results/                           # Experiment JSON outputs
├── README.md
├── requirements.txt
└── setup.py
```

---

## Quick Start

### Install

```bash
git clone https://github.com/yugantm/epinet.git
cd epinet
pip install -e ".[dev]"
```

### Train on synthetic sentiment (single task)

```bash
# EpigeneticNetwork
python -m src.training.train --model epinet --epochs 5 --n_samples 2000

# Baseline Transformer (for comparison)
python -m src.training.train --model transformer --epochs 5 --n_samples 2000
```

---

## Experiments

### Experiment 1 — Continual Learning

Train on Task A, then Task B without resetting weights.
Measure how much Task A knowledge is retained.

```bash
python -m experiments.continual_learning --epochs 5 --n_samples 2000

# Ablation: clear EpiNet memory at task boundary
python -m experiments.continual_learning --reset_memory
```

**Protocol:**
```
1. Train both models on Task A (sentiment: positive vs negative)
2. Evaluate on Task A  → acc_A_before
3. Continue training on Task B (topic: tech vs sports) — no weight reset
4. Evaluate on Task A  → acc_A_after
5. Forgetting score F = acc_A_before − acc_A_after  (lower = better)
```

**Measured results:**

| Model | Task-A (after B) | Forgetting | Speed | Params |
|-------|-----------------|------------|-------|--------|
| EpigeneticNetwork | 0.83 | 0.13 | **0.7s/epoch** | 87k |
| BaselineTransformer | 0.95 | 0.05 | 2.7s/epoch | 110k |

EpiNet is **3.9× faster per epoch** because it avoids O(T²) self-attention in
its layer stack (the Transformer runs it once per block, per sequence position).
On this small synthetic benchmark the Transformer forgets less; EpiNet's retention
advantage is expected to emerge on longer sequences and with task-boundary signalling.

---

### Experiment 2 — Context Adaptation

Show that identical token sequences produce different predictions depending on
the epigenetic state — impossible for a stateless Transformer without re-encoding
the context as tokens.

```bash
python -m experiments.context_adaptation --epochs 8 --n_samples 1600
```

**Protocol:**
```
Each sample presented twice:
  context = "formal"  →  expected label = 1
  context = "casual"  →  expected label = 0
  tokens: identical

Context sensitivity = P(pred_formal ≠ pred_casual | same tokens)
```

**Measured results:**

| Model | Val Accuracy | Context Sensitivity |
|-------|-------------|---------------------|
| ContextualEpiNet | **1.00** | **1.00** (100% flip) |
| BaselineTransformer | 1.00 | 0.00 (no mechanism) |

Every single sample flipped its prediction when the context changed while
tokens stayed identical.  This is the **core capability** of EpiNet: same DNA,
different phenotype.

```
Sample 1: formal=1 | casual=0  ← FLIP
Sample 2: formal=1 | casual=0  ← FLIP
Sample 3: formal=1 | casual=0  ← FLIP
Sample 4: formal=1 | casual=0  ← FLIP
Sample 5: formal=1 | casual=0  ← FLIP
```

---

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing
```

**122 tests, all passing.**

| Test file | Count | What is verified |
|-----------|-------|-----------------|
| `test_epigenetic_neuron` | 20 | Gate bounds (0,1), memory impact, gradient flow, all activations |
| `test_epigenetic_layer` | 10 | Multi-head aggregation, residual connection, LayerNorm stats |
| `test_memory_store` | 13 | Read/write shapes, LRU eviction, decay, reset, differentiability |
| `test_epigenetic_network` | 22 | End-to-end forward, state evolution, memory write, training loop |
| `test_controllers` | 22 | EMA correctness (`e_next = 0.3` verified), context save/restore, homeostasis |
| `test_training` | 18 | Full training loop, checkpoint save+load roundtrip, all metric formulas |
| `test_data` | 17 | Vocabulary, special tokens, dataset balance, DataLoader shapes |

---

## Key Design Decisions

**Why detach `e_t` between batches?**
The epigenetic state is a running average across batches, not a differentiable
recurrence.  Detaching prevents gradient accumulation across batch boundaries
(which would be BPTT and incompatible with standard training).

**Why `clone().detach()` on memory buffers?**
PyTorch's version counter tracks in-place modifications.  The `write()` method
modifies `keys[slot]` in-place; if the same buffer was used in the forward pass
without cloning, backward would see a version mismatch.  Cloning before the
forward computation creates an independent copy that is safe to graph-compute on.

**Why a fixed-size memory with LRU eviction?**
Biological working memory has bounded capacity.  Fixed-size memory also avoids
unbounded memory growth during long inference runs.  Least-importance eviction
(analogous to forgetting unimportant events) is a simple but effective policy.

---

## Extending EpiNet

| Goal | Entry point |
|------|-------------|
| New activation | Add to `EpigeneticNeuron._ACTIVATIONS` |
| Different memory policy | Override `MemoryStore.write()` |
| Hierarchical state | Stack `EpigeneticController` instances |
| Custom state update | Subclass `EpigeneticNetwork`, override `update_epigenetic_state()` |
| New dataset | Implement in `src/data/dataset_loader.py`, add to `build_dataloaders()` |

---

## Future Work

| Direction | What it enables |
|-----------|----------------|
| Meta-learning for `f(·)` | Few-shot adaptation with MAML / Reptile |
| Persistent disk memory | Cross-session episodic continuity |
| Binary epigenetic gates | Spiking-compatible hardware deployment |
| Attention-weighted state | Richer temporal dependencies than EMA |
| Benchmark: Permuted/Split-MNIST | Standardised continual learning comparison |
| Domain-shift NLP tasks | Real-world context-sensitivity demonstration |

---

## License

MIT
