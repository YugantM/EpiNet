# EpiNet — Epigenetic Neural Network

> A novel neural network architecture where neurons evolve their behaviour
> during inference, inspired by the epigenetic regulation of gene expression.

---

## Table of Contents

1. [Concept](#1-concept)
2. [Architecture](#2-architecture)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Repository Structure](#4-repository-structure)
5. [Quick Start](#5-quick-start)
6. [Running Experiments](#6-running-experiments)
7. [Test Suite](#7-test-suite)
8. [Expected Results](#8-expected-results)
9. [Future Improvements](#9-future-improvements)

---

## 1. Concept

### Genetics vs Epigenetics — the Analogy

| Biology | EpiNet |
|---------|--------|
| DNA sequence (unchanging) | Base weights **W, b** (trained, then fixed) |
| Gene expression (on/off) | Epigenetic gate **g = σ(W_e · e)** |
| Epigenetic marks (methylation) | Epigenetic state vector **e_t** |
| Environmental response | State update **e_{t+1} = f(input, hidden, memory)** |
| Cellular memory | Key-value **MemoryStore** |
| Homeostasis | **HomeostasisModule** (stress-adaptive LR) |

In biology, two cells with identical DNA can behave completely differently
because of *epigenetic* marks — chemical modifications that silence or amplify
gene expression without altering the underlying sequence.

EpiNet replicates this: every neuron has **stable base weights** (the "DNA")
and a **dynamic gate** controlled by the epigenetic state.  The state evolves
during inference via an exponential moving average, giving the model a form of
short-term plasticity that does not require gradient descent.

---

## 2. Architecture

```
                         ┌────────────────────────────────────────────┐
                         │            EpigeneticNetwork               │
                         └────────────────────────────────────────────┘

  Input tokens / features
         │
         ▼
  ┌──────────────┐
  │   Encoder    │  Token embeddings + positional encoding (TextEncoder)
  │   (TextEnc   │  OR dense MLP (MLPEncoder)
  │   /MLPEnc)   │
  └──────┬───────┘
         │  x_summary ∈ R^(B × D)
         │
  ┌──────▼───────────────────────────────────────────────┐
  │  MemoryStore.read(x_summary)  → memory_ctx           │
  │  (soft attention over key-value slots)               │
  └──────┬───────────────────────────────────────────────┘
         │                               ▲
         │                               │  write(h, importance)
         ▼                               │
  ┌──────────────────────────────────────┴──────────────┐
  │  EpigeneticLayer 1                                  │◄── e_t (epigenetic state)
  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │
  │  │ Head 1 │  │ Head 2 │  │ Head 3 │  │ Head 4 │    │
  │  │EpiNrn  │  │EpiNrn  │  │EpiNrn  │  │EpiNrn  │    │
  │  └────────┘  └────────┘  └────────┘  └────────┘    │
  │        concat → output_proj → LayerNorm + residual  │
  └──────────────────────┬──────────────────────────────┘
                         │  h₁
  ┌──────────────────────▼──────────────────────────────┐
  │  EpigeneticLayer 2  (same structure)                │◄── e_t
  └──────────────────────┬──────────────────────────────┘
                         │  h  (final hidden state)
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │  EpigeneticStateUpdate                              │
  │  e_{t+1} = (1-α)·e_t + α·f(x_summary, h, mem_ctx)  │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │  OutputHead: Linear → GELU → Linear → logits        │
  └─────────────────────────────────────────────────────┘
```

### EpigeneticNeuron (atomic unit)

```
Step 1 — Linear pre-activation:   u   = W·x + b
Step 2 — Epigenetic gate:         g   = σ(W_e · e)      g ∈ (0,1)^d_out
Step 3 — Gated activation:        u'  = g ⊙ u
Step 4 — Memory injection:        u'' = u' + λ·proj(m)
Step 5 — Non-linearity:           y   = act(u'')
```

---

## 3. Mathematical Formulation

### Epigenetic Gate

```
g_i = σ(W_e^i · e)    ∀ i ∈ {1, …, d_out}
```

The gate `g_i` acts as a *soft switch* on neuron `i`.  When `g_i ≈ 0` the
neuron is silenced regardless of its base weight `W^i`.  When `g_i ≈ 1` the
neuron fires at its "natural" level.

### State Update (EMA)

```
e_{t+1} = (1 − α) · e_t  +  α · f(x_summary, h, memory_ctx)

where  f: R^{D + H + M} → R^{d_e}  is a learned 1-layer network.
```

- `α = 0`  → completely rigid (no adaptation)
- `α = 1`  → completely reactive (no memory of past)
- `α = 0.3` (default) → 70% persistence, 30% new signal

### Memory Read (Soft Attention)

```
q = W_q · x                               (query projection)
k = W_k · keys                            (key projection)

score_m = cosine(q, k_m) + softmax(importance_m)

attn = softmax(score / τ)                 (τ = learned temperature)

memory_ctx = Σ_m  attn_m · values_m
```

### Forgetting Score (Continual Learning)

```
F = acc_A_before  −  acc_A_after

F > 0  →  model forgot Task A after learning Task B
F < 0  →  model improved on Task A (positive transfer)
F = 0  →  perfect retention
```

### Backward Transfer

```
BWT_task  =  acc_task_final  −  acc_task_initial
```

---

## 4. Repository Structure

```
EpiNet/
├── src/
│   ├── models/
│   │   ├── epigenetic_neuron.py      # Atomic EpigeneticNeuron
│   │   ├── epigenetic_layer.py       # Multi-head EpigeneticLayer
│   │   ├── epigenetic_network.py     # Full end-to-end EpigeneticNetwork
│   │   └── baseline_transformer.py  # Comparison Transformer
│   ├── memory/
│   │   └── memory_store.py           # Differentiable key-value memory
│   ├── controllers/
│   │   ├── epigenetic_controller.py  # State update & context management
│   │   └── homeostasis.py            # Adaptive LR regulation
│   ├── training/
│   │   ├── train.py                  # Trainer class + CLI entry point
│   │   └── evaluate.py               # Evaluation utilities
│   ├── data/
│   │   └── dataset_loader.py         # Synthetic + task datasets
│   └── utils/
│       ├── config.py                 # Dataclass-based config system
│       └── metrics.py                # Accuracy, forgetting, stability
├── experiments/
│   ├── continual_learning.py         # Task A → Task B forgetting experiment
│   └── context_adaptation.py         # Same input / different context
├── tests/
│   ├── test_epigenetic_neuron.py     # 20 unit tests
│   ├── test_epigenetic_layer.py      # 10 unit tests
│   ├── test_memory_store.py          # 13 unit tests
│   ├── test_epigenetic_network.py    # 22 integration tests
│   ├── test_controllers.py           # 22 unit tests
│   ├── test_training.py              # 18 integration + metric tests
│   └── test_data.py                  # 17 data pipeline tests
├── results/                          # Auto-generated experiment outputs
├── reports/
│   └── performance_report.md         # Benchmark analysis
├── notebooks/
│   └── visualization.ipynb           # Interactive exploration
├── README.md
├── requirements.txt
└── setup.py
```

---

## 5. Quick Start

### Install

```bash
git clone https://github.com/yugantm/epinet.git
cd epinet
pip install -e ".[dev]"
```

### Train EpiNet (single task)

```bash
python -m src.training.train \
    --model epinet \
    --task sentiment \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --n_samples 2000
```

### Train Baseline Transformer

```bash
python -m src.training.train \
    --model transformer \
    --task sentiment \
    --epochs 5
```

---

## 6. Running Experiments

### Experiment 1 — Continual Learning

```bash
python -m experiments.continual_learning \
    --epochs 5 \
    --n_samples 2000
```

**What it does:**
1. Trains both models on Task A (sentiment classification).
2. Evaluates Task A performance.
3. Continues training on Task B (topic classification) — no weight reset.
4. Re-evaluates Task A.
5. Computes forgetting score `F = acc_A_before − acc_A_after`.

### Experiment 2 — Context Adaptation

```bash
python -m experiments.context_adaptation \
    --epochs 8 \
    --n_samples 1600
```

**What it does:**
1. Creates paired samples with two contexts: `formal` / `casual`.
2. Trains ContextualEpiNet: context → initial `e_0` → different predictions.
3. Measures *context sensitivity*: fraction of samples that flip prediction
   when context changes while token input remains identical.
4. Baseline Transformer has no context mechanism → sensitivity = 0.

---

## 7. Test Suite

```bash
# Run all 122 tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

**Test breakdown:**

| Module | Tests | Coverage areas |
|--------|-------|----------------|
| `test_epigenetic_neuron` | 20 | Shapes, gates, memory, gradients, activations |
| `test_epigenetic_layer`  | 10 | Multi-head, residual, LayerNorm, gradients |
| `test_memory_store`      | 13 | Read/write, eviction, decay, reset, grad flow |
| `test_epigenetic_network`| 22 | End-to-end, state evolution, memory, training |
| `test_controllers`       | 22 | EMA update, context save/restore, homeostasis |
| `test_training`          | 18 | Training loop, checkpoint, metrics |
| `test_data`              | 17 | Vocabulary, dataset construction, loaders |

---

## 8. Expected Results

### Continual Learning

| Model | Task-A Acc (after B) | Forgetting (F) | Params |
|-------|---------------------|----------------|--------|
| EpigeneticNetwork | 0.83 | 0.13 | 87k |
| BaselineTransformer | 0.95 | 0.05 | 110k |

*Note:* On this small synthetic dataset both models achieve high accuracy on
each individual task, and the Transformer (with its higher capacity and
cleaner gradient flow) forgets less.  On longer, more complex continual
learning benchmarks (Split-MNIST, Permuted-MNIST, NLP domain shift) the
EpigeneticNetwork's memory and adaptive gating are expected to confer greater
retention advantages.  See `reports/performance_report.md` for a detailed
analysis.

### Context Adaptation

| Model | Val Accuracy | Context Sensitivity |
|-------|-------------|---------------------|
| ContextualEpiNet | **1.00** | **1.00** (100% flip) |
| BaselineTransformer | 1.00 | N/A (0.00) |

The key result: **100% of samples produce different predictions under
different contexts** while tokens remain identical — demonstrating that the
epigenetic state acts as a true context-modulation signal.  A standard
Transformer cannot achieve this without token-level context injection.

---

## 9. Future Improvements

| Direction | Description |
|-----------|-------------|
| **Spiking Epigenetic Neurons** | Replace sigmoid gates with binary (Heaviside) spikes for event-driven computation |
| **Meta-learning integration** | Use MAML / Reptile to train the epigenetic update network `f(·)` for faster adaptation |
| **Persistent memory across sessions** | Checkpoint the MemoryStore to disk for truly long-term episodic memory |
| **Attention-weighted state update** | Replace EMA with a learned attention mechanism over past states |
| **Continual benchmark evaluation** | Test on Split-MNIST, Permuted-MNIST, Split-CIFAR-10 for fair comparison |
| **Hierarchical epigenetic states** | Separate short-term (layer-level) and long-term (network-level) states |
| **Neuromodulator signals** | Add dopamine/serotonin analogues to modulate the alpha and lambda hyperparameters dynamically |
| **GPU scaling** | Add `torch.cuda` device awareness for large-scale experiments |

---

## License

MIT
