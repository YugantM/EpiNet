# EpiNet

> The name comes from epigenetics — the biological observation that the same
> gene sequence can produce radically different cell behaviour depending on
> which genes are currently active. That idea — **shared weights, context-
> dependent routing** — is what this architecture implements in ML terms.

EpiNet is a neural network where every forward pass is conditioned on a
**persistent modulation state** that evolves across batches without gradient
descent. The weights stay fixed after training; behaviour changes because a
separate state vector continuously re-gates which neurons are active.

```bash
pip install -e ".[dev]"
python -m experiments.benchmark          # 4-model head-to-head
python -m experiments.context_adaptation # context routing demo
python -m pytest tests/ -v               # 122 tests, all green
```

---

## Benchmark Results

CPU only · Intel x86-64 · synthetic dataset · 8 epochs · 3 000 samples · seq-len 64

### Continual Learning — Task A then Task B, no weight reset

**Forgetting F = acc\_A\_before − acc\_A\_after** — lower is better.

| Model | Params | Task-A | Task-B | Forgetting ↓ | s / epoch |
|-------|-------:|:------:|:------:|:------------:|----------:|
| EpigeneticNetwork | 86 947 | 1.0000 | 1.0000 | 0.60 | **0.96 s** |
| Transformer | 110 114 | 1.0000 | 1.0000 | 0.51 | 4.00 s |
| BiLSTM | 178 690 | 1.0000 | 1.0000 | **0.14** | 3.71 s |
| MLP | 20 994 | 1.0000 | 1.0000 | 0.25 | 0.23 s |

EpiNet is **4.2× faster per epoch** than Transformer because it has no O(T²)
self-attention in its layer stack. On the naive sequential benchmark BiLSTM
forgets least; EpiNet forgets most — see [Why](#why-epinet-forgets-more-here)
below for the exact reason and when this reverses.

### Context Routing — identical tokens, different modulation state

**Context sensitivity** = fraction of samples where the prediction flips when
the modulation state is changed while the token input stays identical.

| Model | Val Accuracy | Context Sensitivity |
|-------|:-----------:|:-------------------:|
| EpigeneticNetwork | **1.0000** | **1.0000** (100 %) |
| Transformer | 1.0000 | 0.0000 |
| BiLSTM | 1.0000 | 0.0000 |
| MLP | 1.0000 | 0.0000 |

Stateless models cannot distinguish contexts without re-encoding context as
extra tokens. EpiNet routes the same input to a completely different prediction
purely by changing the modulation state — no extra parameters, no extra tokens.

```
sample 1:  context_A → class 1  |  context_B → class 0   ← flip
sample 2:  context_A → class 1  |  context_B → class 0   ← flip
sample 3:  context_A → class 1  |  context_B → class 0   ← flip
sample 4:  context_A → class 1  |  context_B → class 0   ← flip
sample 5:  context_A → class 1  |  context_B → class 0   ← flip
```

### Why EpiNet forgets more here

The benchmark dataset has only 65 tokens. Every model saturates at 100 % on
each task individually within 1 epoch (ceiling effect). Under ceiling conditions
the Task B gradient signal is large and overwrites representations across the
entire weight matrix regardless of architecture.

EpiNet has an additional factor: the modulation state `e_t` drifts toward
Task B's distribution during Task B training. At Task A evaluation time, the
evolved `e_t` gates the network into Task B mode even though the weights have
not changed. This is not a flaw in the gating mechanism — it is the gating
mechanism working correctly in a setting it was not designed for (unconstrained
sequential fine-tuning).

**The intended usage for continual learning is:**
1. Save the Task A modulation context: `controller.save_context(e_t, "task_a")`
2. Reset memory at task boundary: `model.reset_memory()`
3. Restore context at evaluation: `e_t = controller.restore_context("task_a")`

With that protocol the weight updates still interfere (same as any other model)
but the routing state is isolated per task, which is not possible in any
stateless architecture.

---

## Architecture

The network is a linear pipeline. Each block has a specific job:

```
  token_ids or feature vector
          │
          ▼
  ┌───────────────────────────────────────────────────────┐
  │  1. Encoder                                           │
  │     token_ids → embedding + positional encoding       │
  │     → mean-pool → x_summary  ∈ R^(B × D)             │
  └──────────────────────┬────────────────────────────────┘
                         │
  ┌──────────────────────▼────────────────────────────────┐
  │  2. Memory Read                                       │
  │     query = x_summary                                 │
  │     cosine-attention over M stored (key, value) pairs │
  │     → memory_ctx  ∈ R^(B × D)                        │
  └──────────────────────┬──────────────────────┬─────────┘
                         │                      │ write after forward
  ┌──────────────────────▼──────────────────────┤
  │  3. Modulated Layer Stack (×N)              │◄── modulation state e_t
  │                                             │
  │  for each head:                             │
  │    u   = W·x + b          linear transform  │
  │    g   = σ(W_e · e_t)     gate from state   │
  │    u'  = g ⊙ u            scale activations │
  │    u'' = u' + λ·m         inject memory     │
  │    y   = act(u'')         non-linearity     │
  │                                             │
  │  heads concat → project → LayerNorm + skip  │
  └──────────────────────┬──────────────────────┘
                         │  h  (hidden state)
  ┌──────────────────────▼────────────────────────────────┐
  │  4. State Update                                      │
  │     e_{t+1} = (1−α)·e_t + α·f(x_summary, h, mem_ctx) │
  │     f is a learned single-layer network               │
  │     α = 0.3 → 70% of previous state is preserved     │
  └──────────────────────┬────────────────────────────────┘
                         │
  ┌──────────────────────▼────────────────────────────────┐
  │  5. Output Head                                       │
  │     Linear → GELU → Linear → logits                  │
  └───────────────────────────────────────────────────────┘
```

---

## What Each Block Does and Why

### 1. Encoder

**What:** Maps token indices to dense vectors via a learned embedding table,
adds sinusoidal positional encodings, then mean-pools the sequence into a
single vector `x_summary`.

**Why it comes first:** Everything downstream operates on a fixed-size vector.
Mean-pooling is used instead of a CLS token to keep the encoder stateless and
fast. For non-text inputs an MLP encoder is swapped in automatically.

**Comparable systems:** Standard in every transformer (BERT, GPT), CNNs,
and RNNs. This block is not novel.

---

### 2. Memory Read (MemoryStore)

**What:** A fixed-size bank of M (key, value) pairs. Given `x_summary` as a
query, it computes cosine-similarity scores against all stored keys, weights
them by a per-slot importance score, and returns a weighted sum of values as
`memory_ctx`. Slots decay in importance over time; the least-important slot
is evicted on each write.

**Why it comes before the layer stack:** The layer stack needs context before
it computes. Reading memory first means every neuron gate and every linear
transform can be influenced by accumulated past inputs. If memory were read
after the layers, it could only affect the output head.

**Why this ordering matters specifically:** The gate in step 3 is conditioned
on `e_t`, and `e_t` is updated using `memory_ctx` (step 4). If memory were
read after the layers, the state update would have no memory signal and `e_t`
would only ever track the current input — losing the cross-batch accumulation
property.

**Comparable systems:** Neural Turing Machine (Graves et al., 2014),
Differentiable Neural Computer (Graves et al., 2016), Memory Networks
(Weston et al., 2015). The difference here is that the memory is read *before*
the main computation, not after, so it conditions gating rather than
augmenting output.

---

### 3. Modulated Layer Stack (the core)

**What:** N stacked layers, each containing H parallel heads. Every head
applies five operations in order:

| Step | Operation | Formula | What it does |
|------|-----------|---------|--------------|
| 1 | Linear transform | `u = W·x + b` | Standard affine projection |
| 2 | Gate from state | `g = σ(W_e · e_t)` | Each neuron gets a scalar in (0,1) from the modulation state |
| 3 | Apply gate | `u' = g ⊙ u` | Neurons with g≈0 are suppressed; g≈1 pass through |
| 4 | Inject memory | `u'' = u' + λ·m` | Additive memory contribution, scaled by λ |
| 5 | Non-linearity | `y = act(u'')` | ReLU by default |

Heads are concatenated, projected to the output dimension, then added to a
residual connection and normalised.

**Why this sequence specifically:**

- Step 1 before step 2: the linear transform is the stable, trained
  computation. The gate (step 2) is applied *after* so it multiplies the
  result of the learned weights, not the raw input — suppressing an output
  neuron, not an input feature.
- Step 2 produces a gate per *output* neuron, not per input. This means the
  gate selectively silences individual output dimensions of the projection
  rather than masking input channels.
- Step 4 after step 3: memory is added *after* gating so the memory signal
  is not itself gated. A suppressed neuron (g≈0) can still receive a memory
  contribution. This keeps the memory channel independent of the modulation
  state — the two influences are additive, not multiplicative.
- Residual + LayerNorm last: standard for training stability, same rationale
  as in transformer blocks.

**What is novel here:** The gate `g = σ(W_e · e_t)` conditioning on a
*separate, evolving state vector* rather than on the input `x` is the
distinguishing mechanism. The result is that the same weight matrix `W` computes
different effective functions at different points in time without any weight
update. This is different from:

- **LSTM/GRU gates** — those gate the hidden state update, not the output
  of a linear projection; they also have no separate persistent modulation
  vector.
- **FiLM (Feature-wise Linear Modulation, Perez et al. 2018)** — closest
  analogy; FiLM applies `γ·x + β` where γ, β come from a conditioning network.
  Here the conditioning is multiplicative-only (`g ⊙ u`) and the conditioning
  signal `e_t` is not computed fresh each step — it persists and evolves
  across batches via EMA.
- **HyperNetworks (Ha et al. 2016)** — a secondary network generates weights
  for the primary network. Here no new weights are generated; only a per-neuron
  scalar mask is produced from a persistent state.
- **Highway Networks (Srivastava et al. 2015)** — gate depends on the input
  `x`, not on an external state. The gate resets completely for every new
  input.

---

### 4. State Update

**What:** An exponential moving average (EMA) over a learned transformation
of the current input, hidden state, and memory context:

```
e_{t+1} = (1 − α) · e_t  +  α · f(x_summary, h, memory_ctx)

f: R^(D + H + M)  →  R^(d_e)    (single linear layer + tanh)
α = 0.3  →  each step moves 30% toward the new signal
```

**Why after the layer stack and not before:** `f` takes the hidden state `h`
as input — which is only available after the layer stack runs. The state
update therefore requires the forward pass to complete first. The updated
`e_{t+1}` is returned to the caller and used as `e_t` for the *next* batch,
not the current one.

**What this achieves:** `e_t` is a compressed, decaying summary of everything
the network has seen so far (inputs, hidden activations, memory reads). It
changes continuously across batches without any gradient descent. This is
inference-time adaptation: a model with fixed weights behaving differently
on batch 100 than on batch 1 because its modulation state has shifted.

**Comparable systems:** GRU hidden state update has the same EMA structure
(`h_t = (1−z)·h_{t-1} + z·ĥ`). The difference is that in a GRU the hidden
state *is* the representation; here `e_t` is a separate control signal that
modulates the main representation pathway. The two are decoupled.

---

### 5. Output Head

**What:** Two linear layers with GELU activation and dropout, mapping the
final hidden state `h` to class logits.

**Why:** Standard classification head. Nothing novel. Kept simple
intentionally so that all benchmark differences are attributable to the
layer stack and modulation mechanism, not head capacity.

---

## What Is Novel vs What Is Existing

| Component | Status | Closest existing technique |
|-----------|--------|---------------------------|
| Encoder | Existing | Standard in BERT, GPT, etc. |
| Fixed-size K/V memory with importance-weighted LRU | Existing | DNC, NTM |
| Per-neuron multiplicative gate from input | Existing | Highway Networks |
| Per-neuron gate from a *conditioning* signal | Existing | FiLM |
| EMA state update | Existing | GRU update gate |
| **Gate conditioned on a persistent cross-batch state `e_t`** | **Novel combination** | None directly |
| **`e_t` updated from (input + hidden + memory) jointly** | **Novel combination** | None directly |
| **Memory read *before* gating so it conditions the gate** | **Novel ordering** | DNC reads after layers |
| **Same weights, different routing via `e_t` at inference** | **Emergent capability** | Not available in stateless models |

The individual operations (linear projection, sigmoid gate, EMA, attention
over K/V slots) all exist. The novelty is the **specific composition**: a
persistent modulation state that is conditioned on memory, applied before
the main computation, and updated after it — creating a feedback loop that
lets the network's effective function drift over time without any weight change.

---

## Mathematics

```
Modulated neuron:
  u   = W·x + b                     linear pre-activation
  g   = σ(W_e · e_t)   g∈(0,1)^d   gate from modulation state
  u'  = g ⊙ u                       scale each output neuron
  u'' = u' + λ · proj(m)            add memory, scaled by λ
  y   = act(u'')                    non-linearity

State update (EMA):
  e_{t+1} = (1−α)·e_t + α·tanh(W_f · [x; h; m])

Memory read (cosine attention):
  score_i = cosine(W_q·x, W_k·key_i) + importance_i
  attn    = softmax(score / τ)
  ctx     = Σ_i  attn_i · value_i
```

---

## Repository Structure

```
EpiNet/
├── src/
│   ├── models/
│   │   ├── epigenetic_neuron.py       # Steps 1-5 of the modulated neuron
│   │   ├── epigenetic_layer.py        # Multi-head aggregation + residual
│   │   ├── epigenetic_network.py      # Full pipeline (blocks 1–5 above)
│   │   ├── baseline_transformer.py   # Standard Transformer
│   │   └── baselines.py              # BiLSTM, MLP
│   ├── memory/
│   │   └── memory_store.py            # Block 2: K/V memory with LRU eviction
│   ├── controllers/
│   │   ├── epigenetic_controller.py   # Block 4: state update + context save/restore
│   │   └── homeostasis.py             # Adaptive LR scaling from gate activity
│   ├── training/
│   │   ├── train.py                   # Trainer + CLI
│   │   └── evaluate.py                # Evaluation + comparison table
│   ├── data/
│   │   └── dataset_loader.py          # Synthetic sentiment + topic + context datasets
│   └── utils/
│       ├── config.py                  # Dataclass config
│       └── metrics.py                 # Accuracy, forgetting, adaptation speed
├── experiments/
│   ├── benchmark.py                   # 4-model head-to-head
│   ├── benchmark_extended.py          # + EpiNet+Boundary variant
│   ├── continual_learning.py          # Sequential task experiment
│   └── context_adaptation.py          # Context routing experiment
├── tests/                             # 122 tests, all passing
├── reports/performance_report.md      # Raw numbers + analysis
├── notebooks/visualization.ipynb      # Gate activations, state trajectories
├── results/                           # JSON outputs
├── requirements.txt
└── setup.py
```

---

## Quick Start

```bash
git clone https://github.com/yugantm/epinet.git
cd epinet
pip install -e ".[dev]"

# 4-model benchmark (~5 min CPU)
python -m experiments.benchmark --epochs 8 --n_samples 3000

# Context routing
python -m experiments.context_adaptation --epochs 8 --n_samples 1600

# Continual learning
python -m experiments.continual_learning --epochs 5 --n_samples 2000

# Single model training
python -m src.training.train --model epinet --epochs 5
python -m src.training.train --model transformer --epochs 5
```

---

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing
```

**122 tests, all passing.**

| File | Tests | Covers |
|------|------:|--------|
| `test_epigenetic_neuron` | 20 | Gate bounds (0,1), memory impact, all activations, gradients |
| `test_epigenetic_layer` | 10 | Multi-head, residual, LayerNorm stats |
| `test_memory_store` | 13 | Read/write, LRU eviction, decay, differentiability |
| `test_epigenetic_network` | 22 | End-to-end forward, state evolution, training loop |
| `test_controllers` | 22 | EMA update correctness, context save/restore, homeostasis |
| `test_training` | 18 | Training loop, checkpoint roundtrip, all metric formulas |
| `test_data` | 17 | Vocabulary, dataset balance, DataLoader shapes |

---

## Design Notes

**Why detach `e_t` between batches?**
`e_t` is a running average, not a differentiable recurrence. Detaching it
prevents gradient accumulation across batch boundaries (BPTT), keeping
training standard single-step backprop.

**Why `clone().detach()` on memory buffers before the forward pass?**
`write()` does an in-place slot assignment on the `keys` buffer during the
same forward pass that read it. PyTorch's version counter would see a mutation
and fail the backward pass. Cloning before the forward creates an independent
tensor that the graph can safely differentiate through.

**Why fixed-size memory with LRU eviction?**
Unbounded memory grows linearly with the number of forward passes, which is
impractical for long inference runs. Fixed capacity forces the model to
prioritise what it retains, analogous to a cache.

---

## Extending

| Goal | Where to change |
|------|----------------|
| New activation function | `EpigeneticNeuron._ACTIVATIONS` dict |
| Different memory eviction policy | `MemoryStore.write()` |
| Different state update (e.g. GRU-style) | `EpigeneticNetwork.update_epigenetic_state()` |
| Multi-level modulation state | Stack `EpigeneticController` instances |
| New classification task | `src/data/dataset_loader.py` → `build_dataloaders()` |

---

## Future Work

| Direction | What it unlocks |
|-----------|----------------|
| Task-boundary auto-detection | Gate the context save/restore on distribution shift, removing manual calls |
| MAML / Reptile on `f(·)` | Train the state update network for fast few-shot adaptation |
| Persistent memory across sessions | Checkpoint the K/V store to disk |
| Evaluate on Permuted / Split-MNIST | Standard continual learning leaderboard |
| Variable-length sequence benchmark | Show O(M) vs O(T²) speed gap at T=512+ |
| Multi-context NLI dataset | Real-world context-routing evaluation |

---

## License

MIT
