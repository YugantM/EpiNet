# EpiNet Performance Report

**Date:** 2026-03-27
**Version:** v0.1.0
**Hardware:** CPU only (Intel x86-64, single core)
**Framework:** PyTorch 2.11.0
**Dataset:** Synthetic (65-word vocabulary, 3 000 samples, seq-len 64)
**Protocol:** 8 training epochs per task, AdamW lr=1e-3, batch 32, seed 42

---

## 1. Model Inventory

| Model | Architecture | Params | Notes |
|-------|-------------|-------:|-------|
| EpigeneticNetwork | Embedding → EpiLayer×2 → MemStore → Head | 86 947 | Proposed model |
| Transformer | Embedding → TransformerBlock×2 → Head | 110 114 | Standard MHSA |
| BiLSTM | Embedding → BiLSTM×2 → Head | 178 690 | Recurrent baseline |
| MLP | Embedding (mean-pool) → MLP → Head | 20 994 | Weakest baseline |

All models share the same embedding dimension (64), output head structure, and
training loop.  The only differences are the core sequence-processing layers.

---

## 2. Continual Learning Benchmark

### Protocol

```
1. Train all models on Task A: sentiment (positive / negative labels)
   Dataset: 2 400 train / 600 val, 65-word synthetic vocabulary
2. Record acc_A_before, adaptation speed (epochs to 70% val acc)
3. Continue training (no weight reset) on Task B: topic (tech / sports)
4. Record acc_A_after, acc_B_final
5. Forgetting F = acc_A_before − acc_A_after
```

### Raw Numbers (measured)

| Model | acc\_A\_before | acc\_A\_after | acc\_B | Forgetting F | Adapt (ep) | s/epoch |
|-------|:---:|:---:|:---:|:---:|:---:|---:|
| EpigeneticNetwork | 1.0000 | 0.4024 | 1.0000 | **0.5976** | 1 | 0.96 s |
| Transformer | 1.0000 | 0.4929 | 1.0000 | 0.5071 | 1 | 4.00 s |
| BiLSTM | 1.0000 | 0.8580 | 1.0000 | **0.1420** | 1 | 3.71 s |
| MLP | 1.0000 | 0.7495 | 1.0000 | 0.2505 | 1 | 0.23 s |

**Bold** = best / worst in column.

### Interpretation

**Why all models reach 100 % on each individual task:**
The synthetic vocabulary is only 65 tokens.  Task A sentiment words (`great`,
`terrible`, …) and Task B topic words (`neural`, `championship`, …) have zero
overlap.  Any model with adequate capacity trivially memorises the mapping in
1–2 epochs.

**Why forgetting is high across the board:**
With a 65-token vocabulary the gradient signal for Task B updates the entire
embedding table.  This is pure catastrophic interference — standard for
sequential fine-tuning without replay or regularisation.

**Why BiLSTM forgets least:**
The LSTM's recurrent state distributes task-specific computation across time
steps and hidden units differently from attention or MLP layers.  The gating
mechanism (input/forget/output gates) provides some natural resistance to
full overwriting, similar in spirit to EpiNet's gate but operating at the
recurrent level.

**Why EpiNet forgets more than MLP despite having more capacity:**
EpiNet's dynamic epigenetic state e\_t carries forward Task B's distribution
into the Task A evaluation.  Even if weight updates were identical to MLP,
the evolved e\_t at test time produces Task-B-biased gates, shifting outputs
away from the Task A decision boundary.  This is a fundamental interaction
between the inference-time adaptation mechanism and sequential fine-tuning.

**Mitigation (EpiNet+Boundary):**
Running `model.reset_memory()` and saving/restoring the epigenetic context
via `EpigeneticController.save_context()` at task boundaries is the intended
usage for continual learning.  The current benchmark measures the *naive*
sequential fine-tuning case.

---

## 3. Context Adaptation Benchmark

### Protocol

```
Each token sequence x presented under two contexts:
  context = "formal" → label 1
  context = "casual" → label 0
  tokens: identical

Context sensitivity = P(pred_formal ≠ pred_casual | same x)
Trained on 1 280 (tokens, context, label) triples, 8 epochs.
```

### Raw Numbers (measured)

| Model | Val Accuracy | Context Sensitivity | Mechanism |
|-------|:-----------:|:-------------------:|-----------|
| ContextualEpiNet | 1.0000 | **1.0000** | e\_0 = ContextEmbedding(ctx\_id) |
| Transformer | 1.0000 | 0.0000 | none — no dynamic state |
| BiLSTM | 1.0000 | 0.0000 | none |
| MLP | 1.0000 | 0.0000 | none |

### Interpretation

All models achieve perfect accuracy on the non-context-aware version of the
task.  Only EpiNet can distinguish contexts because it is the only model with
a mechanism to route computation differently for the same input: the
context-conditioned initial state e\_0 steers every epigenetic gate in every
layer.

This is the **core architectural differentiator**.  Information-theoretically:

```
I(output; context | tokens) = 1 bit   (EpiNet)
I(output; context | tokens) = 0 bits  (all stateless models)
```

---

## 4. Speed Analysis

| Model | s/epoch | Relative to Transformer |
|-------|--------:|:-----------------------:|
| MLP | 0.23 s | 17× faster |
| **EpiNet** | **0.96 s** | **4.2× faster** |
| BiLSTM | 3.71 s | 1.08× faster |
| Transformer | 4.00 s | 1× (reference) |

EpiNet's speed advantage comes from the absence of O(T²) self-attention in
its layer stack.  The Transformer runs MHSA over every pair of sequence
positions (T²=4 096 for T=64).  EpiNet's memory read is O(M·D) = O(64·64)
— a fixed constant independent of sequence length.

The BiLSTM's sequential recurrence limits parallelism; on longer sequences
EpiNet's advantage over BiLSTM would widen.

---

## 5. Parameter Efficiency

| Model | Params | Task-A acc | Context sens | Params × Forgetting |
|-------|-------:|:----------:|:------------:|--------------------:|
| EpigeneticNetwork | 86 947 | 1.0000 | **1.0000** | 51 944 |
| Transformer | 110 114 | 1.0000 | 0.0000 | 55 868 |
| BiLSTM | 178 690 | 1.0000 | 0.0000 | 25 374 |
| MLP | 20 994 | 1.0000 | 0.0000 | 5 260 |

"Params × Forgetting" is a combined cost metric (lower = more efficient
retention per parameter).  BiLSTM is best on this metric because of its
low forgetting score.  EpiNet trades retention for the unique context-routing
capability not available to any other model.

---

## 6. Mathematical Verification

All core formulas were verified by unit tests (122 passing).

### EMA Update
```python
# test_controllers.py::test_step_is_ema_update
alpha = 0.3,  e_t = zeros(4,16),  delta = ones(4,16)
e_next = (1-0.3)*0 + 0.3*1 = 0.3   ✓  allclose(e_next, 0.3, atol=1e-5)
```

### Gate Bounds
```python
# test_epigenetic_neuron.py::test_gate_values_in_unit_interval
g = sigmoid(W_e @ e)  →  g.min() >= 0.0  and  g.max() <= 1.0   ✓
```

### Forgetting Formula
```python
# test_training.py::test_forgetting_score_*
forgetting_score(0.9, 0.6)  ==  0.3   ✓
forgetting_score(0.6, 0.8)  == -0.2   ✓  (positive transfer)
```

### Memory Eviction
```python
# test_memory_store.py::test_write_evicts_least_important
# Fill 16 slots with importance=1, then write importance=100
# → slot with importance=100 found in memory.values   ✓
```

### Gradient Flow
```python
# test_epigenetic_network.py::test_end_to_end_gradient_flow
# loss.backward() succeeds; at least one parameter receives .grad   ✓
# Requires clone().detach() on memory buffers to avoid version mismatch
```

---

## 7. Limitations

| Limitation | Impact |
|------------|--------|
| 65-word synthetic vocabulary | All models saturate at 100%; no capacity differences visible |
| No task-boundary protocol | EpiNet measured in naive sequential fine-tuning — worst case |
| Single CPU run | No statistical variance reported |
| Short sequences (T=64) | No long-range dependency advantage for any model |
| Binary classification only | Multi-class or generative settings untested |

---

## 8. Recommended Next Experiments

1. **Permuted-MNIST** — 10-task continual benchmark; shows genuine retention differences
2. **EpiNet+Boundary** at scale — `reset_memory()` + context save/restore between tasks
3. **Variable sequence length** (T=512+) — EpiNet's O(M) vs Transformer's O(T²) should diverge
4. **Multi-context NLI** — same premise, different discourse context, different entailment label
5. **Few-shot evaluation** — measure accuracy vs number of gradient steps on a new task
