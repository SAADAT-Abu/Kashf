# Kashf — Architecture Deep Dive

Kashf (كشف, *unveiling*) is a **Recurrent-Depth Transformer** — a language model that runs the same transformer block multiple times per forward pass instead of stacking many separate layers. Each pass is one *loop*. The model deepens its understanding with each loop and stops when it is confident enough to answer.

This document explains every architectural decision: what it is, what problem it solves, how it works, and what makes it different from conventional approaches.

---

## The Core Idea: Depth Through Repetition

A standard transformer like GPT-2 has, say, 12 separate layers. Each layer has its own independent weights. More layers = more parameters = more cost.

Kashf takes a different approach: **one shared block, run T times**.

```
Standard transformer (12 layers, 12× the weights):
  Layer 1 → Layer 2 → Layer 3 → ... → Layer 12 → Output

Kashf (1 block, run 6 times, same weights):
  Block → Block → Block → Block → Block → Block → Output
  (loop 0) (loop 1) (loop 2) (loop 3) (loop 4) (loop 5)
```

This is called **weight tying across depth**. The same weights perform a different function at each loop because the hidden state `h` evolves — it carries richer information from prior loops. Early loops do rough pattern recognition. Later loops refine and reason. The same parameters do different jobs at different depths because of what they receive as input, not because they are different parameters.

**Why this matters:** A 6-loop Kashf model has roughly the same parameter count as a 1-layer transformer, but the effective computational depth of a 6-layer one. You can also increase `n_loops` at inference time — running 12 loops instead of 6 — to get deeper reasoning from the same weights, with no retraining. This is called **depth extrapolation**.

---

## Full Forward Pass

Here is the complete data flow for a batch of token sequences:

```
Input: token IDs  (B, T)
          │
          ▼
  ┌─────────────────────────────┐
  │     FactoredEmbedding       │  token IDs → dense vectors (B, T, dim=256)
  └─────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────┐
  │       PreludeBlock          │  one pass through the shared block (dense FFN)
  │   MQAttention + SwiGLU      │
  └─────────────────────────────┘
          │  save as e (frozen input encoding)
          │
          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                   KashfRecurrentBlock                            │
  │  ┌─────────────────────────────────────────────────────────┐    │
  │  │  loop t = 0, 1, 2, ... T-1                              │    │
  │  │                                                          │    │
  │  │   h  ←  loop_index_embedding(h, t)   depth signal       │    │
  │  │   x  ←  RMSNorm(h + e)               fuse with input    │    │
  │  │   x  ←  MQAttention(x)               attend over seq    │    │
  │  │   x  ←  MicroMoEFFN(x)               route + compute    │    │
  │  │   x  ←  LoopGate(x, t)               scale by depth     │    │
  │  │   h  ←  LTIInjection(h, e, x)        stable state upd.  │    │
  │  │   p  ←  ACTHalting(h)                should we stop?    │    │
  │  │                                                          │    │
  │  │   if all positions halted: break early                   │    │
  │  └─────────────────────────────────────────────────────────┘    │
  │  return weighted sum of h across loops (ACT weights)            │
  └──────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────┐
  │       CodaBlock             │  same block instance as Prelude (weight-tied)
  │   MQAttention + SwiGLU      │
  └─────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────┐
  │    RMSNorm → LM Head        │  project to vocabulary logits (B, T, vocab_size)
  └─────────────────────────────┘
```

---

## Component-by-Component Breakdown

---

### 1. FactoredEmbedding

**What it does:**
Converts integer token IDs into dense vectors in two steps instead of one.

```
Standard:   Embedding(vocab=50257, dim=256)     → 12.9M parameters
Kashf:      Embedding(vocab=50257, embed_dim=128)
            + Linear(embed_dim=128, dim=256)     → 6.6M parameters
```

**The problem it solves:**
At small model sizes, the embedding table is a disproportionately large fraction of total parameters. With `vocab=50257` and `dim=256`, a standard embedding is 12.9M parameters — more than half the entire model. Most of that capacity is wasted; a 128-dimensional lookup is plenty to distinguish tokens.

**How it works:**
The two-stage lookup first maps each token to a 128-dim vector (cheap lookup), then projects to the model's 256-dim working space (cheap linear). This cuts the embedding cost roughly in half while the model still operates in 256-dim throughout.

**What's new:**
Standard small models either accept the oversized embedding or shrink the vocabulary. Factored embedding is a third option — shrink the lookup without shrinking the vocabulary or the model dimension.

---

### 2. PreludeBlock and CodaBlock (Weight Tying)

**What they do:**
The Prelude runs once before the recurrent loops, reading the raw token embeddings and building an initial contextual representation. The Coda runs once after all loops, doing a final pass over the converged hidden state before projecting to vocabulary logits.

**The trick:**
```python
self.prelude_block = KashfBlock(cfg, use_moe=False)
self.coda_block    = self.prelude_block   # same Python object
```

These are not two separate blocks — they are one block used twice. Every gradient update from Prelude usage and Coda usage both flow into the same weights. The model learns a block that is good at *both* reading raw input *and* doing final cleanup after deep recurrence.

**Why this works:**
The KV cache uses distinct keys (`"prelude"` vs `"coda"`) so the two passes don't interfere. From the model's perspective the two uses are functionally different because the hidden state is in a different condition — raw embeddings at entry, refined recurrent output at exit.

**What's saved:**
The Prelude and Coda together would normally cost ~3M parameters. Weight tying reduces that to zero extra cost — the Coda is free.

---

### 3. MQAttention (Multi-Query Attention)

**What it does:**
Computes scaled dot-product attention where all query heads share a single key head and a single value head.

**Standard Multi-Head Attention (MHA):**
```
Q: (B, T, n_heads, head_dim)     4 heads × 64 dim = 256 dim
K: (B, T, n_heads, head_dim)     4 heads × 64 dim = 256 dim  ← all separate
V: (B, T, n_heads, head_dim)     4 heads × 64 dim = 256 dim  ← all separate
```

**Multi-Query Attention (MQA):**
```
Q: (B, T, n_heads, head_dim)     4 heads × 64 dim = 256 dim  ← per-head
K: (B, T, 1,       head_dim)     1 head  × 64 dim = 64  dim  ← shared
V: (B, T, 1,       head_dim)     1 head  × 64 dim = 64  dim  ← shared
```

Each query head learns to attend differently, but they all attend over the same keys and values. This reduces the KV projection parameters by `n_heads×` and, more importantly, reduces KV cache size by `n_heads×` at inference.

**Kashf adds two things to vanilla MQA:**

**QK-Norm:** RMSNorm is applied to every query and key vector individually, before RoPE:
```python
q = self.q_norm(q)   # normalize each (head_dim,) vector
k = self.k_norm(k)
```
Without this, in a recurrent model the hidden state magnitude grows or shrinks across loops, which causes attention logits to explode or collapse. QK-Norm keeps the dot-product scale stable regardless of loop depth, hidden state magnitude, or sequence length.

**Attention Soft-Cap:** After computing `Q·Kᵀ`, the logits are passed through:
```python
attn = torch.tanh(attn / 50.0) * 50.0
```
This smoothly clamps pre-softmax values to the range (−50, +50). Without it, very large or very small logit values cause the softmax to produce a near-one-hot distribution — the model "stares" at one token and ignores everything else. The soft-cap prevents this entropy collapse with zero extra parameters, unlike adding a temperature hyperparameter or an auxiliary loss.

---

### 4. MicroMoEFFN (Micro Mixture-of-Experts)

**What it does:**
Replaces the standard dense FFN with a small mixture of experts: each token is processed by one *routed* expert (selected dynamically) plus one *shared* expert (always active).

```
Standard dense FFN:
  every token → same FFN weights → output

MicroMoEFFN:
  every token → router picks Expert A or Expert B (routed)
              + Expert C always fires (shared)
              → sum of outputs
```

**Why use MoE in a small model?**
The recurrent block runs the same weights T times. Using a dense FFN means those same FFN weights fire on every token at every loop. With MoE, different tokens activate different experts at different loops. The router learns to send tokens to the expert whose specialisation matches the current stage of computation. This gives the model more expressiveness per parameter in the recurrent body.

**The routing problem:**
A naive argmax router is non-differentiable. The router weights get no gradient, so the router never learns — both experts receive equal traffic forever.

**Kashf's solution — Straight-Through Estimator (STE):**
```python
scores      = F.softmax(logits, dim=-1)          # soft, differentiable
hard        = one_hot(argmax(logits), n_experts)  # hard, sparse, non-differentiable
ste_weights = hard + scores - scores.detach()    # STE trick
```

In the **forward pass**: `hard` is used — one expert fires per token, output is sparse and efficient.

In the **backward pass**: gradients flow through `scores` (the softmax), not `hard` (the argmax). The router actually learns which expert to send which tokens to.

The result: inference efficiency of hard routing, training quality of soft routing.

**Load balancing:**
In standard MoE, the router collapses — one expert becomes dominant and the other starves. Kashf uses a load-balance bias buffer (a non-learned vector updated externally during training) to nudge underused experts into getting more tokens. This is the same aux-loss-free scheme used in DeepSeek-V3. The bias shifts *selection* without distorting the *weighting*, so it never appears in the loss gradient.

---

### 5. LoopGate

**What it does:**
A tiny learned scalar gate applied to the transformer output at each loop:
```python
gate_value = sigmoid(embedding[loop_t])   # one scalar per loop depth
output     = transformer_output × gate_value
```

The gate is an `Embedding(max_loops, 1)` — a table of `max_loops` scalar values, one per depth. At training start, all values are initialised to 1.0, so `sigmoid(1.0) ≈ 0.73` — the gate is mostly open.

**Why this exists:**
Without depth differentiation, the recurrent block receives no signal about which loop it is on. The same weights must handle both first-pass rough reading (loop 0) and late-stage refinement (loop 5) identically. The model is forced to find a compromise that is mediocre at both.

LoopGate lets the model learn per-depth *volume* — it can learn to suppress the transformer output heavily at early loops (slow integration) or amplify it (fast update), differently for each depth.

**What makes it different from LoRA adapters:**
The original architecture this was derived from used per-loop LoRA (low-rank update matrices) for depth differentiation. LoRA costs `2 × dim × rank × max_loops` parameters. LoopGate costs `max_loops` parameters — just one scalar per loop. It is the minimal possible depth signal with zero expressiveness overhead.

---

### 6. LTI Injection (Linear Time-Invariant State Update)

**What it does:**
Updates the recurrent hidden state `h` after each loop:
```
h_new = A · h  +  B · e  +  transformer_output
```

Where:
- `h` is the current hidden state (carries memory from previous loops)
- `e` is the frozen output of the Prelude (the original input encoding, injected every loop)
- `transformer_output` is what the block just computed
- `A` and `B` are learned diagonal vectors

**The stability guarantee:**
In any recurrent system, if the state matrix `A` has any eigenvalue with magnitude ≥ 1, the hidden state can grow without bound across loops — **exploding hidden states**. This causes NaN losses, failed training runs, and sensitivity to learning rate.

Kashf guarantees ρ(A) < 1 (spectral radius strictly less than one) by construction:
```python
A = exp( -exp( log_dt + log_A ) )
```

Breaking this down:
1. `log_A` and `log_dt` are learned parameters (unconstrained, any real value)
2. `exp(log_A)` and `exp(log_dt)` are always positive (exp is always > 0)
3. Their sum `log_dt + log_A` fed into a second `exp` is always positive
4. Negating it: `-exp(...)` is always negative
5. The outer `exp(-positive)` is always in (0, 1)

So regardless of what gradient descent does to `log_A` and `log_dt`, `A` is always a vector of values in (0, 1). The model **cannot** become unstable. This is called Zero-Order Hold (ZOH) discretization of a continuous-time stable system.

**The input injection term `B · e`:**
Without re-injecting the original input at every loop, the hidden state drifts away from the original meaning of the tokens as loops progress. `B · e` is a learned anchor — the model controls how strongly to pull each channel of `h` back toward the original encoding.

---

### 7. ACT Halting (Adaptive Computation Time)

**What it does:**
Lets each token position decide independently how many loops it needs.

At every loop, a tiny linear layer (`dim → 1`) reads the current hidden state and outputs a halting probability `p ∈ (0, 1)`:

```python
p = sigmoid(Linear(h))
```

Positions accumulate `p` across loops. When the cumulative sum reaches the threshold (0.99), that position is considered *halted* — it stops accumulating further updates and contributes its current `h` to the output.

**The ACT weighted sum:**
The final output is not just the last `h`. It is a weighted sum of `h` from all loops:
```
h_out = Σ  weight_t × h_t
         t

weight_t = p_t               (for normal loops, before threshold)
weight_t = 1 - Σ p_<t       (for the final loop — the remainder)
```

The remainder trick ensures weights always sum to exactly 1.0 per position, even if the threshold is never reached (position runs all loops without halting).

**Why this matters:**
- **Easy tokens** (common words, punctuation) reach high halting probability quickly — they exit after 1–2 loops.
- **Hard tokens** (rare words, ambiguous references, reasoning steps) stay active through all loops.
- The compute budget is allocated dynamically, not uniformly.
- At inference, you can increase `n_loops` beyond the training maximum — positions will simply use the remainder as their final weight. This is depth extrapolation.

**What's novel:**
Standard ACT (Graves, 2016) was designed for RNNs processing sequences step-by-step. Here it is applied **across loop depth** rather than across time steps. Each loop is a "time step" in the ACT sense, but the loop runs over the entire sequence in parallel. This gives per-token depth adaptation within a single forward pass.

---

### 8. Loop Index Embedding

**What it does:**
Before each loop pass, a sinusoidal signal encoding the current loop number is added to the first `dim//8` channels of the hidden state:
```python
h = h + sinusoidal_embedding(loop_t, dim=dim//8)
```

**The problem:**
Without this, the shared block weights have no way to tell loop 0 from loop 5. Every loop receives an identically-typed input signal. The block is forced to be completely general — it cannot specialise the first loop for "initial reading" and the last loop for "final refinement."

**Why sinusoidal and not learned:**
A learned embedding table of size `max_loops × dim` would work but costs parameters and fails to generalise beyond `max_loops`. Sinusoidal embeddings are parameter-free and mathematically extend to any loop depth, enabling clean extrapolation.

This is the same principle as RoPE for sequence positions — a coordinate signal injected into the representation so the model's weights know where in the computation they are.

---

### 9. Scaled Residual Initialisation

**What it does:**
Output projection weights (the matrices that write back into the residual stream) are initialised with a smaller standard deviation than input projection weights:

```python
# Output projections (wo in attention, down in FFN experts)
std = 0.02 / sqrt(2)

# All other weights
std = 0.02
```

**Why:**
In a residual network, every block adds its output to the stream: `x = x + block(x)`. At initialisation, if `block(x)` has the same scale as `x`, the residual stream variance doubles with every block. In a recurrent model with T loops, this means variance grows as `2^T` before any training — weights start in a numerically extreme region.

Initialising output projections with `std / sqrt(2)` means each block adds approximately equal variance to what already exists, keeping the residual stream variance roughly constant regardless of depth. This reduces gradient explosion at the start of training, especially important when `n_loops` is large.

This technique, used in GPT-2 and later models, is especially critical in recurrent-depth architectures where the same block's output is accumulated T times.

---

## Parameter Budget

With the default config (`dim=256`, `vocab=50257`, `n_loops=6`):

| Component | Parameters | Notes |
|---|---|---|
| FactoredEmbedding | 6.6M | vocab×128 + 128×256 |
| PreludeBlock / CodaBlock | 3.1M | counted once (weight-tied) |
| KashfRecurrentBlock | 10.4M | MoE attn + FFN + LTI + gate |
| LM Head | 12.9M | dim × vocab, not tied to embedding |
| **Total unique** | **~20.5M** | |
| **Total with ties counted** | **~21M** | Prelude counted twice |

The LM Head is the largest single component. This is unavoidable at GPT-2 vocabulary size with a small model — it can be reduced by tying the LM head to the embedding projection (a common technique not yet applied in Kashf).

---

## What Kashf Does Differently — Summary Table

| Feature | Standard Transformer | Kashf |
|---|---|---|
| Depth | N separate layer stacks | 1 shared block × T loops |
| Embedding | `vocab × dim` | `vocab × embed_dim + embed_dim × dim` |
| Attention | MHA or GQA | MQA + QK-Norm + soft-cap |
| FFN | Dense SwiGLU | Micro-MoE with STE routing |
| Depth signal | Implicit (position in stack) | Loop index sinusoidal + LoopGate scalar |
| Recurrent state | None | LTI update, ρ(A) < 1 guaranteed |
| Compute allocation | Uniform (every layer fires) | Adaptive (ACT halts easy tokens early) |
| Inference scaling | Fixed | `n_loops` adjustable post-training |
| Residual init | `std=0.02` everywhere | `std=0.02/√2` for output projections |
| Prelude/Coda cost | 2× block params | 1× block params (weight-tied) |

---

## Inference-Time Scaling

One of the most practically useful properties of Kashf is that compute at inference can be increased without retraining:

```python
# Fast, cheap: 3 loops
logits = model(input_ids, n_loops=3)

# Slow, deeper: 12 loops (trained on 6, extrapolates cleanly)
logits = model(input_ids, n_loops=12)
```

The quality improves with more loops because:
1. LTI injection continues refining `h` beyond the trained maximum
2. ACT assigns the remaining probability mass to the final loop via the remainder trick
3. Loop index sinusoidal embeddings extend analytically to any `t` (no learned table boundary)

This means you can ship one set of weights and let the user choose the compute/quality tradeoff at runtime — a property most transformers do not have.
