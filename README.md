# Kashf

**Kashf** (كشف - *unveiling*) is a compact Recurrent-Depth Transformer that deepens its understanding loop by loop, halting when it is certain enough to answer.

Designed to train on a single 8 GB GPU and run on consumer hardware.

---

## Architecture

```
Token IDs
   ↓
FactoredEmbedding          vocab × embed_dim → Linear → dim
   ↓
PreludeBlock               MQAttention + dense SwiGLU  (weight-tied with Coda)
   ↓  ←── frozen input e injected every loop ──────────────────────┐
KashfRecurrentBlock        looped T times                           │
   ├─ loop_index_embedding  sinusoidal depth signal                 │
   ├─ MQAttention           QK-Norm + soft-cap, single KV head      │
   ├─ MicroMoEFFN           2 routed experts + 1 shared (STE)       │
   ├─ LoopGate              per-loop scalar gate                    │
   ├─ LTIInjection          h = A·h + B·e + out  (ρ(A) < 1)         │
   └─ ACTHalting            halt when cumulative p ≥ threshold ─────┘
   ↓
CodaBlock                  same block instance as Prelude
   ↓
RMSNorm → LM Head
```

| Property | Value |
|---|---|
| Parameters | ~21M total / ~20M unique (GPT-2 vocab) |
| Peak VRAM | 2.8 GB (batch 8 × 8, seq 512, bfloat16) |
| Throughput | ~45K tok/s on RTX 3080 |
| Stable recurrence | ρ(A) < 1 guaranteed by construction |
| Adaptive compute | Easy tokens exit early; hard tokens run deeper |

---

## Quick Start

```python
import torch
from kashf.model import KashfModel, KashfConfig

cfg   = KashfConfig()
model = KashfModel(cfg)
model.eval()

ids    = torch.randint(0, cfg.vocab_size, (1, 32))
logits = model(ids, n_loops=4)                    # (1, 32, 50257)
out    = model.generate(ids, max_new_tokens=20)   # (1, 52)
```

---

## Installation

```bash
git clone https://github.com/SAADAT-Abu/Kashf.git
cd kashf
pip install -e .
```

---

## Training

Single GPU, ~24 hours on English Wikipedia (~4B tokens, 20 GB):

```bash
python training/train_wikipedia.py
```

See [`docs/colab_training_guide.md`](docs/colab_training_guide.md) for the Google Colab step-by-step guide.

---

## Tests

```bash
pytest tests/test_kashf.py -v
```

---

## License

GPL v3
