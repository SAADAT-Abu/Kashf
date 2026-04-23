# Training Kashf on Wikipedia — Google Colab Guide

This guide trains Kashf on English Wikipedia (`wikimedia/wikipedia`, `20231101.en`, ~20 GB) using a free or paid Google Colab GPU.

---

## Hardware requirements

| Tier | GPU | VRAM | Est. time (4B tokens) |
|---|---|---|---|
| Free | T4 | 16 GB | ~48 hours (use checkpoints) |
| Colab Pro | A100 | 40 GB | ~8 hours |
| Colab Pro+ | A100 80GB | 80 GB | ~6 hours |

The model uses only ~2.8 GB VRAM peak, so even a T4 has headroom to double the batch size.

---

## Step 1 — Open a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

Set the runtime: **Runtime → Change runtime type → T4 GPU** (or A100 if available).

---

## Step 2 — Mount Google Drive (for checkpoints)

Paste this into the first cell and run it:

```python
from google.colab import drive
drive.mount('/content/drive')
```

This lets checkpoints survive session resets. Your Drive will be at `/content/drive/MyDrive/`.

---

## Step 3 — Install dependencies

```bash
%%bash
pip install -q torch transformers datasets
```

---

## Step 4 — Clone Kashf

```bash
%%bash
git clone https://github.com/your-username/kashf /content/kashf
pip install -q -e /content/kashf
```

---

## Step 5 — Verify the install

```python
import torch
from kashf.model import KashfModel, KashfConfig

cfg   = KashfConfig()
model = KashfModel(cfg).cuda()
counts = model.parameter_count()
print(f"Parameters: {counts['unique (deduped)']:,} unique")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

Expected output:
```
Parameters: 20,513,926 unique
VRAM: 0.08 GB
```

---

## Step 6 — Run the training script

Copy this cell — it calls the training script with checkpoints going to your Drive:

```python
import subprocess, sys

cmd = [
    sys.executable, "/content/kashf/training/train_wikipedia.py",
]

# Override checkpoint directory to survive session resets
import os
os.environ["KASHF_CKPT_DIR"] = "/content/drive/MyDrive/kashf_checkpoints"

subprocess.run(cmd, check=True)
```

Or run the script directly with custom settings in a `%%python` cell:

```python
%%python
import os, math, time, torch, torch.nn as nn
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from kashf.model import KashfConfig, KashfModel

# ── Settings ────────────────────────────────────────────────────────────────
SEQ_LEN       = 512
MICRO_BATCH   = 8          # reduce to 4 if OOM on T4
GRAD_ACCUM    = 8
LR            = 3e-4
TARGET_TOKENS = 4_000_000_000
CKPT_DIR      = "/content/drive/MyDrive/kashf_checkpoints"
LOG_EVERY     = 10
CKPT_EVERY    = 500        # checkpoint every 500 steps (~500M tokens seen)
DEVICE        = "cuda"

# ── Tokenizer ───────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained("gpt2")

# ── Dataset ─────────────────────────────────────────────────────────────────
class WikipediaDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
    def __iter__(self):
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                          split="train", streaming=True)
        buf = []
        for sample in ds:
            buf.extend(self.tokenizer.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[:self.seq_len + 1]; buf = buf[self.seq_len + 1:]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:],  dtype=torch.long))

# ── Model ───────────────────────────────────────────────────────────────────
cfg = KashfConfig(vocab_size=tok.vocab_size, dim=256, embed_dim=128,
                  n_heads=4, head_dim=64, max_seq_len=SEQ_LEN,
                  max_loop_iters=6, n_routed_experts=2,
                  expert_dim=256, act_threshold=0.99)
model     = KashfModel(cfg).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=0.1, betas=(0.9, 0.95), fused=True)
amp_ctx   = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# ── Resume ──────────────────────────────────────────────────────────────────
start_step = 0
os.makedirs(CKPT_DIR, exist_ok=True)
ckpts = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pt"))
if ckpts:
    ckpt = torch.load(os.path.join(CKPT_DIR, ckpts[-1]), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    print(f"Resumed at step {start_step}")

# ── LR schedule ─────────────────────────────────────────────────────────────
global_batch_tok = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
total_steps      = TARGET_TOKENS // global_batch_tok

def get_lr(step):
    warmup = 500
    if step < warmup: return LR * step / warmup
    decay = (step - warmup) / max(1, total_steps - warmup)
    return 3e-5 + 0.5 * (LR - 3e-5) * (1 + math.cos(math.pi * decay))

# ── Training loop ────────────────────────────────────────────────────────────
dataset   = WikipediaDataset(tok, SEQ_LEN)
loader    = DataLoader(dataset, batch_size=MICRO_BATCH, num_workers=0,
                       pin_memory=True)
data_iter = iter(loader)
model.train()
t0 = time.perf_counter()

print(f"{'step':>7}  {'loss':>7}  {'gnorm':>6}  {'tok/s':>9}  {'tokens':>10}")
print("-" * 55)

for step in range(start_step, total_steps):
    for g in optimizer.param_groups: g["lr"] = get_lr(step)
    optimizer.zero_grad(); loss_accum = 0.0

    for _ in range(GRAD_ACCUM):
        try: x, y = next(data_iter)
        except StopIteration: data_iter = iter(loader); x, y = next(data_iter)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with amp_ctx:
            logits = checkpoint(model, x, use_reentrant=False)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, cfg.vocab_size), y.view(-1)) / GRAD_ACCUM
        loss.backward(); loss_accum += loss.item()

    gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if (step + 1) % LOG_EVERY == 0:
        dt  = time.perf_counter() - t0
        tps = global_batch_tok * LOG_EVERY / dt
        print(f"{step+1:7d}  {loss_accum:7.4f}  {float(gnorm):6.3f}"
              f"  {tps:9,.0f}  {(step+1)*global_batch_tok/1e9:8.2f}B")
        t0 = time.perf_counter()

    if (step + 1) % CKPT_EVERY == 0:
        path = os.path.join(CKPT_DIR, f"step_{step+1:07d}.pt")
        torch.save({"step": step+1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "cfg": cfg}, path)
        print(f"  [ckpt] → {path}")

print("Done.")
```

---

## Step 7 — Monitor VRAM usage

Run this in a separate cell to check memory at any time:

```python
import torch
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved()  / 1e9
peak      = torch.cuda.max_memory_allocated() / 1e9
print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {peak:.2f} GB")
```

---

## Step 8 — Resume after a session reset

Colab sessions disconnect after ~12 hours on free tier. Checkpoints are saved to Drive every 500 steps. To resume, re-run Steps 3–6 — the script detects the latest checkpoint automatically and continues from where it left off.

---

## Step 9 — Generate text after training

```python
from transformers import AutoTokenizer
import torch
from kashf.model import KashfModel

# Load latest checkpoint
import os, glob
ckpt_dir = "/content/drive/MyDrive/kashf_checkpoints"
latest   = sorted(glob.glob(f"{ckpt_dir}/*.pt"))[-1]
ckpt     = torch.load(latest, map_location="cpu", weights_only=False)

model = KashfModel(ckpt["cfg"]).eval().cuda()
model.load_state_dict(ckpt["model"])
print(f"Loaded checkpoint from step {ckpt['step']:,}")

tok   = AutoTokenizer.from_pretrained("gpt2")
ids   = torch.tensor([tok.encode("The theory of relativity")]).cuda()
out   = model.generate(ids, max_new_tokens=100, n_loops=6, temperature=0.8, top_k=40)
print(tok.decode(out[0].tolist()))
```

---

## Tips

- **OOM on T4?** Reduce `MICRO_BATCH` from 8 to 4. Effective batch stays the same — increase `GRAD_ACCUM` to 16 to compensate.
- **Faster convergence?** Increase `n_loops` at inference (e.g. `n_loops=12`) — the model uses more compute per token with no extra parameters.
- **Better quality?** Run multiple epochs over Wikipedia, or combine with a second dataset (e.g. `Skylion007/openwebtext`).
- **Longer context?** Change `max_seq_len=1024` in `KashfConfig` and retrain — RoPE extrapolates beyond training length.
