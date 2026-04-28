# Kashf on TPU v4 — Setup & Training Guide

This guide provisions a **TPU v4-8** using the **Queued Resource API**, installs Kashf, and trains on FineWeb-Edu (10B tokens). Everything runs inside a `tmux` session so training survives SSH disconnects.

**Project:** `kashf-494319`  
**Zone:** `us-central2-b`  
**Quota:** 32 spot + 32 on-demand v4 chips (30-day free tier)

---

## Prerequisites — local machine

### 1. Install and authenticate gcloud

```bash
# Install the Google Cloud SDK if not already installed
# https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project kashf-494319
gcloud config set compute/zone us-central2-b
```

### 2. Enable the Cloud TPU API

```bash
gcloud services enable tpu.googleapis.com
```

### 3. Check available runtime versions for v4

```bash
gcloud compute tpus tpu-vm list-versions \
  --zone=us-central2-b \
  --project=kashf-494319
```

Look for a version that contains `v4` or `pytorch`. Note the exact string — you'll use it in Step 2.

---

## Step 1 — Create a GCS bucket for checkpoints

TPU VMs can be preempted (spot) or disconnected. Store checkpoints in Google Cloud Storage so they survive.

```bash
# Create the bucket in the same region as your TPU
gsutil mb -p kashf-494319 -l US-CENTRAL2 gs://kashf-checkpoints

# Verify
gsutil ls gs://kashf-checkpoints
```

---

## Step 2 — Request a Queued Resource (spot v4-8)

The Queued Resource API queues your request and provisions the TPU as soon as capacity is available. Spot instances are free within your quota window.

```bash
gcloud compute tpus queued-resources create kashf-qr-1 \
  --node-id=kashf-v4 \
  --project=kashf-494319 \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --runtime-version=tpu-vm-v4-base \
  --spot
```

> **`--node-id`** is the name of the TPU VM that will be created once provisioned.  
> **`kashf-qr-1`** is the name of the queued resource request itself (used to check status and delete it later).

If you prefer on-demand (guaranteed, no preemption):

```bash
# Replace --spot with nothing — on-demand is the default
gcloud compute tpus queued-resources create kashf-qr-1 \
  --node-id=kashf-v4 \
  --project=kashf-494319 \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --runtime-version=tpu-vm-v4-base
```

---

## Step 3 — Wait for provisioning

Check the status every few minutes:

```bash
gcloud compute tpus queued-resources list \
  --zone=us-central2-b \
  --project=kashf-494319
```

Status transitions: `WAITING_FOR_RESOURCES` → `PROVISIONING` → `ACTIVE`

Once **ACTIVE**, the TPU VM named `kashf-v4` is ready to SSH into.

To watch status automatically (re-checks every 30 seconds):

```bash
watch -n 30 "gcloud compute tpus queued-resources list \
  --zone=us-central2-b --project=kashf-494319"
```

---

## Step 4 — SSH into the TPU VM

```bash
gcloud compute tpus tpu-vm ssh kashf-v4 \
  --zone=us-central2-b \
  --project=kashf-494319
```

All remaining steps run **inside the TPU VM** unless noted otherwise.

---

## Step 5 — Start a tmux session

`tmux` keeps your training alive after SSH disconnects.

```bash
sudo apt-get update -q && sudo apt-get install -y tmux
tmux new-session -s kashf
```

You are now inside the `kashf` tmux session. Training will continue here even if your SSH connection drops.

**Detach** (training keeps running): `Ctrl+B`, then `D`  
**Re-attach** after reconnecting: `tmux attach -t kashf`

---

## Step 6 — Install Miniconda and set up a Python 3.10 environment

The TPU VM ships with Python 3.8, which has no stable `torch_xla` wheel. The `python3.10-venv` and `python3.10-dev` apt packages are also unavailable on this image. **Miniconda** is the most reliable way to get a clean Python 3.10 environment.

```bash
# Download and install Miniconda (non-interactive)
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3

# Create a Python 3.10 environment named "kashf"
~/miniconda3/bin/conda create -n kashf python=3.10 -y

# Activate it
source ~/miniconda3/bin/activate kashf

# Persist activation across SSH sessions and tmux windows
echo 'source ~/miniconda3/bin/activate kashf' >> ~/.bashrc
```

Now install PyTorch (CPU build — no CUDA on TPU VMs) and torch_xla:

```bash
pip install --upgrade pip
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-linux_x86_64.whl

# Dataset + tokenizer libraries
pip install datasets transformers
```

Verify torch_xla sees the TPU:

```bash
python - <<'EOF'
import torch
import torch_xla.core.xla_model as xm
print("torch:", torch.__version__)
print("torch_xla:", __import__("torch_xla").__version__)
print("XLA device:", xm.xla_device())
EOF
```

Expected output:
```
torch: 2.4.0+cpu
torch_xla: 2.4.0
XLA device: xla:0
```

---

## Step 7 — Clone and install Kashf

```bash
git clone https://github.com/SAADAT-Abu/kashf ~/kashf
pip install ~/kashf
```

Verify the install:

```bash
python - <<'EOF'
import torch
import torch_xla.core.xla_model as xm
from kashf.model import KashfConfig, KashfModel

device = xm.xla_device()
cfg    = KashfConfig(max_seq_len=4096)
model  = KashfModel(cfg).to(torch.bfloat16).to(device)
counts = model.parameter_count()
print(f"Parameters: {counts['unique (deduped)']:,} unique")
print(f"Device: {device}")
EOF
```

---

## Step 8 — Configure checkpoint directory (GCS)

Point the training script at your GCS bucket. Checkpoints written there survive preemption.

```bash
export KASHF_CKPT_DIR=gs://kashf-checkpoints/fineweb-run1

# Make the export permanent in this session
echo 'export KASHF_CKPT_DIR=gs://kashf-checkpoints/fineweb-run1' >> ~/.bashrc
```

> **Note:** `xm.save()` in the training script writes to local paths. For GCS paths, install `gcsfs` and `torch_xla`'s GCS integration, **or** use the local-to-GCS sync approach in Step 10.
>
> Simplest approach for spot instances: use a local checkpoint dir and sync to GCS periodically (Step 10).

```bash
# Use local checkpoints (simpler, fast)
export KASHF_CKPT_DIR=/home/$USER/kashf_checkpoints
mkdir -p $KASHF_CKPT_DIR
```

---

## Step 9 — Start training

Inside the `kashf` tmux session:

```bash
cd ~/kashf
python training/train_fineweb_tpu.py --nprocs 8
```

The first ~60–90 seconds are XLA graph compilation — the script will appear frozen. After compilation the log table appears:

```
Pre-caching GPT-2 tokenizer...
Spawning training across 8 TPU chips...
Parameters : 20,513,926 total | 20,513,926 unique
Chips      : 8  |  SEQ_LEN: 4096  |  micro-batch/chip: 4

Dataset    : FineWeb-Edu sample-10BT
Target     : 10B tokens  |  19,073 steps
Global batch: 524,288 tokens/step

   step     loss   gnorm        lr       tok/s      tokens
──────────────────────────────────────────────────────────────────
      20  10.5231   1.234  6.00e-06    480,000    0.01B
      40   9.8812   0.981  1.20e-05    520,000    0.02B
     ...
```

---

## Step 10 — Sync checkpoints to GCS (spot safety)

Spot TPUs can be preempted without notice. Run this in a **second tmux window** (`Ctrl+B C` to create a new window, `Ctrl+B 0/1` to switch):

```bash
# Sync every 10 minutes — adjust interval to taste
while true; do
  gsutil -m rsync -r /home/$USER/kashf_checkpoints gs://kashf-checkpoints/fineweb-run1
  sleep 600
done
```

Or add a one-liner after each checkpoint save by setting:

```bash
export KASHF_CKPT_DIR=/home/$USER/kashf_checkpoints
```

and running this in a loop alongside training.

---

## Step 11 — Monitor training

**Re-attach after disconnect:**

```bash
gcloud compute tpus tpu-vm ssh kashf-v4 \
  --zone=us-central2-b \
  --project=kashf-494319

tmux attach -t kashf
```

**Check TPU utilisation** (in a separate tmux window):

```bash
# Profile every 5 seconds
watch -n 5 "curl -s http://localhost:8475/requestz?qname=tpu_utilization 2>/dev/null || echo 'profiler not available'"
```

**Expected throughput on v4-8:**

| Metric | Value |
|---|---|
| Tokens / second | ~400K–600K |
| Steps / hour | ~2,750–4,100 |
| Total time (10B) | ~5–8 hours |

---

## Step 12 — Resume after preemption

If the spot instance is preempted, re-request it and resume:

```bash
# On your local machine — re-request the same resource
gcloud compute tpus queued-resources create kashf-qr-2 \
  --node-id=kashf-v4 \
  --project=kashf-494319 \
  --zone=us-central2-b \
  --accelerator-type=v4-8 \
  --runtime-version=tpu-vm-v4-base \
  --spot

# After it becomes ACTIVE, SSH in and restore from GCS
gsutil -m rsync -r gs://kashf-checkpoints/fineweb-run1 ~/kashf_checkpoints

# Training auto-detects the latest checkpoint and resumes
python training/train_fineweb_tpu.py --nprocs 8
```

---

## Step 13 — Generate text after training

```python
import torch, glob
import torch_xla.core.xla_model as xm
from transformers import AutoTokenizer
from kashf.model import KashfModel

device  = xm.xla_device()
latest  = sorted(glob.glob("/home/$USER/kashf_checkpoints/*.pt"))[-1]
ckpt    = torch.load(latest, map_location="cpu", weights_only=False)

model   = KashfModel(ckpt["cfg"]).to(torch.bfloat16).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tok  = AutoTokenizer.from_pretrained("gpt2")
ids  = torch.tensor([tok.encode("The key to understanding")]).to(device)
out  = model.generate(ids, max_new_tokens=200, n_loops=6, temperature=0.8, top_k=40)
print(tok.decode(out[0].tolist()))
```

---

## Step 14 — Delete the queued resource when done

You are billed for running TPU time (even at $0 within your free quota, the chips count against the 30-day allocation). Delete the resource when training is complete.

```bash
# From your local machine
gcloud compute tpus tpu-vm delete kashf-v4 \
  --zone=us-central2-b \
  --project=kashf-494319

gcloud compute tpus queued-resources delete kashf-qr-1 \
  --zone=us-central2-b \
  --project=kashf-494319
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `WAITING_FOR_RESOURCES` for > 30 min | Try `--zone=us-east1-d` with v6e chips (64 available) |
| `ModuleNotFoundError: torch_xla` after SSH | Run `source ~/miniconda3/bin/activate kashf` — conda env not active |
| `RuntimeError: torch_xla not found` | Re-run the pip install in Step 6 inside the conda env |
| Script freezes at startup | Normal — XLA compilation takes 60–90 s |
| `OOM on chip` | Reduce `MICRO_BATCH` from 4 to 2 in `train_fineweb_tpu.py` |
| Spot preempted mid-run | Follow Step 12; checkpoints every 500 steps (~262M tokens) |
| Loss not decreasing after 1B tokens | Increase `max_loop_iters` from 6 to 8 at inference; model is fine |

---

## Using v6e chips instead (64 available)

If v4 capacity is unavailable, switch to `v6e-8` in `europe-west4-a`:

```bash
gcloud compute tpus queued-resources create kashf-qr-v6e \
  --node-id=kashf-v6e \
  --project=kashf-494319 \
  --zone=europe-west4-a \
  --accelerator-type=v6e-8 \
  --runtime-version=v2-alpha-tpuv6e \
  --spot
```

Then SSH with `--zone=europe-west4-a`. The training script needs no changes — torch_xla abstracts the chip generation.
