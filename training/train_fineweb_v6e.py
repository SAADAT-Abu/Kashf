#!/usr/bin/env python3
"""
Kashf — TPU v6e (Trillium) pretraining on FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT).
PJRT training on v6e via xmp.spawn (gRPC child-process init).

    PJRT_DEVICE=TPU python training/train_fineweb_v6e.py

Provision (v2-alpha-tpuv6e runtime required):
    gcloud compute tpus tpu-vm create node-2 \
      --zone=europe-west4-a --project=kashf-494319 \
      --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e

Architecture:
  The v2-alpha-tpuv6e runtime serves TPU chips through a gRPC Docker container.
  Direct PJRT init from the main process (xm.xla_device()) tries VFIO and fails.
  xmp.spawn initialises PJRT inside spawned child processes which use the gRPC
  path instead.  With TPU_PROCESS_BOUNDS=1,1,1 (default for this runtime) xmp.spawn
  produces 1 child process whose xla:0 device represents all 8 chips via gRPC/SPMD.
  Do NOT override TPU_PROCESS_BOUNDS — that breaks VFIO exclusion.

Data pipeline (PrefetchedDataset):
  Background daemon thread downloads FineWeb-Edu and buffers tokenized int32 pairs
  in a bounded queue.  Training reads via next_batch() — O(1) queue.get, no network
  latency on the XLA critical path.  Auto-retries on HF errors.  Set HF_TOKEN env
  var for authenticated downloads (higher rate limits).

Batch:  8 micro × 32 accum × 2048 = 524 288 tokens/step
Run-2 changes vs run-1: TARGET_TOKENS 10B→50B, max_loop_iters 2→4,
WARMUP_STEPS 500→1000, load_checkpoint handles LoopGate size mismatch.
"""

import os
import math
import time
import queue
import threading
import argparse
import torch
import torch.nn as nn

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from datasets import load_dataset
from transformers import AutoTokenizer
from kashf.model import KashfConfig, KashfModel

# ── Hyperparameters ──────────────────────────────────────────────────────────

SEQ_LEN      = 2048
MICRO_BATCH  = 8
GRAD_ACCUM   = 32   # 8 × 32 × 2048 = 524 288 tokens/step (same global batch)

LR           = 3e-4
MIN_LR       = 3e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 1000
TARGET_TOKENS = 50_000_000_000

LOG_EVERY    = 10
CKPT_EVERY   = 500
CKPT_DIR     = os.environ.get("KASHF_CKPT_DIR", "kashf_checkpoints")
GRAD_CLIP    = 1.0
USE_GRAD_CKPT = False

# ── Data prefetch ─────────────────────────────────────────────────────────────
# Memory: MAXBUF_SEQS × SEQ_LEN × 4 bytes (int32)
# Default: 4000 × 2048 × 4 ≈ 32 MB — well within v6e VM RAM.
# Raise PREFILL_SEQS for a larger head-start before training begins.
PREFILL_SEQS = 2000   # sequences buffered before training starts
MAXBUF_SEQS  = 4000   # hard cap on queue depth (backpressure)


# ── Prefetched dataset ────────────────────────────────────────────────────────

class PrefetchedDataset:
    """
    Streams HuggingFace FineWeb-Edu in a background daemon thread and buffers
    tokenized (x, y) int32 pairs in a bounded in-memory queue.

    next_batch() is an O(1) queue.get() — no network latency on the XLA path.
    The download thread auto-retries on any HF error without crashing training.

    Supports HF_TOKEN env var for authenticated requests (higher rate limits).
    Each instance is shard-aware: ordinal / world_size determines which slice of
    the dataset this chip downloads.
    """

    def __init__(self, tokenizer, seq_len: int, ordinal: int, world_size: int):
        self._q       = queue.Queue(maxsize=MAXBUF_SEQS)
        self._ready   = threading.Event()
        self._seq_len = seq_len
        t = threading.Thread(
            target=self._worker,
            args=(tokenizer, seq_len, ordinal, world_size),
            daemon=True,
        )
        t.start()

    def _worker(self, tokenizer, seq_len: int, ordinal: int, world_size: int):
        hf_token = os.environ.get("HF_TOKEN")
        while True:
            try:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True,
                    token=hf_token,
                ).shard(num_shards=world_size, index=ordinal)

                buf: list[int] = []
                for sample in ds:
                    buf.extend(tokenizer.encode(sample["text"]))
                    while len(buf) >= seq_len + 1:
                        chunk = buf[: seq_len + 1]
                        buf   = buf[seq_len + 1 :]
                        x = torch.tensor(chunk[:-1], dtype=torch.int32)
                        y = torch.tensor(chunk[1:],  dtype=torch.int32)
                        self._q.put((x, y))          # blocks when buffer is full
                        if not self._ready.is_set() and self._q.qsize() >= PREFILL_SEQS:
                            self._ready.set()

                print("  [data] dataset pass complete, restarting", flush=True)

            except Exception as exc:
                print(f"  [data] error: {exc!r} — retrying in 10 s", flush=True)
                time.sleep(10)

    def wait_ready(self, mprint) -> None:
        if not self._ready.is_set():
            mb = PREFILL_SEQS * (self._seq_len + 1) * 4 / 1e6
            mprint(
                f"  [data] pre-filling buffer — target {PREFILL_SEQS} seqs "
                f"(~{mb:.0f} MB) …"
            )
            self._ready.wait()
            mprint(f"  [data] buffer ready — {self._q.qsize()} seqs queued.")

    def next_batch(self, micro_batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(micro_batch):
            x, y = self._q.get()
            xs.append(x)
            ys.append(y)
        return (
            torch.stack(xs).long().to(device),
            torch.stack(ys).long().to(device),
        )

    def buf_depth(self) -> int:
        return self._q.qsize()


# ── LR schedule ──────────────────────────────────────────────────────────────


def get_lr(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / max(1, WARMUP_STEPS)
    if step >= total:
        return MIN_LR
    decay = (step - WARMUP_STEPS) / (total - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * decay))


# ── Checkpointing ─────────────────────────────────────────────────────────────


def _list_ckpts(d: str) -> list[str]:
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(model, optimizer, step: int, cfg, ckpt_dir: str, keep_last: int = 3):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp  = path + ".tmp"
    xm.save({"step": step, "model": model.state_dict(),
              "optimizer": optimizer.state_dict(), "cfg": cfg}, tmp)
    os.replace(tmp, path)
    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass
    print(f"  [ckpt] saved → {path}", flush=True)


def load_checkpoint(model, optimizer, path: str) -> int:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    # LoopGate is nn.Embedding(max_loop_iters, 1); pad if checkpoint has fewer iters.
    # New rows initialised to 1.0, matching nn.init.ones_ in LoopGate.__init__.
    key = "recurrent.loop_gate.gate.weight"
    if key in state:
        saved        = state[key]
        target_size  = model.recurrent.loop_gate.gate.weight.shape[0]
        if saved.shape[0] < target_size:
            pad       = torch.ones(target_size - saved.shape[0], 1, dtype=saved.dtype)
            state[key] = torch.cat([saved, pad], dim=0)
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


# ── Per-process training function ────────────────────────────────────────────


def _train_fn(index: int, cli_args):
    """
    Spawned by xmp.spawn.  PJRT init happens inside this child process via gRPC
    — do not call xm.xla_device() in main() before spawn (causes VFIO conflict).
    """
    device     = xm.xla_device()
    ordinal    = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    is_master  = xm.is_master_ordinal()

    def mprint(*a, **kw):
        if is_master:
            print(*a, **kw, flush=True)

    tok        = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tok.vocab_size

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = KashfConfig(
        vocab_size       = vocab_size,
        dim              = 256,
        embed_dim        = 128,
        n_heads          = 4,
        head_dim         = 64,
        max_seq_len      = SEQ_LEN,
        max_loop_iters   = 4,
        n_routed_experts = 2,
        n_shared_experts = 1,
        expert_dim       = 256,
        act_threshold    = 0.99,
        rope_theta       = 500_000.0,
    )
    model = KashfModel(cfg).to(torch.bfloat16).to(device)

    if is_master:
        counts = model.parameter_count()
        mprint(f"Parameters : {counts['total']:,} total | {counts['unique (deduped)']:,} unique")
        mprint(f"Device     : {device}  |  world_size: {world_size}")
        mprint(f"SEQ_LEN: {SEQ_LEN}  |  micro-batch: {MICRO_BATCH}  |  grad-accum: {GRAD_ACCUM}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    existing   = _list_ckpts(CKPT_DIR)
    if existing:
        mprint(f"Resuming from {existing[-1]}")
        start_step = load_checkpoint(model, optimizer, existing[-1])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        mprint(f"Resumed at step {start_step}")

    # ── Prefetched data pipeline ──────────────────────────────────────────────
    dataset = PrefetchedDataset(tok, SEQ_LEN, ordinal, world_size)
    dataset.wait_ready(mprint)

    # ── Training constants ─────────────────────────────────────────────────────
    global_batch_tok = MICRO_BATCH * world_size * GRAD_ACCUM * SEQ_LEN
    total_steps      = TARGET_TOKENS // global_batch_tok

    mprint(f"\nDataset    : FineWeb-Edu sample-10BT")
    mprint(f"Target     : {TARGET_TOKENS/1e9:.0f}B tokens  |  {total_steps:,} steps")
    mprint(f"Global batch: {global_batch_tok:,} tokens/step\n")
    mprint(
        f"{'step':>7}  {'loss':>7}  {'gnorm':>6}  {'lr':>8}"
        f"  {'tok/s':>10}  {'tokens':>9}  {'buf':>5}"
    )
    mprint("-" * 72)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0   = time.perf_counter()
    step = start_step

    step_loss  = torch.zeros((), dtype=torch.float32, device=device)
    step_gnorm = torch.zeros((), dtype=torch.float32, device=device)

    while step < total_steps:
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, total_steps)

        optimizer.zero_grad()
        step_loss  = torch.zeros((), dtype=torch.float32, device=device)
        step_gnorm = torch.zeros((), dtype=torch.float32, device=device)

        for _ in range(GRAD_ACCUM):
            x, y = dataset.next_batch(MICRO_BATCH, device)

            if USE_GRAD_CKPT:
                from torch.utils.checkpoint import checkpoint as grad_ckpt
                logits = grad_ckpt(model, x, use_reentrant=True)
            else:
                logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1)
            ) / GRAD_ACCUM
            loss.backward()
            with torch.no_grad():
                step_loss = step_loss + loss.detach()
            xm.mark_step()  # flush each micro-batch separately; keeps graph size small

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        with torch.no_grad():
            step_gnorm = step_gnorm + gnorm

        xm.optimizer_step(optimizer)
        step += 1

        if step % LOG_EVERY == 0:
            xm.mark_step()
            dt          = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * LOG_EVERY / dt
            tokens_seen = step * global_batch_tok
            cur_lr      = get_lr(step, total_steps)
            buf         = dataset.buf_depth()
            mprint(
                f"{step:7d}  {step_loss.item():7.4f}  {step_gnorm.item():6.3f}"
                f"  {cur_lr:.2e}  {tok_per_sec:10,.0f}  {tokens_seen/1e9:7.2f}B"
                f"  {buf:5d}"
            )
            t0 = time.perf_counter()

        if step % CKPT_EVERY == 0:
            xm.mark_step()
            if is_master:
                save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)

    if step > start_step and step % CKPT_EVERY != 0:
        xm.mark_step()
        if is_master:
            save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)

    mprint("\nTraining complete.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    argparse.ArgumentParser(description="Kashf v6e training on FineWeb-Edu").parse_args()
    # Pre-cache tokenizer BEFORE spawn — avoids every child process racing to download.
    # Do NOT call xm.xla_device() here — that triggers VFIO mode and crashes.
    print("Pre-caching GPT-2 tokenizer...")
    AutoTokenizer.from_pretrained("gpt2")
    print("Spawning via xmp.spawn (nprocs=None, PJRT gRPC child-process init)…")
    xmp.spawn(_train_fn, args=(None,), nprocs=None)


if __name__ == "__main__":
    main()
