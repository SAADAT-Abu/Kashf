#!/usr/bin/env python3
"""
Kashf — TPU v6e (Trillium) pretraining on FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT).
Multi-chip data-parallel training via torch_xla.

    PJRT_DEVICE=TPU python training/train_fineweb_v6e.py [--nprocs 8]

Provision:
    gcloud compute tpus queued-resources create kashf-qr-v6e \
      --node-id=kashf-v6e --project=kashf-494319 \
      --zone=europe-west4-a --accelerator-type=v6e-8 \
      --runtime-version=v2-alpha-tpuv6e --spot

TPU:     v6e-8  (8 chips × 32 GB HBM, ~2–4× MXU throughput vs v4)
Dataset: FineWeb-Edu sample-10BT (~10B tokens of high-quality educational web text)
Context: 4096 tokens
Batch:   8 chips × 8 micro × 2 accum × 4096 = 524,288 tokens / step  (same as v4 script)
Steps:   ~19,073 for 10B tokens
Est:     ~2–4 hours on v6e-8

Differences vs. train_fineweb_tpu.py (v4):
  - MICRO_BATCH 4 → 8:  larger per-chip batch better saturates v6e's wider MXU
  - GRAD_ACCUM  4 → 2:  fewer accumulation passes per step — same global batch, less overhead
  - LOG_EVERY  20 → 10: steps are the same count but land faster; denser progress output
  - CKPT_EVERY 500 → 500: unchanged — checkpoints every ~262M tokens
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from datasets import load_dataset
from transformers import AutoTokenizer
from kashf.model import KashfConfig, KashfModel

# ── Hyperparameters ──────────────────────────────────────────────────────────

SEQ_LEN       = 4096
MICRO_BATCH   = 8            # per-chip; v6e MXU is ~2–4× wider — double the micro batch
GRAD_ACCUM    = 2            # halved vs v4; global batch stays 524,288 tokens/step
# Global batch (v6e-8): 8 micro × 8 chips × 2 accum × 4096 = 524,288 tokens / step

LR            = 3e-4
MIN_LR        = 3e-5
WEIGHT_DECAY  = 0.1
WARMUP_STEPS  = 500          # ~2.6% of ~19,073 total steps
TARGET_TOKENS = 10_000_000_000

LOG_EVERY     = 10           # steps land faster on v6e — log more frequently
CKPT_EVERY    = 500          # checkpoint every ~262M tokens; keep last 3
CKPT_DIR      = os.environ.get("KASHF_CKPT_DIR", "kashf_checkpoints")
GRAD_CLIP     = 1.0
USE_GRAD_CKPT = False        # 32 GB HBM per chip is sufficient

# ── Dataset ──────────────────────────────────────────────────────────────────


class FineWebEduDataset(IterableDataset):
    """
    Streams FineWeb-Edu sample-10BT, packs text into fixed SEQ_LEN+1 chunks.
    No padding — short docs concatenate into the same chunk as the next doc.
    Sharded by XMP ordinal so each TPU chip consumes a distinct subset.
    """

    def __init__(self, tokenizer, seq_len: int, ordinal: int = 0, world_size: int = 1):
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.ordinal    = ordinal
        self.world_size = world_size

    def __iter__(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        ).shard(num_shards=self.world_size, index=self.ordinal)

        buf = []
        for sample in ds:
            buf.extend(self.tokenizer.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf   = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:],  dtype=torch.long),
                )


# ── LR schedule: linear warmup → cosine decay ────────────────────────────────


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
    xm.save(
        {"step": step, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "cfg": cfg},
        tmp,
    )
    os.replace(tmp, path)
    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass
    print(f"  [ckpt] saved → {path}", flush=True)


def load_checkpoint(model, optimizer, path: str) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


# ── Per-chip training function ────────────────────────────────────────────────


def _train_fn(index: int, cli_args):
    """Spawned once per chip by xmp.spawn. `index` is the chip ordinal (0…nprocs-1)."""

    device     = xm.xla_device()
    ordinal    = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    is_master  = xm.is_master_ordinal()

    def mprint(*a, **kw):
        if is_master:
            print(*a, **kw, flush=True)

    tok        = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tok.vocab_size   # 50257

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = KashfConfig(
        vocab_size       = vocab_size,
        dim              = 256,
        embed_dim        = 128,
        n_heads          = 4,
        head_dim         = 64,
        max_seq_len      = SEQ_LEN,
        max_loop_iters   = 6,
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
        mprint(f"Chips      : {world_size}  |  SEQ_LEN: {SEQ_LEN}  |  micro-batch/chip: {MICRO_BATCH}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
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

    # ── Dataset + device loader ───────────────────────────────────────────────
    dataset = FineWebEduDataset(tok, SEQ_LEN, ordinal=ordinal, world_size=world_size)
    loader  = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        num_workers=2,
        prefetch_factor=2,
    )
    device_loader = pl.MpDeviceLoader(loader, device)

    # ── Derived training constants ─────────────────────────────────────────────
    global_batch_tok = MICRO_BATCH * world_size * GRAD_ACCUM * SEQ_LEN
    total_steps      = TARGET_TOKENS // global_batch_tok

    mprint(f"\nDataset    : FineWeb-Edu sample-10BT")
    mprint(f"Target     : {TARGET_TOKENS/1e9:.0f}B tokens  |  {total_steps:,} steps")
    mprint(f"Global batch: {global_batch_tok:,} tokens/step\n")
    mprint(f"{'step':>7}  {'loss':>7}  {'gnorm':>6}  {'lr':>8}  {'tok/s':>10}  {'tokens':>10}")
    mprint("-" * 66)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    data_iter  = iter(device_loader)
    t0         = time.perf_counter()
    step       = start_step

    step_loss  = torch.zeros((), dtype=torch.float32, device=device)
    step_gnorm = torch.zeros((), dtype=torch.float32, device=device)

    while step < total_steps:
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, total_steps)

        optimizer.zero_grad()
        step_loss  = torch.zeros((), dtype=torch.float32, device=device)
        step_gnorm = torch.zeros((), dtype=torch.float32, device=device)

        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(device_loader)
                x, y = next(data_iter)

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
            mprint(
                f"{step:7d}  {step_loss.item():7.4f}  {step_gnorm.item():6.3f}"
                f"  {cur_lr:.2e}  {tok_per_sec:10,.0f}  {tokens_seen/1e9:8.2f}B"
            )
            t0 = time.perf_counter()

        if step % CKPT_EVERY == 0:
            xm.mark_step()   # flush pending ops on every chip before master writes
            if is_master:
                save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)

    if step > start_step and step % CKPT_EVERY != 0:
        xm.mark_step()
        if is_master:
            save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)

    mprint("\nTraining complete.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Kashf v6e training on FineWeb-Edu")
    parser.add_argument(
        "--nprocs", type=int, default=8,
        help="Number of TPU chips to use (8 for v6e-8)",
    )
    cli_args = parser.parse_args()

    print("Pre-caching GPT-2 tokenizer...")
    AutoTokenizer.from_pretrained("gpt2")

    print(f"Spawning training across {cli_args.nprocs} TPU chips...")
    xmp.spawn(_train_fn, args=(cli_args,), nprocs=cli_args.nprocs)


if __name__ == "__main__":
    main()
