#!/usr/bin/env python3
"""
Kashf — TPU v4 pretraining on FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT).
Multi-chip data-parallel training via torch_xla.

    python training/train_fineweb_tpu.py [--nprocs 8]

TPU:     v4-8  (8 chips × 32 GB HBM each)
Dataset: FineWeb-Edu sample-10BT (~10B tokens of high-quality educational web text)
Context: 4096 tokens
Batch:   8 chips × 4 micro × 4 accum × 4096 = 524,288 tokens / step
Steps:   ~19,073 for 10B tokens
Est:     ~6–8 hours on v4-8

Key differences vs. GPU scripts:
  - torch_xla replaces CUDA — no torch.compile, no AMP autocast, no fused optimizer
  - Model is cast to bfloat16 up front (TPU's native dtype)
  - xm.optimizer_step() all-reduces gradients across chips before updating weights
  - pl.MpDeviceLoader prefetches batches to each chip asynchronously
  - Dataset sharded by XMP ordinal — each chip sees a distinct slice
  - .item() / .backward() are lazy in XLA; xm.mark_step() flushes the graph
  - Checkpoints written only from the master chip; xm.rendezvous() keeps chips in sync
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

SEQ_LEN       = 4096         # full context window — TPU HBM supports this comfortably
MICRO_BATCH   = 4            # per-chip; 4 × 4096 = 16,384 tokens / chip / accum-step
GRAD_ACCUM    = 4            # accumulation steps before weight update
# Global batch (v4-8): 4 micro × 8 chips × 4 accum × 4096 = 524,288 tokens / step

LR            = 3e-4
MIN_LR        = 3e-5
WEIGHT_DECAY  = 0.1
WARMUP_STEPS  = 500          # ~2.6% of ~19,073 total steps
TARGET_TOKENS = 10_000_000_000   # full FineWeb-Edu sample-10BT pass

LOG_EVERY     = 20           # print every N optimizer steps (each step = GRAD_ACCUM forward passes)
CKPT_EVERY    = 500          # checkpoint every N steps; keep last 3
CKPT_DIR      = os.environ.get("KASHF_CKPT_DIR", "kashf_checkpoints")
GRAD_CLIP     = 1.0
USE_GRAD_CKPT = False        # 32 GB HBM per chip is enough — disabled for maximum speed

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
    # xm.save() flushes all pending XLA ops before writing — safe on TPU
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

    # Tokenizer files are already cached by main() before spawn — no download race
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

    # Cast to bfloat16 before moving to device — TPU's native dtype; no autocast wrapper needed
    model = KashfModel(cfg).to(torch.bfloat16).to(device)

    if is_master:
        counts = model.parameter_count()
        mprint(f"Parameters : {counts['total']:,} total | {counts['unique (deduped)']:,} unique")
        mprint(f"Chips      : {world_size}  |  SEQ_LEN: {SEQ_LEN}  |  micro-batch/chip: {MICRO_BATCH}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # fused=True is a CUDA-only kernel — not used on XLA
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
        # Optimizer states were loaded to CPU; move them to the XLA device
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
        num_workers=4,       # tokenisation runs in parallel with XLA compute
        prefetch_factor=2,
    )
    # MpDeviceLoader asynchronously prefetches the next batch to the chip
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

    # Lazy XLA tensor for loss — avoids forcing a graph flush inside the micro-batch loop
    step_loss = torch.zeros((), dtype=torch.float32, device=device)
    step_gnorm = torch.zeros((), dtype=torch.float32, device=device)

    while step < total_steps:
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, total_steps)

        optimizer.zero_grad()
        step_loss  = torch.zeros((), dtype=torch.float32, device=device)
        step_gnorm = torch.zeros((), dtype=torch.float32, device=device)

        # Gradient accumulation — forward + backward GRAD_ACCUM times before stepping
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

            # Accumulate without .item() — stays lazy in the XLA graph
            with torch.no_grad():
                step_loss = step_loss + loss.detach()

        # Clip local gradients, then all-reduce across chips and apply the update
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        with torch.no_grad():
            step_gnorm = step_gnorm + gnorm

        xm.optimizer_step(optimizer)   # all-reduce gradients + weight update
        step += 1

        if step % LOG_EVERY == 0:
            # Flush all pending XLA ops before reading tensor values
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
            if is_master:
                save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)
            # All chips rendezvous — prevents any chip racing ahead while master writes
            xm.rendezvous("checkpoint")

    # Final checkpoint if the run didn't land exactly on a CKPT_EVERY boundary
    if step > start_step and step % CKPT_EVERY != 0:
        if is_master:
            save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)
        xm.rendezvous("final_checkpoint")

    mprint("\nTraining complete.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Kashf TPU training on FineWeb-Edu")
    parser.add_argument(
        "--nprocs", type=int, default=8,
        help="Number of TPU chips to use (8 for v4-8, 4 for a single TPU board)",
    )
    cli_args = parser.parse_args()

    # Pre-cache tokenizer before spawning — avoids every chip racing to download it
    print("Pre-caching GPT-2 tokenizer...")
    AutoTokenizer.from_pretrained("gpt2")

    print(f"Spawning training across {cli_args.nprocs} TPU chips...")
    xmp.spawn(_train_fn, args=(cli_args,), nprocs=cli_args.nprocs)


if __name__ == "__main__":
    main()
