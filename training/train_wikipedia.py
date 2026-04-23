#!/usr/bin/env python3
"""
Kashf — pretraining on Wikipedia English (wikimedia/wikipedia, 20231101.en).

    python training/train_wikipedia.py

Dataset:  wikimedia/wikipedia  20231101.en  (~20 GB, ~4B tokens)
Hardware: single GPU, tested on 8 GB VRAM
Time:     ~24 hours on a mid-range GPU (RTX 3080 / A100 ~6 h)
"""

import os
import math
import time
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from datasets import load_dataset
from transformers import AutoTokenizer

from kashf.model import KashfConfig, KashfModel

# ---------------------------------------------------------------------------
# Hyperparameters — edit here to customise the run
# ---------------------------------------------------------------------------

SEQ_LEN        = 512
MICRO_BATCH    = 8
GRAD_ACCUM     = 8          # effective batch = 8 × 8 × 512 = 32,768 tokens
LR             = 3e-4
MIN_LR         = 3e-5
WEIGHT_DECAY   = 0.1
WARMUP_STEPS   = 500
TARGET_TOKENS  = 4_000_000_000   # full Wikipedia pass (~4B tokens)
LOG_EVERY      = 10
CKPT_EVERY     = 1000
CKPT_DIR       = "kashf_checkpoints"
GRAD_CLIP      = 1.0
USE_GRAD_CKPT  = True        # saves ~40% VRAM, ~20% slower

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WikipediaDataset(IterableDataset):
    """
    Streaming Wikipedia loader yielding (input_ids, target_ids) pairs of length SEQ_LEN.

    Documents are concatenated into a rolling buffer so short articles pack together
    and long ones split cleanly — no padding, no wasted tokens.
    """

    def __init__(self, tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len

    def __iter__(self):
        worker     = get_worker_info()
        num_w      = worker.num_workers if worker else 1
        worker_id  = worker.id if worker else 0

        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        ).shard(num_shards=num_w, index=worker_id)

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


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, total: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / max(1, WARMUP_STEPS)
    if step >= total:
        return MIN_LR
    decay = (step - WARMUP_STEPS) / (total - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(model, optimizer, step, cfg, ckpt_dir, keep_last=3):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp  = path + ".tmp"
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "cfg": cfg}, tmp)
    os.replace(tmp, path)
    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(model, optimizer, path) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA device found — training on CPU will be very slow.")

    # Tokenizer
    print("Loading tokenizer (GPT-2)...")
    tok        = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tok.vocab_size   # 50257

    # Model config
    cfg = KashfConfig(
        vocab_size     = vocab_size,
        dim            = 256,
        embed_dim      = 128,
        n_heads        = 4,
        head_dim       = 64,
        max_seq_len    = SEQ_LEN,
        max_loop_iters = 6,
        n_routed_experts = 2,
        n_shared_experts = 1,
        expert_dim     = 256,
        act_threshold  = 0.99,
        rope_theta     = 500000.0,
    )

    model  = KashfModel(cfg).to(device)
    counts = model.parameter_count()
    print(f"Parameters: {counts['total']:,} total | {counts['unique (deduped)']:,} unique")

    if device == "cuda":
        print(f"VRAM after model load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # AMP
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda" else nullcontext()
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR,
        weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )

    # Resume
    start_step = 0
    existing   = _list_ckpts(CKPT_DIR)
    if existing:
        print(f"Resuming from {existing[-1]}")
        start_step = load_checkpoint(model, optimizer, existing[-1])
        print(f"Resumed at step {start_step}")

    # Dataset
    dataset = WikipediaDataset(tok, SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=MICRO_BATCH, num_workers=0,
                         pin_memory=(device == "cuda"))

    global_batch_tok = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
    total_steps      = TARGET_TOKENS // global_batch_tok
    print(f"\nseq_len={SEQ_LEN} | micro_batch={MICRO_BATCH} | grad_accum={GRAD_ACCUM}")
    print(f"global_batch={global_batch_tok:,} tokens | total_steps={total_steps:,}")
    print(f"target={TARGET_TOKENS/1e9:.0f}B tokens\n")
    print(f"{'step':>7}  {'loss':>7}  {'gnorm':>6}  {'lr':>8}  {'tok/s':>9}  {'tokens':>10}")
    print("-" * 62)

    model.train()
    data_iter = iter(loader)
    t0   = time.perf_counter()
    step = start_step

    while step < total_steps:
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step, total_steps)

        optimizer.zero_grad()
        loss_accum = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
                if USE_GRAD_CKPT:
                    logits = checkpoint(model, x, use_reentrant=False)
                else:
                    logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / GRAD_ACCUM

            loss.backward()
            loss_accum += loss.item()

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            dt           = time.perf_counter() - t0
            tok_per_sec  = global_batch_tok * LOG_EVERY / dt
            tokens_seen  = step * global_batch_tok
            cur_lr       = get_lr(step, total_steps)
            print(
                f"{step:7d}  {loss_accum:7.4f}  {float(gnorm):6.3f}"
                f"  {cur_lr:.2e}  {tok_per_sec:9,.0f}  {tokens_seen/1e9:8.2f}B"
            )
            t0 = time.perf_counter()

        if step % CKPT_EVERY == 0:
            save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)

    if step > start_step and step % CKPT_EVERY != 0:
        save_checkpoint(model, optimizer, step, cfg, CKPT_DIR)

    print("\nTraining complete.")
    if device == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
