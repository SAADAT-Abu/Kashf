#!/usr/bin/env python3
"""
Kashf — TPU v6e (Trillium) pretraining on FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT).
PJRT training on v6e via xmp.spawn (gRPC child-process init).

    PJRT_DEVICE=TPU python training/train_fineweb_v6e.py

Architecture:
  The v2-alpha-tpuv6e runtime serves TPU chips through a gRPC Docker container.
  xmp.spawn initialises PJRT inside spawned child processes which use the gRPC
  path instead of VFIO.  Do NOT call xm.xla_device() in main().

Data pipeline (ChunkedDiskDataset):
  Downloads FineWeb-Edu in large chunks (~400M tokens) to a local binary file,
  trains through the file via numpy memmap, then deletes it and downloads the
  next chunk.  A background thread pre-fetches the next chunk while training
  runs on the current one — the XLA compute path never blocks on the network.
  Chunk file size: ~1.6 GB (400M × int32).  Disk usage: ≤2 chunks at a time.

Batch:  8 micro × 32 accum × 2048 = 524 288 tokens/step
"""

import os
import math
import time
import threading
import argparse
import numpy as np
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
GRAD_ACCUM   = 32   # 8 × 32 × 2048 = 524 288 tokens/step

LR           = 3e-4
MIN_LR       = 3e-5
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 1000
TARGET_TOKENS = 50_000_000_000

LOG_EVERY    = 10
CKPT_EVERY   = 500
CKPT_DIR     = os.environ.get("KASHF_CKPT_DIR", "kashf_checkpoints")
GRAD_CLIP    = 1.0

# ── Chunked disk dataset ──────────────────────────────────────────────────────
# Each chunk: 400M tokens × 4 bytes (int32) ≈ 1.6 GB on disk.
# At 107K tok/s: provides ~45 min of training before the next chunk is needed.
# Max disk usage: 2 chunks concurrently (current + prefetch) = ~3.2 GB.
CHUNK_TOKENS = 400_000_000
CHUNK_DIR    = os.path.join(os.path.expanduser("~"), "kashf_data_cache")


class ChunkedDiskDataset:
    """
    Downloads FineWeb-Edu via HuggingFace streaming, accumulates CHUNK_TOKENS
    tokens into a flat int32 binary file, then serves next_batch() directly
    from a numpy memmap (no live network calls on the XLA path).

    When the current chunk is exhausted:
      1. Waits for the pre-fetched next chunk (background thread).
      2. Deletes the old file.
      3. Starts pre-fetching the following chunk in the background.

    Training never stalls: the background download of ~1.6 GB typically
    completes in < 60 s, well within the ~45 min the current chunk lasts.
    """

    def __init__(self, tokenizer, seq_len: int, ordinal: int, world_size: int):
        self.tok        = tokenizer
        self.seq_len    = seq_len
        self.ordinal    = ordinal
        self.world_size = world_size

        os.makedirs(CHUNK_DIR, exist_ok=True)

        self._stream     = self._make_stream()
        self._mmap       = None
        self._mmap_path  = None
        self._pos        = 0
        self._next_mmap  = None
        self._next_path  = None
        self._next_ready = threading.Event()

    # ── public ────────────────────────────────────────────────────────────────

    def build_first_chunk(self, mprint) -> None:
        """Build the initial chunk synchronously (called before training starts)."""
        mprint(f"  [data] downloading first chunk ({CHUNK_TOKENS // 1_000_000}M tokens) …")
        self._mmap, self._mmap_path = self._build_chunk()
        mprint(f"  [data] chunk ready — {len(self._mmap):,} tokens on disk.")
        self._launch_prefetch()

    def next_batch(self, micro_batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        needed = micro_batch * (self.seq_len + 1)

        if self._pos + needed > len(self._mmap):
            self._swap_chunk()

        raw = np.array(self._mmap[self._pos: self._pos + needed], dtype=np.int64)
        self._pos += needed

        ids = torch.from_numpy(raw).view(micro_batch, self.seq_len + 1)
        return ids[:, :-1].to(device), ids[:, 1:].to(device)

    def buf_depth(self) -> int:
        remaining = max(0, len(self._mmap) - self._pos) // (self.seq_len + 1)
        return min(remaining, 99_999)

    # ── internals ─────────────────────────────────────────────────────────────

    def _make_stream(self):
        hf_token = os.environ.get("HF_TOKEN")
        while True:
            try:
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-10BT",
                    split="train",
                    streaming=True,
                    token=hf_token,
                ).shard(num_shards=self.world_size, index=self.ordinal)
                for sample in ds:
                    yield sample["text"]
                # dataset exhausted — loop back for multi-epoch
            except Exception as exc:
                print(f"  [data] stream error: {exc!r} — retrying in 10 s", flush=True)
                time.sleep(10)

    def _build_chunk(self) -> tuple[np.memmap, str]:
        path  = os.path.join(CHUNK_DIR, f"chunk_{os.getpid()}_{time.monotonic_ns()}.bin")
        total = 0
        with open(path, "wb") as f:
            for text in self._stream:
                ids = self.tok.encode(text)
                if ids:
                    np.array(ids, dtype=np.int32).tofile(f)
                    total += len(ids)
                if total >= CHUNK_TOKENS:
                    break
        return np.memmap(path, dtype=np.int32, mode="r"), path

    def _launch_prefetch(self) -> None:
        self._next_ready.clear()

        def _worker():
            buf, path = self._build_chunk()
            self._next_mmap = buf
            self._next_path = path
            self._next_ready.set()

        threading.Thread(target=_worker, daemon=True).start()

    def _swap_chunk(self) -> None:
        self._next_ready.wait()

        # Release and delete the old chunk
        if self._mmap is not None:
            del self._mmap
        if self._mmap_path and os.path.exists(self._mmap_path):
            try:
                os.remove(self._mmap_path)
            except OSError:
                pass

        self._mmap      = self._next_mmap
        self._mmap_path = self._next_path
        self._pos       = 0
        self._next_mmap = None
        self._next_path = None
        self._launch_prefetch()


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

    # Pad LoopGate embedding when resuming from a checkpoint with fewer
    # loop iterations. New rows init to 1.0 (matches nn.init.ones_).
    key = "recurrent.loop_gate.gate.weight"
    if key in state:
        saved       = state[key]
        target_size = model.recurrent.loop_gate.gate.weight.shape[0]
        if saved.shape[0] < target_size:
            pad        = torch.ones(target_size - saved.shape[0], 1, dtype=saved.dtype)
            state[key] = torch.cat([saved, pad], dim=0)

    model.load_state_dict(state)

    # Load optimizer moments; drop any entry whose shape no longer matches
    # (e.g. LoopGate after increasing max_loop_iters). AdamW reinitialises
    # missing entries on the first step — no training correctness issue.
    opt_sd       = ckpt["optimizer"]
    model_params = list(model.parameters())
    stale        = []
    for pid, pstate in opt_sd["state"].items():
        for val in pstate.values():
            if isinstance(val, torch.Tensor) and pid < len(model_params):
                if val.shape != model_params[pid].shape:
                    stale.append(pid)
                    break
    for pid in stale:
        del opt_sd["state"][pid]

    optimizer.load_state_dict(opt_sd)
    return int(ckpt["step"])


# ── Per-process training function ────────────────────────────────────────────


def _train_fn(index: int, cli_args):
    device     = xm.xla_device()
    ordinal    = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    is_master  = xm.is_master_ordinal()

    def mprint(*a, **kw):
        if is_master:
            print(*a, **kw, flush=True)

    tok        = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tok.vocab_size

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
        lm_head_dim      = 64,
    )
    model = KashfModel(cfg).to(torch.bfloat16).to(device)

    if is_master:
        counts = model.parameter_count()
        mprint(f"Parameters : {counts['total']:,} total | {counts['unique (deduped)']:,} unique")
        mprint(f"Device     : {device}  |  world_size: {world_size}")
        mprint(f"SEQ_LEN: {SEQ_LEN}  |  micro-batch: {MICRO_BATCH}  |  grad-accum: {GRAD_ACCUM}")
        mprint(f"max_loop_iters: 4  |  lm_head_dim: 64  |  target: {TARGET_TOKENS/1e9:.0f}B tokens")

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
        for pstate in optimizer.state.values():
            for k, v in pstate.items():
                if isinstance(v, torch.Tensor):
                    pstate[k] = v.to(device)
        mprint(f"Resumed at step {start_step}")

    # ── Data pipeline ─────────────────────────────────────────────────────────
    dataset = ChunkedDiskDataset(tok, SEQ_LEN, ordinal, world_size)
    dataset.build_first_chunk(mprint)

    # ── Training constants ────────────────────────────────────────────────────
    global_batch_tok = MICRO_BATCH * world_size * GRAD_ACCUM * SEQ_LEN
    total_steps      = TARGET_TOKENS // global_batch_tok

    mprint(f"\nDataset    : FineWeb-Edu sample-10BT")
    mprint(f"Target     : {TARGET_TOKENS/1e9:.0f}B tokens  |  {total_steps:,} steps")
    mprint(f"Global batch: {global_batch_tok:,} tokens/step\n")
    mprint(
        f"{'step':>7}  {'loss':>7}  {'gnorm':>6}  {'lr':>8}"
        f"  {'tok/s':>10}  {'tokens':>9}  {'buf':>8}"
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
            logits = model(x)
            loss   = nn.functional.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1)
            ) / GRAD_ACCUM
            loss.backward()
            with torch.no_grad():
                step_loss = step_loss + loss.detach()
            xm.mark_step()

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
                f"  {buf:8,}"
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
    print("Pre-caching GPT-2 tokenizer...")
    AutoTokenizer.from_pretrained("gpt2")
    print("Spawning via xmp.spawn (nprocs=None, PJRT gRPC child-process init)…")
    xmp.spawn(_train_fn, args=(None,), nprocs=None)


if __name__ == "__main__":
    main()
