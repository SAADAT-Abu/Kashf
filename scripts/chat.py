#!/usr/bin/env python3
"""
Pull the latest Kashf checkpoint from GCS and run an interactive completion session.

Usage:
    python scripts/chat.py                        # latest checkpoint from GCS
    python scripts/chat.py --ckpt path/to/step.pt # local checkpoint
    python scripts/chat.py --temperature 1.0 --top-k 40 --max-tokens 300
"""

import argparse
import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from kashf.model import KashfConfig, KashfModel

GCS_PREFIX = "gs://kashf-checkpoints-eu/fineweb-v6e-run1"
LOCAL_CACHE = os.path.expanduser("~/.cache/kashf_checkpoints")


# ── Checkpoint download ──────────────────────────────────────────────────────

def _gsutil(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gsutil", *args], capture_output=True, text=True)


def pull_latest(gcs_prefix: str, cache_dir: str) -> str:
    print(f"Listing checkpoints at {gcs_prefix} …", flush=True)
    r = _gsutil("ls", f"{gcs_prefix}/step_*.pt")
    if r.returncode != 0:
        sys.exit(f"gsutil ls failed:\n{r.stderr.strip()}\n"
                 "Make sure gcloud is authenticated: gcloud auth login")

    remote_paths = sorted(line.strip() for line in r.stdout.splitlines() if line.strip())
    if not remote_paths:
        sys.exit("No checkpoints found in GCS bucket.")

    latest_remote = remote_paths[-1]
    filename      = os.path.basename(latest_remote)
    local_path    = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        print(f"Found cached checkpoint: {local_path}")
    else:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Downloading {latest_remote} → {local_path} …", flush=True)
        r = _gsutil("cp", latest_remote, local_path)
        if r.returncode != 0:
            sys.exit(f"gsutil cp failed:\n{r.stderr.strip()}")
        print("Download complete.")

    return local_path


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> tuple[KashfModel, KashfConfig, int]:
    print(f"Loading checkpoint: {ckpt_path} …", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg   = ckpt["cfg"]
    step  = int(ckpt["step"])

    model = KashfModel(cfg)
    model.load_state_dict(ckpt["model"])
    # keep bfloat16 on GPU (native); cast to float32 on CPU (limited bfloat16 support)
    model = model.to(device) if device == "cuda" else model.float()
    model.eval()
    return model, cfg, step


# ── Generation ───────────────────────────────────────────────────────────────

def complete(
    model: KashfModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt")
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        n_loops=model.cfg.max_loop_iters,
        temperature=temperature,
        top_k=top_k,
    )
    new_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chat with a local Kashf checkpoint")
    parser.add_argument("--ckpt",        default=None,     help="Local .pt path (skips GCS download)")
    parser.add_argument("--gcs",         default=GCS_PREFIX)
    parser.add_argument("--cache-dir",   default=LOCAL_CACHE)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k",       type=int,   default=50)
    parser.add_argument("--max-tokens",  type=int,   default=200)
    args = parser.parse_args()

    ckpt_path = args.ckpt or pull_latest(args.gcs, args.cache_dir)
    model, cfg, step = load_model(ckpt_path)

    device   = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nKashf  |  step {step:,}  |  {n_params/1e6:.1f}M params  |  "
          f"device={device}  loop_iters={cfg.max_loop_iters}  seq_len={cfg.max_seq_len}")
    print(f"temp={args.temperature}  top_k={args.top_k}  max_tokens={args.max_tokens}")
    print("─" * 60)
    print("Enter a prompt and press Enter to complete  (Ctrl-C to quit)\n")

    tok = AutoTokenizer.from_pretrained("gpt2")

    while True:
        try:
            prompt = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt.strip():
            continue

        try:
            output = complete(model, tok, prompt,
                              max_new_tokens=args.max_tokens,
                              temperature=args.temperature,
                              top_k=args.top_k)
            print(output)
            print()
        except Exception as exc:
            print(f"[error] {exc}")


if __name__ == "__main__":
    main()
