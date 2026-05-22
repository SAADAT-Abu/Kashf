#!/usr/bin/env bash
# Setup script for Kashf on a fresh TPU v6e VM (europe-west4-a, v2-alpha-tpuv6e runtime).
# Run this once on a new VM. Safe to re-run — all steps are idempotent.
#
# Usage (from your local machine):
#   gcloud compute tpus tpu-vm ssh node-3 --zone=europe-west4-a \
#     --project=kashf-494319 -- 'bash -s' < scripts/setup_tpu_vm.sh

set -euo pipefail

CONDA_DIR="$HOME/miniconda3"
ENV_NAME="kashf"
REPO_URL="https://github.com/SAADAT-Abu/Kashf"
REPO_DIR="$HOME/kashf"

# ── 1. Miniconda ──────────────────────────────────────────────────────────────
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo ">>> Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
else
    echo ">>> Miniconda already installed, skipping."
fi

CONDA="$CONDA_DIR/bin/conda"
PIP="$CONDA_DIR/envs/$ENV_NAME/bin/pip"

# ── 2. Conda environment ──────────────────────────────────────────────────────
if ! "$CONDA" env list | grep -q "^$ENV_NAME "; then
    echo ">>> Creating conda env '$ENV_NAME' with Python 3.10..."
    "$CONDA" create -n "$ENV_NAME" python=3.10 -y
else
    echo ">>> Conda env '$ENV_NAME' already exists, skipping creation."
fi

# ── 3. LD_LIBRARY_PATH hook (scoped to conda env) ────────────────────────────
ACTIVATE_D="$CONDA_DIR/envs/$ENV_NAME/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
cat > "$ACTIVATE_D/ld_library_path.sh" <<'EOF'
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/kashf/lib:${LD_LIBRARY_PATH:-}
EOF
echo ">>> LD_LIBRARY_PATH hook written."

# ── 4. PyTorch + torch_xla ───────────────────────────────────────────────────
echo ">>> Installing torch 2.6.0+cpu and torch_xla 2.6.0+libtpu..."
"$PIP" install --upgrade pip --quiet
"$PIP" install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu --quiet
"$PIP" install "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0+libtpu-cp310-cp310-manylinux_2_28_x86_64.whl" --quiet

# ── 5. datasets + transformers ────────────────────────────────────────────────
echo ">>> Installing datasets and transformers..."
"$PIP" install datasets transformers --quiet

# ── 6. Clone / update Kashf repo ─────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo ">>> Cloning Kashf repo..."
    # Avoid LD_LIBRARY_PATH shadowing system libs during git clone
    env -u LD_LIBRARY_PATH git clone "$REPO_URL" "$REPO_DIR"
else
    echo ">>> Repo already cloned, pulling latest..."
    env -u LD_LIBRARY_PATH git -C "$REPO_DIR" pull origin main
fi

# ── 7. Install kashf package ──────────────────────────────────────────────────
echo ">>> Installing kashf in editable mode..."
"$PIP" install -e "$REPO_DIR" --quiet

# ── 8. bashrc additions ───────────────────────────────────────────────────────
BASHRC="$HOME/.bashrc"
add_if_missing() {
    grep -qxF "$1" "$BASHRC" || echo "$1" >> "$BASHRC"
}
add_if_missing "source $CONDA_DIR/bin/activate $ENV_NAME"
add_if_missing "export PJRT_DEVICE=TPU"
add_if_missing "export KASHF_CKPT_DIR=/home/\$USER/kashf_checkpoints_run2"
echo ">>> .bashrc updated."

# ── 9. Checkpoint directories ────────────────────────────────────────────────
mkdir -p "$HOME/kashf_checkpoints"
mkdir -p "$HOME/kashf_checkpoints_run2"
echo ">>> Checkpoint dirs created."

# ── 10. Fix GCS write auth (override GCE read-only service account scope) ────
cat > "$HOME/.boto" <<'EOF'
[GoogleCompute]
service_account =
EOF
echo ">>> GCS write auth fixed (~/.boto service_account cleared)."

# ── 11. Verify ────────────────────────────────────────────────────────────────
echo ""
echo ">>> Verifying install..."
"$CONDA_DIR/envs/$ENV_NAME/bin/python" - <<'PYEOF'
import torch
import torch_xla
print(f"torch      : {torch.__version__}")
print(f"torch_xla  : {torch_xla.__version__}")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "To start training (run-1 on node-2), run:"
echo "  source ~/miniconda3/bin/activate kashf"
echo "  export PJRT_DEVICE=TPU"
echo "  export KASHF_CKPT_DIR=\$HOME/kashf_checkpoints"
echo "  set -a; source /home/tpu-runtime/tpu-env; set +a"
echo "  cd ~/kashf"
echo "  python training/train_fineweb_v6e.py 2>&1 | tee ~/training.log"
