#!/bin/bash
# One-time environment setup on a Digital Research Alliance of Canada cluster
# (Killarney / Narval / Nibi). RUN THIS ON A LOGIN NODE — compute nodes have no
# internet, so the venv build and all Hugging Face downloads must happen here.
#
#   bash experiments/cluster/setup.sh
#
# After it finishes, run experiments/cluster/prefetch.py (also on the login node)
# to cache the model + datasets, then submit jobs with sbatch.
set -euo pipefail

# ── Where the persistent virtualenv and HF cache live ──────────────────────
# ~/scratch has the space and is readable from compute nodes.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/envs/influence-rlvr}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

echo "Project:  $PROJECT_DIR"
echo "Venv:     $VENV_DIR"
echo "HF_HOME:  $HF_HOME"

# ── Modules. Versions differ slightly per cluster — `module spider python` to
#    check. StdEnv/2023 + a recent python + cuda + arrow (for pyarrow/datasets). ─
module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/17.0.0 || {
    echo "Module load failed — run 'module spider python cuda arrow' and edit this script."
    exit 1
}

# ── Virtualenv (Alliance recommends virtualenv over conda) ──────────────────
if [ ! -d "$VENV_DIR" ]; then
    virtualenv --no-download "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# Prefer the cluster wheelhouse (--no-index) for the heavy, ABI-sensitive
# packages; fall back to PyPI for the rest (login nodes have internet).
pip install --no-index torch torchvision torchaudio || \
    echo "torch not in wheelhouse; will resolve from PyPI below."

# Editable install of this repo + its deps. vLLM is optional and large; install
# it only if the wheelhouse/PyPI resolves it for this cluster's CUDA.
pip install -e "$PROJECT_DIR"
pip install "trl>=0.17" "peft>=0.14" "transformers>=4.56,<5" "datasets>=3.0" \
            "accelerate>=1.0" "math-verify" wandb || true

echo
echo "Trying optional vLLM (skip on failure; you can run --no-use-vllm with HF generate):"
pip install "vllm==0.11.2" || echo "vLLM install skipped — set --no-use-vllm in jobs."

mkdir -p "$HF_HOME"
echo
echo "Done. Next:"
echo "  source $VENV_DIR/bin/activate"
echo "  HF_HOME=$HF_HOME python experiments/cluster/prefetch.py"
