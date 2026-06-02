#!/bin/bash
# ONE-TIME environment build on a Digital Research Alliance cluster (Killarney /
# Narval / Nibi). Run once to create the venv. For every session afterwards you
# `source experiments/cluster/env.sh` instead (modules + activate) — env.sh does
# NOT replace this; this builds the venv, env.sh opens it.
#
#   bash experiments/cluster/setup.sh           # build (login node or a node with internet)
#   python experiments/cluster/prefetch.py      # then cache models + datasets
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
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0 opencv/4.13.0 || {
    echo "Module load failed — run 'module spider python cuda arrow' and edit this script."
    exit 1
}

# ── Virtualenv (Alliance recommends virtualenv over conda) ──────────────────
if [ ! -d "$VENV_DIR" ]; then
    virtualenv --no-download "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# Heavy ABI-sensitive wheels from the cluster wheelhouse (--no-index). Pulling
# `datasets` here binds it to the module's PyArrow — a PyPI datasets would demand
# pyarrow>=21 and hit Alliance's dummy "pyarrow_noinstall" wheel. (torchvision/
# torchaudio are intentionally NOT installed — unused, and their cluster pins
# mismatch torch.)
pip install --no-index torch datasets numpy scipy pandas

# Our package code only, WITHOUT re-resolving heavy deps — `pip install -e .`
# (with deps) would re-resolve datasets→pyarrow and rebuild the dummy wheel.
pip install -e "$PROJECT_DIR" --no-deps

# Remaining runtime deps. transformers/trl/peft/accelerate come from the
# wheelhouse and pull most transitive deps; the trailing names backfill torch's
# runtime deps that --no-deps skipped (these caused import errors otherwise).
pip install transformers accelerate trl peft math-verify wandb \
            typing_extensions sympy mpmath networkx jinja2 filelock fsspec

python -c "import torch, transformers, trl, peft, datasets, pyarrow, numpy, scipy, pandas; \
print('deps import OK —', 'torch', torch.__version__, '| trl', trl.__version__)"

echo
echo "Optional vLLM (big; for generation speed at scale — runs use --no-use-vllm without it):"
pip install vllm || echo "vLLM install skipped — set --no-use-vllm in jobs (HF generation)."

mkdir -p "$HF_HOME"
echo
echo "Done. Next:"
echo "  python experiments/cluster/prefetch.py       # cache models + datasets"
echo "  # then every session:  source experiments/cluster/env.sh"
