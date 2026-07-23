#!/bin/bash
# Repair the ifrlvr venv after a $SCRATCH eviction, then PROVE the training code
# runs on the current stack. The eviction dropped only a couple of pure-python
# deps (typing_extensions dir gone but its .dist-info lingers -> force-reinstall;
# math_verify absent) and broke the editable stub / setuptools shim. The heavy
# wheels (torch/vllm/numpy/...) are intact on disk. This:
#   1. refreshes pip/setuptools/wheel  (fixes _distutils_hack add_shim + editable finder)
#   2. restores typing_extensions (force) + math_verify
#   3. rebuilds the influence_rlvr editable stub
#   4. import-probes the whole stack
#   5. imports the training module (catches trl/transformers API breakage at import)
#   6. runs a 2-step CLEAN smoke via train.slurm (catches runtime API breakage)
#
# Run INSIDE the container:
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/cluster/clariden_fix.sh
#
#SBATCH --job-name=clariden-fix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=clariden-fix-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export VENV_DIR="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "python: $(which python)"

echo "=== 1. refresh build tooling (fixes add_shim + editable finder) ==="
pip install -U pip setuptools wheel 2>&1 | tail -3

echo "=== 2. restore missing deps (typing_extensions dir gone; math_verify absent) ==="
pip install --force-reinstall --no-deps typing_extensions 2>&1 | tail -2
pip install --no-deps math-verify 2>&1 | tail -3
python -c "import math_verify" 2>/dev/null || { echo "  math-verify --no-deps import failed, retry with deps"; pip install math-verify 2>&1 | tail -3; }

echo "=== 3. rebuild editable stub ==="
pip install -e . --no-deps 2>&1 | tail -2

echo "=== 4. import probe ==="
python - <<'PY'
import importlib
for m in ["typing_extensions","torch","numpy","pyarrow","datasets","transformers",
          "trl","peft","accelerate","math_verify","vllm","influence_rlvr"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:<14} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:<14} {type(e).__name__}: {str(e)[:110]}")
PY

echo "=== 5. training module imports on this stack? ==="
TRAIN_OK=0
if python -c "import experiments.protocol2.train" 2>err.$$; then
    echo "  train module import OK"; TRAIN_OK=1
else
    echo "  train import FAILED:"; tail -20 err.$$
fi
rm -f err.$$

echo "=== 6. 2-step CLEAN smoke (only if train imports) ==="
if [ "$TRAIN_OK" = 1 ]; then
    SMOKE_POOL="experiments/protocol2/dataset/data/pool_clean_smoke.jsonl"
    rm -f "$SMOKE_POOL"
    RUN_NAME=smoke_fix \
    OUTPUT_DIR="$SCRATCH/p2_runs/smoke_fix" \
    ATTACK=none POOL="$SMOKE_POOL" \
    MAX_STEPS=2 SAVE_STEPS=2 PER_DEVICE=8 GRAD_ACCUM=2 VLLM_GPU_MEM=0.30 \
    bash experiments/protocol2/scripts/train.slurm 2>&1 | tail -45
    echo "=== smoke exit: ${PIPESTATUS[0]} ==="
else
    echo "  SKIP smoke — train import failed (likely trl/transformers API break on the newer stack)"
fi
echo "=== clariden_fix done ==="
