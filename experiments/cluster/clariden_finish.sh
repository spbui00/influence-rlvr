#!/bin/bash
# Last step: the pinned stack is in and imports; only matplotlib (a pyproject dep
# skipped by `pip install -e . --no-deps`, pulled in via analysis/plots.py) is
# missing. Install it, then prove training runs end-to-end with a 2-step CLEAN
# smoke (HF generation, --no-use-vllm).
#
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/cluster/clariden_finish.sh
#
#SBATCH --job-name=clariden-finish
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=clariden-finish-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export VENV_DIR="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
unset PYTHONPATH

echo "=== 1. matplotlib ==="
pip install matplotlib 2>&1 | tail -3

echo "=== 2. training module import ==="
TRAIN_OK=0
if python -c "import experiments.protocol2.train; print('  train import OK')" 2>err.$$; then
    TRAIN_OK=1
else
    echo "  train import FAILED:"; tail -25 err.$$
fi
rm -f err.$$

echo "=== 3. 2-step CLEAN smoke, HF generation (--no-use-vllm) ==="
if [ "$TRAIN_OK" = 1 ]; then
    SMOKE_POOL="experiments/protocol2/dataset/data/pool_clean_smoke.jsonl"
    rm -f "$SMOKE_POOL"
    RUN_NAME=smoke_finish OUTPUT_DIR="$SCRATCH/p2_runs/smoke_finish" \
    ATTACK=none POOL="$SMOKE_POOL" \
    MAX_STEPS=2 SAVE_STEPS=2 PER_DEVICE=8 GRAD_ACCUM=2 EXTRA="--no-use-vllm" \
    bash experiments/protocol2/scripts/train.slurm 2>&1 | tail -50
    echo "=== smoke exit: ${PIPESTATUS[0]} ==="
else
    echo "  SKIP smoke — train import failed"
fi
echo "=== clariden_finish done ==="
