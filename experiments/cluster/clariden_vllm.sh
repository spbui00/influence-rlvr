#!/bin/bash
# Add vllm to the restored ifrlvr venv (the container doesn't ship it, and the
# clean 200-step run needs it — HF generation is days at 64 prompts/step). The old
# Clariden venv paired vllm 0.23.0 with torch 2.11, so pin that. WATCH the log: if
# vllm tries to change torch away from 2.11.0+cu130, that's a pin conflict to flag.
# Verifies vllm imports + a 2-step CLEAN smoke WITH vllm colocate.
#
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/cluster/clariden_vllm.sh
#
#SBATCH --job-name=clariden-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=clariden-vllm-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export VENV_DIR="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
unset PYTHONPATH

echo "=== torch BEFORE vllm ==="; python -c "import torch; print(' ', torch.__version__)"
echo "=== install vllm==0.23.0 ==="
pip install "vllm==0.23.0" 2>&1 | tail -20
echo "=== torch AFTER vllm (must still be 2.11.0+cu130) ==="; python -c "import torch; print(' ', torch.__version__)"
python -c "import vllm; print('  vllm', vllm.__version__)" 2>&1 | tail -3

echo "=== 2-step CLEAN smoke WITH vllm colocate ==="
SMOKE_POOL="experiments/protocol2/dataset/data/pool_clean_smoke.jsonl"
rm -f "$SMOKE_POOL"
RUN_NAME=smoke_vllm OUTPUT_DIR="$SCRATCH/p2_runs/smoke_vllm" \
ATTACK=none POOL="$SMOKE_POOL" \
MAX_STEPS=2 SAVE_STEPS=2 PER_DEVICE=8 GRAD_ACCUM=2 VLLM_GPU_MEM=0.30 \
bash experiments/protocol2/scripts/train.slurm 2>&1 | tail -50
echo "=== smoke exit: ${PIPESTATUS[0]} ==="
echo "=== clariden_vllm done ==="
