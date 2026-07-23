#!/bin/bash
# The --system-site-packages rebuild inherited the CONTAINER's torch, which is
# 2.6.0a0+nv25.01 -- a pre-stable alpha that lacks the public FSDPModule API that
# modern trl imports. The working Clariden venv never used the container torch; it
# carried its OWN modern torch 2.11 + trl 1.6. So: install a modern torch (shadows
# the container alpha) and the trl version that actually ran here (1.6.0, NOT the
# Alliance pin 1.2.0), then prove the training code imports + runs (HF gen).
#
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/cluster/clariden_torchfix.sh
#
#SBATCH --job-name=clariden-torchfix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=clariden-torchfix-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export VENV_DIR="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "python: $(which python)"

echo "=== 1. modern torch (has FSDPModule; shadows container 2.6-alpha) ==="
python -c "import torch; print('  before:', torch.__version__)"
pip install -U torch 2>&1 | tail -6
python -c "import torch; print('  after :', torch.__version__)"
python -c "from torch.distributed.fsdp import FSDPModule; print('  FSDPModule import OK')" 2>&1 | tail -3

echo "=== 2. trl 1.6.0 (Clariden-proven; replaces the wrong 1.2.0 pin) ==="
pip install "trl==1.6.0" 2>&1 | tail -8

echo "=== 3. import probe ==="
python - <<'PY'
import importlib
for m in ["torch","numpy","pyarrow","datasets","transformers","trl","peft",
          "accelerate","math_verify","influence_rlvr"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:<14} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:<14} {type(e).__name__}: {str(e)[:110]}")
PY

echo "=== 4. training module import ==="
TRAIN_OK=0
if python -c "import experiments.protocol2.train" 2>err.$$; then
    echo "  train module import OK"; TRAIN_OK=1
else
    echo "  train import FAILED:"; tail -25 err.$$
fi
rm -f err.$$

echo "=== 5. 2-step CLEAN smoke, HF generation (--no-use-vllm) ==="
if [ "$TRAIN_OK" = 1 ]; then
    SMOKE_POOL="experiments/protocol2/dataset/data/pool_clean_smoke.jsonl"
    rm -f "$SMOKE_POOL"
    RUN_NAME=smoke_torchfix OUTPUT_DIR="$SCRATCH/p2_runs/smoke_torchfix" \
    ATTACK=none POOL="$SMOKE_POOL" \
    MAX_STEPS=2 SAVE_STEPS=2 PER_DEVICE=8 GRAD_ACCUM=2 EXTRA="--no-use-vllm" \
    bash experiments/protocol2/scripts/train.slurm 2>&1 | tail -45
    echo "=== smoke exit: ${PIPESTATUS[0]} ==="
else
    echo "  SKIP smoke — train import failed"
fi
echo "=== clariden_torchfix done ==="
