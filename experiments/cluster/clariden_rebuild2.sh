#!/bin/bash
# Rebuild ifrlvr as a FULLY ISOLATED venv pinned to the exact coherent stack that
# actually ran on Clariden before the $SCRATCH purge (read off the old venv's
# dist-info). Lessons from the two failed attempts:
#   * --system-site-packages=TRUE was wrong: it inherits the container's OLD
#     torch 2.6-alpha + torchvision 0.20 pair; overriding just torch -> mismatch
#     ("torchvision::nms does not exist"). The old venv was ISOLATED (=false) and
#     carried its own coherent torch, seeing no container torchvision at all.
#   * letting pip pick latest overshot (torch 2.13+cu130, transformers 5.14 which
#     dropped BloomPreTrainedModel that peft 0.19.1 imports). PIN everything.
#   * NO torchvision/torchaudio (text-only training; they only cause conflicts).
#
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/cluster/clariden_rebuild2.sh
#
#SBATCH --job-name=clariden-rebuild2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:55:00
#SBATCH --output=clariden-rebuild2-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export VENV_DIR="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"

echo "=== 0. container PYTHONPATH (would it leak into an isolated venv?) ==="
echo "PYTHONPATH=${PYTHONPATH:-<unset>}"

echo "=== 1. preserve old venv, create fresh ISOLATED venv ==="
if [ -d "$VENV_DIR" ]; then mv "$VENV_DIR" "${VENV_DIR}.broken.${SLURM_JOB_ID}"; echo "  moved old -> ${VENV_DIR}.broken.${SLURM_JOB_ID}"; fi
python -m venv "$VENV_DIR"                 # NO --system-site-packages: fully isolated
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
unset PYTHONPATH                           # belt-and-suspenders: don't inherit container dist-packages
echo "  venv python: $(which python)"

echo "=== 2. build tooling ==="
pip install -U pip setuptools wheel 2>&1 | tail -2

echo "=== 3. pinned coherent stack (matches the old working Clariden venv; no torchvision) ==="
pip install \
    torch==2.11.0 \
    transformers==5.12.1 \
    trl==1.6.0 \
    peft==0.19.1 \
    accelerate==1.14.0 \
    datasets==5.0.0 \
    "math-verify" wandb typing_extensions scipy pandas matplotlib 2>&1 | tail -20

echo "=== 4. editable install ==="
pip install -e . --no-deps 2>&1 | tail -2

echo "=== 5. import probe ==="
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
python -c "from torch.distributed.fsdp import FSDPModule; print('  FSDPModule OK'); import torch; print('  cuda avail:', torch.cuda.is_available())" 2>&1 | tail -3

echo "=== 6. training module import ==="
TRAIN_OK=0
if python -c "import experiments.protocol2.train" 2>err.$$; then
    echo "  train module import OK"; TRAIN_OK=1
else
    echo "  train import FAILED:"; tail -25 err.$$
fi
rm -f err.$$

echo "=== 7. 2-step CLEAN smoke, HF generation (--no-use-vllm) ==="
if [ "$TRAIN_OK" = 1 ]; then
    SMOKE_POOL="experiments/protocol2/dataset/data/pool_clean_smoke.jsonl"
    rm -f "$SMOKE_POOL"
    RUN_NAME=smoke_rebuild2 OUTPUT_DIR="$SCRATCH/p2_runs/smoke_rebuild2" \
    ATTACK=none POOL="$SMOKE_POOL" \
    MAX_STEPS=2 SAVE_STEPS=2 PER_DEVICE=8 GRAD_ACCUM=2 EXTRA="--no-use-vllm" \
    bash experiments/protocol2/scripts/train.slurm 2>&1 | tail -45
    echo "=== smoke exit: ${PIPESTATUS[0]} ==="
else
    echo "  SKIP smoke — train import failed"
fi
echo "=== clariden_rebuild2 done ==="
