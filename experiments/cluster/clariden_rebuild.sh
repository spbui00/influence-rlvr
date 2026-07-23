#!/bin/bash
# Rebuild the ifrlvr venv the RIGHT way for the NGC PyTorch container.
#
# Why the old venv was unfixable: it was created with include-system-site-packages
# = FALSE, so inside the nvidia+pytorch container its python (a symlink to the
# container's /usr/bin/python3.12) got its imports SHADOWED by the container's own
# torch/numpy/typing_extensions, while pip wrote underneath them -> installs never
# took effect. The fix is a venv that INHERITS the container stack:
# --system-site-packages=TRUE gives us the container's ABI-hard torch 2.6nv +
# pyarrow for free, and we layer only the app deps (datasets/transformers/trl/...)
# on top.
#
# Preserves the old venv as ifrlvr.broken.<jobid> (delete later to reclaim 5.1G).
#
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/cluster/clariden_rebuild.sh
#
#SBATCH --job-name=clariden-rebuild
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=clariden-rebuild-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export VENV_DIR="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"

echo "=== 0. container base python (before any venv) ==="
which python python3.12 2>&1
python -c "import torch,pyarrow,numpy; print('base torch',torch.__version__,'| pyarrow',pyarrow.__version__,'| numpy',numpy.__version__)" 2>&1 | tail -3
python -c "import vllm; print('base vllm', vllm.__version__)" 2>&1 | tail -1

echo "=== 1. preserve old venv, create fresh --system-site-packages venv ==="
if [ -d "$VENV_DIR" ]; then mv "$VENV_DIR" "${VENV_DIR}.broken.${SLURM_JOB_ID}"; echo "  moved old -> ${VENV_DIR}.broken.${SLURM_JOB_ID}"; fi
python -m venv --system-site-packages "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "  venv python: $(which python)"
python -c "import torch,pyarrow,numpy; print('  inherited torch',torch.__version__,'| numpy',numpy.__version__)"

echo "=== 2. build tooling ==="
pip install -U pip setuptools wheel 2>&1 | tail -2

echo "=== 3. app deps (torch/pyarrow/numpy inherited from container, not reinstalled) ==="
# trl pinned to the version the training code was validated against; transformers
# resolves to a compatible one. If pip tries to pull a new torch/numpy that would
# be a red flag — watch the log.
pip install "trl==1.2.0" transformers accelerate peft datasets "math-verify" wandb typing_extensions 2>&1 | tail -15

echo "=== 4. editable install ==="
pip install -e . --no-deps 2>&1 | tail -2

echo "=== 5. import probe (venv, layered on container) ==="
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
    RUN_NAME=smoke_rebuild OUTPUT_DIR="$SCRATCH/p2_runs/smoke_rebuild" \
    ATTACK=none POOL="$SMOKE_POOL" \
    MAX_STEPS=2 SAVE_STEPS=2 PER_DEVICE=8 GRAD_ACCUM=2 EXTRA="--no-use-vllm" \
    bash experiments/protocol2/scripts/train.slurm 2>&1 | tail -45
    echo "=== smoke exit: ${PIPESTATUS[0]} ==="
else
    echo "  SKIP smoke — train import failed"
fi
echo "=== clariden_rebuild done ==="
