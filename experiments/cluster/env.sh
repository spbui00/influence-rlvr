# Source this at the START of every interactive session / new compute node:
#
#   source experiments/cluster/env.sh
#
# Works on ALL the Alliance clusters we use (Killarney L40S, Nibi H100) — the
# module set below is the same one setup.sh builds the venv against, and it loads
# on both. It enforces the Alliance ordering rule — `module load` BEFORE activating
# the venv (the python module shadows the venv's python; arrow provides PyArrow).
# Safe to source repeatedly. Override paths via env vars (VENV_DIR, HF_HOME, ...).
#
# If a module version is unavailable on some cluster, `module spider <name>` and
# adjust this line (keep it in sync with setup.sh).

# NOTE: do NOT `module purge`/`module reset` on Alliance clusters — it strips the
# scheduler modulefiles (e.g. Killarney's /cm/local Bright Cluster Manager) that
# provide salloc/sbatch. Just layer our modules on top of the default environment.
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0 opencv/4.13.0

source "${VENV_DIR:-$HOME/envs/influence-rlvr}/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export TMPDIR="${SLURM_TMPDIR:-$HOME/scratch/tmp}"; mkdir -p "$TMPDIR"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/wandb}"; mkdir -p "$WANDB_DIR"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/wandb/cache}"; mkdir -p "$WANDB_CACHE_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOMs
# Compute nodes here have internet, so we leave HF online (warm cache is used
# anyway). Uncomment to force offline reads:
# export HF_HUB_OFFLINE=1

echo "python: $(which python)"
python -c "import torch, pyarrow; print('torch', torch.__version__, '| pyarrow', pyarrow.__version__)" \
  2>/dev/null && echo "env ready." \
  || echo "WARN: torch/pyarrow import failed — is the venv active and arrow loaded?"
