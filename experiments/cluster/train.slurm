#!/bin/bash
# Submit a training run on a Digital Research Alliance cluster.
#
#   sbatch --export=ALL,REGIME=baseline,RUN_NAME=qwen3_4b_base  experiments/cluster/train.slurm
#   sbatch --export=ALL,REGIME=if_prune,RUN_NAME=qwen3_4b_prune experiments/cluster/train.slurm
#
# Defaults below target Killarney (H100). For Narval (A100-40G) drop
# --gpus-per-node to 1, lower vllm-gpu-memory-utilization, and use the matching
# --account. Tune --time once you've measured a few steps.
#
#SBATCH --job-name=ifrlvr
#SBATCH --account=rrg-zhijing          # <-- your RRG group (rapi hqw-052-ab). `def-zhijing` for the default allocation.
#SBATCH --gpus-per-node=h100:2         # 2 GPUs: policy+vLLM on GPU0, verifier on GPU1. Use h100:1 to colocate.
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REGIME="${REGIME:-baseline}"
RUN_NAME="${RUN_NAME:-qwen3_4b_${REGIME}}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

VENV_DIR="${VENV_DIR:-$HOME/envs/influence-rlvr}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE=1                 # compute nodes have no internet
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"   # sync later from a login node: `wandb sync`
# Keep temp + wandb off the small/shared /tmp (it fills up; wandb makes a temp
# dir at import). $SLURM_TMPDIR is node-local SSD, auto-cleaned per job.
export TMPDIR="${SLURM_TMPDIR:-$HOME/scratch/tmp}"; mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/wandb/cache}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0
source "$VENV_DIR/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# If 2 GPUs are allocated, put the verifier on the second one so it doesn't
# fight the policy + vLLM for VRAM on GPU 0.
VERIFIER_DEVICE_ARG=""
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
    VERIFIER_DEVICE_ARG="--verifier-device cuda:1"
fi

echo "Regime=$REGIME  Run=$RUN_NAME  Seed=$SEED  $(nvidia-smi -L | wc -l) GPU(s)"

srun python -m experiments.train \
    --regime "$REGIME" \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    $VERIFIER_DEVICE_ARG \
    $EXTRA_ARGS
