#!/bin/bash
# Evaluate a checkpoint on the benchmark suites.
#
#   sbatch --export=ALL,RUN_NAME=qwen3_4b_base,STEP=latest experiments/cluster/eval.slurm
#
#SBATCH --job-name=ifrlvr-eval
#SBATCH --account=rrg-zhijing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

RUN_NAME="${RUN_NAME:?set RUN_NAME}"
STEP="${STEP:-latest}"
BENCHMARKS="${BENCHMARKS:-}"

VENV_DIR="${VENV_DIR:-$HOME/envs/influence-rlvr}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${SLURM_TMPDIR:-$HOME/scratch/tmp}"; mkdir -p "$TMPDIR"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/wandb}"; mkdir -p "$WANDB_DIR"

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0
source "$VENV_DIR/bin/activate"
cd "$SLURM_SUBMIT_DIR"

BENCH_ARG=""
[ -n "$BENCHMARKS" ] && BENCH_ARG="--benchmarks $BENCHMARKS"

srun python -m experiments.evaluate \
    --run-name "$RUN_NAME" \
    --checkpoint-step "$STEP" \
    $BENCH_ARG
