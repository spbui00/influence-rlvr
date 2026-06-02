#!/bin/bash
# Matched BASELINE for cs_signal.slurm: identical balanced WebInstruct pool, same
# budget (30 steps), same MMLU-CS eval partition — but NO influence pruning (plain
# GRPO on the full pool). The only difference vs cs_signal is the if_prune
# intervention, so the two are a fair A/B. Trains THEN evaluates in one job.
#
#   sbatch experiments/cluster/cs_baseline.slurm
#   # watch later:  tail -f slurm-cs_baseline-*.out
#
#SBATCH --job-name=cs_baseline
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

RUN_NAME="${RUN_NAME:-cs_baseline}"
SEED="${SEED:-42}"
VENV_DIR="${VENV_DIR:-$HOME/envs/influence-rlvr}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TMPDIR="${SLURM_TMPDIR:-$HOME/scratch/tmp}"; mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/wandb}"; mkdir -p "$WANDB_DIR"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/wandb/cache}"; mkdir -p "$WANDB_CACHE_DIR"

module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0 opencv/4.13.0
source "$VENV_DIR/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# --no-use-vllm: match cs_signal's HF training backend (if_prune falls back to HF),
#   so backend isn't a confound. n-if-target 32 is kept ONLY to make the MMLU-CS
#   eval partition identical to cs_signal's (baseline doesn't prune/use a target).
echo "================ TRAIN (baseline, full pool) ================"
srun python -m experiments.train --regime baseline --run-name "$RUN_NAME" --seed "$SEED" \
    --eval-benchmarks mmlu_cs,gsm8k,math500 --live-eval-benchmark mmlu_cs \
    --n-train-pool 96 --n-if-target 32 \
    --max-steps 30 --save-steps 10 \
    --g-train 4 --per-device-batch 8 --grad-accum 1 --max-completion-length 1024 \
    --verifier-max-new-tokens 512 --live-eval-examples 16 \
    --eval-max-examples 64 --eval-max-new-tokens 1024 \
    --no-use-vllm

echo "================ EVAL (latest checkpoint) ================"
srun python -m experiments.evaluate --run-name "$RUN_NAME" --checkpoint-step latest \
    --benchmarks mmlu_cs,gsm8k,math500 --eval-max-examples 64

echo "================ DONE ================"
echo "live curve : outputs/$RUN_NAME/live_eval.csv"
echo "final eval : outputs/$RUN_NAME/eval/"
