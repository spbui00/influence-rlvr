#!/bin/bash
# Unattended CS signal-check: balanced WebInstruct training pool, IF target +
# eval = MMLU-CS (disjoint), one prune at step 10. Trains THEN evaluates in one
# job so you can submit and walk away. Small but produces real CG signal.
#
#   sbatch experiments/cluster/cs_signal.slurm
#   # watch later:  tail -f slurm-cs_signal-*.out
#
#SBATCH --job-name=cs_signal
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

RUN_NAME="${RUN_NAME:-cs_signal}"
SEED="${SEED:-42}"
VENV_DIR="${VENV_DIR:-$HOME/envs/influence-rlvr}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"     # Killarney compute has internet; 0 lets first-use downloads work
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TMPDIR="${SLURM_TMPDIR:-$HOME/scratch/tmp}"; mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/wandb}"; mkdir -p "$WANDB_DIR"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/wandb/cache}"; mkdir -p "$WANDB_CACHE_DIR"

module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0 opencv/4.13.0
source "$VENV_DIR/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "================ TRAIN (if_prune, CS) ================"
srun python -m experiments.train --regime if_prune --run-name "$RUN_NAME" --seed "$SEED" \
    --if-target-source mmlu_cs \
    --eval-benchmarks mmlu_cs,gsm8k,math500 --live-eval-benchmark mmlu_cs \
    --n-train-pool 96 --n-if-target 32 \
    --max-steps 30 --prune-step 10 --if-recompute-every 0 --save-steps 10 \
    --g-train 4 --per-device-batch 8 --grad-accum 1 --max-completion-length 1024 \
    --if-g-train 4 --if-max-new-tokens 512 \
    --cg-fisher-examples 8 --cg-fisher-g 4 --cg-iters 20 --cg-fisher-max-tokens 128 \
    --verifier-max-new-tokens 512 --live-eval-examples 16 \
    --eval-max-examples 64 --eval-max-new-tokens 1024

echo "================ EVAL (latest checkpoint) ================"
srun python -m experiments.evaluate --run-name "$RUN_NAME" --checkpoint-step latest \
    --benchmarks mmlu_cs,gsm8k,math500 --eval-max-examples 64

echo "================ DONE ================"
echo "live curve : outputs/$RUN_NAME/live_eval.csv"
echo "IF scores  : outputs/$RUN_NAME/influence/step10/"
echo "final eval : outputs/$RUN_NAME/eval/"
