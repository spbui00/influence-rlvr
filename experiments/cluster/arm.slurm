#!/bin/bash
# One arm of the 4-arm reward-curve experiment. Submit 4 of these (one per arm),
# they run in parallel on separate GPUs. Each does train (with live-eval reward
# curve) then a final benchmark eval.
#
#   RUN_NAME=exp_baseline    SEED=42 ARM_ARGS="--regime baseline --no-use-vllm"                       sbatch experiments/cluster/arm.slurm
#   RUN_NAME=exp_ifg_dot     SEED=42 ARM_ARGS="--regime if_prune --selection if-guided --if-method dot" sbatch experiments/cluster/arm.slurm
#   RUN_NAME=exp_anti_dot    SEED=42 ARM_ARGS="--regime if_prune --selection anti-if  --if-method dot" sbatch experiments/cluster/arm.slurm
#   RUN_NAME=exp_ifg_cg      SEED=42 ARM_ARGS="--regime if_prune --selection if-guided --if-method cg"  sbatch experiments/cluster/arm.slurm
#
#   reward curve per run -> outputs/<RUN_NAME>/live_eval.csv
#
#SBATCH --job-name=ifrlvr_arm
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:?set RUN_NAME}"
SEED="${SEED:-42}"
ARM_ARGS="${ARM_ARGS:?set ARM_ARGS}"

# Shared experiment config — identical across all arms so the ONLY difference is
# the arm (regime/selection/if-method). One-shot prune at step 40 (recompute 0),
# CS target+eval, the validated influence settings (if_g_train 8, 1024 tok,
# normalized Fisher, lambda 0.5, micro-batch 1, score-batch 4).
COMMON="--if-target-source mmlu_cs --eval-benchmarks mmlu_cs,gsm8k,math500 \
    --live-eval-benchmark mmlu_cs --n-train-pool 96 --n-if-target 16 \
    --max-steps 100 --prune-step 40 --if-recompute-every 0 --save-steps 20 \
    --g-train 4 --per-device-batch 8 --grad-accum 1 --max-completion-length 1024 \
    --if-g-train 8 --if-max-new-tokens 1024 --cg-fisher-examples 8 --cg-fisher-g 2 \
    --cg-fisher-max-tokens 128 --cg-iters 50 --lambda-damp 0.5 \
    --if-logps-micro-batch 1 --if-score-batch 4 \
    --live-eval-examples 32 --eval-max-examples 200"

echo "================ TRAIN ($RUN_NAME) ================"
echo "arm: $ARM_ARGS"
srun python -m experiments.train --run-name "$RUN_NAME" --seed "$SEED" $COMMON $ARM_ARGS

echo "================ EVAL ($RUN_NAME) ================"
srun python -m experiments.evaluate --run-name "$RUN_NAME" --checkpoint-step latest \
    --benchmarks mmlu_cs,gsm8k,math500 --eval-max-examples 200

echo "================ DONE ================"
echo "reward curve: outputs/$RUN_NAME/live_eval.csv"
echo "final eval  : outputs/$RUN_NAME/eval/"
