#!/bin/bash
# Smoke-test the JVP (forward-mode) influence scorer vs the reverse-mode dot, on the real
# Qwen3+LoRA gold-NLL loss. Single GPU, no servers (gold = no generation/verifier). ~3 min.
# PASS = Spearman ~1.0 + tiny max-diff -> JVP is correct, wire it into the scorer.
#
#   RUN_NAME=math_if_v2 CKPT=10 VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=debug --time=00:20:00 --mem=64G \
#            --gres=gpu:1 --cpus-per-task=24 experiments/cluster/jvp_smoke.slurm
#
#SBATCH --job-name=jvp-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-math_if_v2}"
CKPT="${CKPT:-10}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
N_POOL="${N_POOL:-16}"          # pool examples compared (dot vs jvp)
POOL_TOTAL="${POOL_TOTAL:-60}"  # --n-train-pool; must be >= N_POOL (balanced over 3 domains)
TARGET="${TARGET:-8}"           # IF target size (h = mean over these)
DTYPE="${DTYPE:-fp32}"          # fp32 (precision reference) | bf16 (production dtype)
TEST_DOMAINS="${TEST_DOMAINS:-physics}"

python -u -m experiments.jvp_smoke --checkpoint-step "$CKPT" --n-pool "$N_POOL" --dtype "$DTYPE" \
    --run-name "$RUN_NAME" --model-id "$MODEL" --lora-r 32 \
    --domains math,physics,finance --n-train-pool "$POOL_TOTAL" \
    --test-from-train --test-from-train-eval 1000 \
    --webinstruct-test-domains "$TEST_DOMAINS" --n-if-target "$TARGET" --seed 42
