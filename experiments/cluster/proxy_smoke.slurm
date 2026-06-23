#!/bin/bash
# Smoke-test the cheap magnitude-PROXY cosine (⟨h,g⟩/(NLL·√tokens)) vs true cosine
# (⟨h,g⟩/|g_train|). Single GPU, no servers (gold reverse-mode grads). ~10-20 min for
# a ~500 pool. PASS = a proxy with high Spearman vs true-cosine AND a target-domain-
# concentrated top-20% (matching true cosine, NOT the raw dot's Economics tilt).
#
#   RUN_NAME=uniphys_if CKPT=10 N_POOL=500 MODEL=Qwen/Qwen3-4B-Base \
#   POOL=4000 TEST_DOMAINS=physics TARGET_DIFFICULTY=University \
#   POOL_DIFFICULTY=Junior_High_School,Senior_High_School \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=debug --time=00:30:00 --mem=64G \
#            --gres=gpu:1 --cpus-per-task=24 experiments/cluster/proxy_smoke.slurm
#
#SBATCH --job-name=proxy-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-uniphys_if}"
CKPT="${CKPT:-10}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
N_POOL="${N_POOL:-500}"
POOL="${POOL:-4000}"
DOMAINS="${DOMAINS:-physics,math,finance}"
TEST_DOMAINS="${TEST_DOMAINS:-physics}"
N_IF_TARGET="${N_IF_TARGET:-256}"
TEST_EVAL="${TEST_EVAL:-1000}"
TARGET_DIFFICULTY="${TARGET_DIFFICULTY:-}"
POOL_DIFFICULTY="${POOL_DIFFICULTY:-}"
DIFF_ARG=""
[ -n "$TARGET_DIFFICULTY" ] && DIFF_ARG="$DIFF_ARG --target-difficulty $TARGET_DIFFICULTY"
[ -n "$POOL_DIFFICULTY" ] && DIFF_ARG="$DIFF_ARG --pool-difficulty $POOL_DIFFICULTY"

python -u -m experiments.proxy_smoke --checkpoint-step "$CKPT" --n-pool "$N_POOL" \
    --run-name "$RUN_NAME" --model-id "$MODEL" --lora-r 32 \
    --domains "$DOMAINS" --n-train-pool "$POOL" \
    --test-from-train --test-from-train-eval "$TEST_EVAL" \
    --webinstruct-test-domains "$TEST_DOMAINS" --n-if-target "$N_IF_TARGET" \
    --if-method tracin-adam $DIFF_ARG
