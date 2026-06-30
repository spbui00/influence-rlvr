#!/bin/bash
# Re-score a checkpoint's pool with a method variant (e.g. #2 common-mode projection),
# reusing the run's saved config so cosine/difficulty/target match the original scoring.
# GOLD scoring needs NO servers — 1 GPU. Output is LDS-scorable against the original scores.
#
#   RUN_NAME=xdomain_phys_gold_c CKPT=10 TAG=proj_toppc COMMON_MODE=top-pc \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#            --mem=200G --time=3:00:00 --cpus-per-task=24 experiments/cluster/rescore.slurm
#   (COMMON_MODE=pool-mean for the external-format-direction variant.)
#
#SBATCH --job-name=ifrlvr-rescore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:?set RUN_NAME}"
CKPT="${CKPT:?set CKPT (checkpoint step to score)}"
TAG="${TAG:-rescore}"
COMMON_MODE="${COMMON_MODE:-}"           # "" | top-pc | pool-mean
IF_METHOD="${IF_METHOD:-}"               # optional override (else run's saved method)
IF_GRAD="${IF_GRAD:-}"                   # optional override (gold = serverless)

ARGS="--run-name $RUN_NAME --checkpoint-step $CKPT --tag $TAG"
[ -n "$COMMON_MODE" ] && ARGS="$ARGS --if-common-mode $COMMON_MODE"
[ -n "$IF_METHOD" ] && ARGS="$ARGS --if-method $IF_METHOD"
[ -n "$IF_GRAD" ] && ARGS="$ARGS --if-grad $IF_GRAD"

echo "GPU state at launch:"; nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || true
echo "================ RESCORE ($RUN_NAME ckpt-$CKPT tag=$TAG common_mode=${COMMON_MODE:-off}) ================"
CUDA_VISIBLE_DEVICES=0 python -u -m experiments.rescore_pool $ARGS
echo "================ DONE — scores under outputs/$RUN_NAME/influence/${TAG}_step${CKPT}/ ================"
