#!/bin/bash
# P1 Stage-1: THE reference training run — full pool, checkpoints WITH optimizer.pt
# every SAVE_STEPS (default 50 -> checkpoints 50,100,150,200: the grid the LDS
# checkpoint sweep scores and branches from; tracin-adam reads each optimizer.pt).
# ~32 prompts/step x G=8, ~0.5-1 min/step on a GH200 -> 200 steps ≈ 2-3.5 h.
# Ends with the target eval -> target_eval.json (y_ref). Preempted? Resubmit the
# SAME command — train.py auto-resumes from the latest full checkpoint, and a
# finished run is a no-op (idempotent).
#
#   Clariden:  REF_DIR=$SCRATCH/p1_runs/p1_ref \
#              VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#              sbatch --environment=pt --account=a0133 --partition=normal \
#                     --gres=gpu:1 --mem=200G --time=6:00:00 --cpus-per-task=32 \
#                     experiments/protocol1/scripts/ref.slurm
#
#   Smoke first:  REF_STEPS=4 SAVE_STEPS=2 EVAL_SAMPLES=2 sbatch ... ref.slurm
#
#SBATCH --job-name=p1-ref
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-p1_ref}"
REF_DIR="${REF_DIR:-experiments/protocol1/outputs/$RUN_NAME}"
REF_STEPS="${REF_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SEED="${SEED:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
EXTRA="${EXTRA:-}"

DATA_DIR="experiments/protocol1/dataset/data"
# Stage-0 data (seeded, no GPU) — assemble if absent. Needs protocol2's pilot artifacts.
if [ ! -f "$DATA_DIR/pool.jsonl" ]; then
    echo "P1 data missing — assembling pool/targets/subsets..."
    python -m experiments.protocol1.make_pool
fi

echo "================ P1 REF: $RUN_NAME  steps=$REF_STEPS save=$SAVE_STEPS -> $REF_DIR ================"
python -m experiments.protocol1.train \
    --run-name "$RUN_NAME" --output-dir "$REF_DIR" \
    --max-steps "$REF_STEPS" --save-steps "$SAVE_STEPS" --seed "$SEED" \
    --eval-samples "$EVAL_SAMPLES" $EXTRA

echo "checkpoints $SAVE_STEPS..$REF_STEPS (adapter + optimizer.pt) under $REF_DIR; y_ref in target_eval.json"
echo "next: per checkpoint C — score.slurm (REF_STEP=C) + subsets.slurm (REF_STEP=C, array over subsets)"
