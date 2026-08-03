#!/bin/bash
# P1 Stage-3: the LDS retraining sweep — one array task per subset. Task m loads
# the checkpoint-REF_STEP adapter, continues GRPO on subsets.npy[m] for
# CONT_STEPS, evaling the targets IN-TRAINING at every EVAL_AT horizon, and
# writes $RUNS_ROOT/step<REF_STEP>/subset_<m>/target_eval.json (one per-horizon
# block per eval — the whole LDS-vs-continuation-length curve from ONE run:
# how far ahead the IF prediction stays good before it drifts).
#
# CHECKPOINT SWEEP (LDS per ref step): submit once per checkpoint — the SAME
# subsets.npy is reused at every step, so subsets are comparable across steps
# and lds.py reports one LDS row per (step, horizon, influence variant). The
# manifest holds 100 subsets; the --array range decides M (LDS noise ≈ 1/sqrt(M):
# 32→±0.18, 64→±0.13, 100→±0.10). The plan is 2 checkpoints x M=100:
#     for C in 100 200; do
#       REF_STEP=$C REF_DIR=... RUNS_ROOT=... sbatch --array=0-99%12 ... subsets.slurm
#     done
#   (each (step, subset) run is ~1.7-3.3 h train + ~25 min evals on a GH200 —
#    2 steps x 100 subsets ≈ 400-800 GPU-h; ~1.4-2.8 days at 12 concurrent GPUs,
#    2.1-4.2 days at 8. Raise/lower the %N throttle to your allocation; tight →
#    --array=0-63 (±0.13) and extend with --array=64-99 later. All extendable:
#    a task whose target_eval.json already has the CONT_STEPS horizon exits
#    immediately; a partial run RESTARTS and overwrites.)
#
#   Clariden (100 subsets from one checkpoint, a few tasks at a time):
#     REF_DIR=$SCRATCH/p1_runs/p1_ref REF_STEP=200 RUNS_ROOT=$SCRATCH/p1_runs \
#     VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal \
#            --gres=gpu:1 --mem=200G --time=6:00:00 --cpus-per-task=32 \
#            --array=0-99%12 experiments/protocol1/scripts/subsets.slurm
#
#   Smoke:  CONT_STEPS=4 EVAL_AT=2 EVAL_SAMPLES=2 sbatch ... --array=0 subsets.slurm
#
#SBATCH --job-name=p1-subset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

REF_DIR="${REF_DIR:?set REF_DIR to the pi_ref run dir}"
REF_STEP="${REF_STEP:-100}"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT for the subset_<m> run dirs}"
CONT_STEPS="${CONT_STEPS:-200}"
EVAL_AT="${EVAL_AT:-1,50,100,150,200}" # horizons; final step always added. k=1 is a
                                       # null anchor (warmup-damped step -> expect LDS≈0).
                                       # For 300-step runs: CONT_STEPS=300 EVAL_AT=1,50,...,300
SUBSET_FILE="${SUBSET_FILE:-}"         # override subsets.npy (e.g. an extremes manifest)
RUN_PREFIX="${RUN_PREFIX:-subset}"     # run dir prefix: subset_<m> (LDS) or row_<m> (extremes)
SAVE_STEPS="${SAVE_STEPS:-0}"          # 0 = final ckpt only (2-4 h tasks restart cheaply
                                       # under the 12 h cap); set 50 to make individual
                                       # tasks mid-run resumable (costs ~0.5 GB/ckpt/run)
SEED="${SEED:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
EXTRA="${EXTRA:-}"

M="${SLURM_ARRAY_TASK_ID:?run via sbatch --array}"
INIT="$REF_DIR/checkpoint-$REF_STEP"
[ -d "$INIT" ] || { echo "ERROR: pi_ref checkpoint $INIT not found"; exit 1; }
OUT="$RUNS_ROOT/step$REF_STEP/${RUN_PREFIX}_$M"
SUBSET_FLAG=""; [ -n "$SUBSET_FILE" ] && SUBSET_FLAG="--subset-file $SUBSET_FILE"

# Done only when the FINAL horizon is measured (partial files come from preempted
# runs — those restart from the ref checkpoint and overwrite).
if [ -f "$OUT/target_eval.json" ] && python -c "
import json, sys
d = json.load(open('$OUT/target_eval.json'))
sys.exit(0 if '$CONT_STEPS' in d.get('horizons', {}) else 1)"; then
    echo "subset $M @ step $REF_STEP fully measured (horizon $CONT_STEPS present) — nothing to do."
    exit 0
fi

echo "============ P1 SUBSET $M @ step $REF_STEP: continue $CONT_STEPS steps from $INIT (evals at $EVAL_AT) -> $OUT ============"
python -m experiments.protocol1.train \
    --run-name "p1_step${REF_STEP}_${RUN_PREFIX}_$M" --output-dir "$OUT" \
    --subset-id "$M" --init-adapter "$INIT" $SUBSET_FLAG \
    --max-steps "$CONT_STEPS" --save-steps "$SAVE_STEPS" --seed "$SEED" \
    --eval-at "$EVAL_AT" --eval-samples "$EVAL_SAMPLES" $EXTRA

echo "subset $M measured (horizons $EVAL_AT) -> $OUT/target_eval.json"
