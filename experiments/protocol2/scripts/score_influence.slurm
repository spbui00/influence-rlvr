#!/bin/bash
# Phase-4 influence scoring for the codeword-sandbagging backdoor poison. Reuses the shared
# streaming pool scorer with a sandbag reward builder so the poison's corrupted
# advantage enters g_train. Single GPU; HF generation (influence needs backward).
#
# Runs where the checkpoints live (they carry optimizer.pt for tracin-adam).
# ROLLOUT mode is the detector; GOLD mode is the reward-blind control.
#
#   Clariden (rollout, tracin-adam, all checkpoints -> P1 + P2):
#     RUN_DIR=$SCRATCH/p2_runs/p2_backdoor_v1 IF_GRAD=rollout IF_METHOD=tracin-adam \
#     VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal \
#            --gres=gpu:1 --mem=200G --time=6:00:00 --cpus-per-task=32 \
#            experiments/protocol2/scripts/score_influence.slurm
#
#   Cheap first pass — ONE mid checkpoint:  CHECKPOINTS=100 TIME shorter.
#   The control:                            IF_GRAD=gold  (fast, no generation).
#
#SBATCH --job-name=p2-score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

RUN_DIR="${RUN_DIR:?set RUN_DIR to the training run dir with checkpoint-N/}"
IF_GRAD="${IF_GRAD:-rollout}"          # rollout = detector | gold = control
IF_METHOD="${IF_METHOD:-tracin-adam}"  # dot | tracin-adam | cg
CHECKPOINTS="${CHECKPOINTS:-all}"      # 'all' or e.g. "100" for a fast single-checkpoint P1
G="${G:-8}"
SCORE_BATCH="${SCORE_BATCH:-8}"
COSINE="${COSINE:-0}"                   # 1 = rank by cosine (direction) not raw dot (magnitude)
OUT="${OUT:-}"                          # override output dir (use a separate one for cosine!)
EXTRA="${EXTRA:-}"
COSINE_FLAG=""; [ "$COSINE" = "1" ] && COSINE_FLAG="--cosine"
OUT_FLAG="";    [ -n "$OUT" ] && OUT_FLAG="--out $OUT"

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NGPU="${NGPU:-1}"
echo "============ P2 SCORE: $IF_GRAD/$IF_METHOD over $RUN_DIR (ckpts=$CHECKPOINTS, cosine=$COSINE, ${NGPU} GPU) ============"

# torchrun shards the pool+targets across the allocated GPUs (all_reduce assembles);
# NGPU=1 degrades to a plain single-process run.
torchrun --nproc_per_node="$NGPU" \
    -m experiments.protocol2.score_influence \
    --run-dir "$RUN_DIR" --if-grad "$IF_GRAD" --if-method "$IF_METHOD" \
    --checkpoints "$CHECKPOINTS" --g "$G" --score-batch "$SCORE_BATCH" $COSINE_FLAG $OUT_FLAG $EXTRA

echo "report + scores under ${OUT:-$RUN_DIR/influence}"
