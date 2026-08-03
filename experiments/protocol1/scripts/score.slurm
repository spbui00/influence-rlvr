#!/bin/bash
# P1 Stage-2: influence scoring at ONE checkpoint (Protocol 1 has no trajectory
# sum). Saves the [T, N] per-target matrix lds.py correlates. Each
# IF_GRAD/IF_METHOD/COSINE combo writes its OWN subdir (step-suffixed filenames,
# so all REF_STEPs coexist) — run one job per (checkpoint, combo):
#
#   the 3-method comparison (dot vs tracin-adam vs ekfac, all on rollout grads —
#   the preconditioner ladder: identity / Adam diagonal / Kronecker inverse;
#   COSINE=1 is the DEFAULT and applies uniformly to all three):
#     for C in 100 200; do
#       REF_STEP=$C IF_GRAD=rollout IF_METHOD=dot         sbatch ... score.slurm
#       REF_STEP=$C IF_GRAD=rollout IF_METHOD=tracin-adam sbatch ... score.slurm
#       REF_STEP=$C IF_GRAD=rollout IF_METHOD=ekfac       sbatch ... score.slurm
#     done
#   (optional cheap extras: IF_GRAD=gold IF_METHOD=dot — no generation, ~15 min;
#    COSINE=0 IF_METHOD=dot — the raw magnitude-carrying dot ablation)
#
#   Clariden:  REF_DIR=$SCRATCH/p1_runs/p1_ref REF_STEP=200 \
#              VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#              sbatch --environment=pt --account=a0133 --partition=normal \
#                     --gres=gpu:1 --mem=200G --time=8:00:00 --cpus-per-task=32 \
#                     experiments/protocol1/scripts/score.slurm
#
# rollout mode generates G=8 rollouts/prompt with HF (slow half) — budget ~1-2 h
# per checkpoint for the 1000-pool (ekfac adds ~20-30 min of factor fitting);
# gold mode has no generation and is much faster.
#
#SBATCH --job-name=p1-score
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

REF_DIR="${REF_DIR:?set REF_DIR to the pi_ref run dir with checkpoint-N/}"
REF_STEP="${REF_STEP:-}"               # empty = latest checkpoint
IF_GRAD="${IF_GRAD:-rollout}"          # rollout | gold
IF_METHOD="${IF_METHOD:-tracin-adam}"  # dot | tracin-adam | ekfac | cg
COSINE="${COSINE:-1}"                  # DEFAULT 1: direction not magnitude, all methods
G="${G:-8}"
SCORE_BATCH="${SCORE_BATCH:-8}"
EXTRA="${EXTRA:-}"                     # e.g. "--fisher-examples 64 --ekfac-rel-damp 0.1"
STEP_FLAG="";   [ -n "$REF_STEP" ] && STEP_FLAG="--step $REF_STEP"
COSINE_FLAG="--cosine"; [ "$COSINE" = "0" ] && COSINE_FLAG="--no-cosine"

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NGPU="${NGPU:-1}"
echo "============ P1 SCORE: $IF_GRAD/$IF_METHOD cosine=$COSINE at $REF_DIR (step=${REF_STEP:-latest}, ${NGPU} GPU) ============"

# torchrun shards pool+targets across GPUs (all_reduce assembles); NGPU=1 = plain run.
# Job-unique port: shared-node clusters collide on the default 29500 (EADDRINUSE).
PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
torchrun --nproc_per_node="$NGPU" --master-addr 127.0.0.1 --master-port "$PORT" \
    -m experiments.protocol1.score_ref \
    --ref-dir "$REF_DIR" $STEP_FLAG --if-grad "$IF_GRAD" --if-method "$IF_METHOD" \
    --g "$G" --score-batch "$SCORE_BATCH" $COSINE_FLAG $EXTRA

echo "per-target matrix + scores under $REF_DIR/influence/"
