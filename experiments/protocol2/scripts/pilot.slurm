#!/bin/bash
# Phase-1 pilot: base-model rollouts to BAND the percentage-flip datasets.
# G=8, temp=1.0, top_p=1.0, top_k=-1 (theory constraints §3 — and the pilot MUST
# sample at the training temperature or the band doesn't predict training).
# Single GPU, no verifier model (banding is pure numeric equivalence to gold).
# ~1.4k prompts x 8 rollouts of a 1.5B model: well under an hour on an L40S/H100.
#
# The Phase-0 JSONLs (experiments/protocol2/dataset/data/) are gitignored; this
# script REBUILDS them if missing (prepare_dataset is deterministic + seeded, so
# the cluster build is byte-identical to a local one at the same seed).
#
#   Clariden:   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#               sbatch --environment=pt --account=a0133 --partition=normal \
#                      --gres=gpu:1 --mem=100G --time=1:30:00 --cpus-per-task=16 \
#                      experiments/protocol2/scripts/pilot.slurm
#   Killarney:  sbatch --account=aip-rgrosse --gres=gpu:l40s:1 --mem=64G \
#                      --time=2:00:00 --cpus-per-task=8 \
#                      experiments/protocol2/scripts/pilot.slurm
#   Nibi:       sbatch --account=def-rgrosse --gres=gpu:h100:1 --mem=64G \
#                      --time=2:00:00 --cpus-per-task=8 \
#                      experiments/protocol2/scripts/pilot.slurm
#
#   LIMIT=32 sbatch ... pilot.slurm   # quick GPU smoke run first
#
#SBATCH --job-name=p2-pilot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
G="${G:-8}"
TEMP="${TEMP:-1.0}"                 # keep == training temperature
SEED="${SEED:-0}"
SETS="${SETS:-candidates_percent,target_percent,target_nonpercent}"
LIMIT="${LIMIT:-0}"                 # e.g. LIMIT=32 for a quick smoke run
MAXTOK="${MAXTOK:-512}"
EXTRA="${EXTRA:-}"                  # passthrough, e.g. EXTRA="--band-lo 0.25"

echo "================ P2 PILOT: model=$MODEL G=$G temp=$TEMP sets=$SETS ================"

# Phase-0 data is gitignored — rebuild if absent (deterministic, seed fixed).
DATA_DIR="experiments/protocol2/dataset/data"
if [ ! -f "$DATA_DIR/candidates_percent.jsonl" ]; then
    echo "Phase-0 data missing under $DATA_DIR — building..."
    python -m experiments.protocol2.dataset.prepare_dataset
fi

python -m experiments.protocol2.pilot \
    --model "$MODEL" --num-generations "$G" --temperature "$TEMP" \
    --seed "$SEED" --sets "$SETS" --limit "$LIMIT" --max-tokens "$MAXTOK" \
    $EXTRA
