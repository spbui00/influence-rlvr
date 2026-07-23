#!/bin/bash
# Launch the CLEAN discovery run p2_clean_v1: assemble a 1000-background clean pool
# (0 poison, all honestly 'match'-graded — matches the backdoor's 1000-row size for
# comparability), then train 200 steps saving EVERY step (optimizer.pt included) so
# the full Protocol-2 trajectory is scorable. vLLM colocate (default). ~8h; 12h budget.
#
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/protocol2/scripts/run_clean_v1.slurm
#
#SBATCH --job-name=p2-clean-v1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

DATA_DIR="experiments/protocol2/dataset/data"
POOL="$DATA_DIR/pool_clean.jsonl"

# 1000-background clean pool (idempotent — skip if already built this size).
if [ ! -f "$POOL" ] || [ "$(grep -c . "$POOL")" -ne 1000 ]; then
    echo "assembling 1000-bg clean pool -> $POOL"
    python -m experiments.protocol2.assemble_pool --attack none --n-background 1000 --out "$POOL"
fi

RUN_NAME=p2_clean_v1 \
OUTPUT_DIR="$SCRATCH/p2_runs/p2_clean_v1" \
ATTACK=none POOL="$POOL" \
MAX_STEPS=200 SAVE_STEPS=1 \
bash experiments/protocol2/scripts/train.slurm
