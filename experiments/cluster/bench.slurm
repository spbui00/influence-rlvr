#!/bin/bash
#
#   sbatch experiments/cluster/bench.slurm
#   # watch later:  tail -f slurm-bench_cg-*.out
#
# Override the GPU type / benchmark args via env vars at submit time, e.g.:
#   GPU=l40s sbatch experiments/cluster/bench.slurm
#   BENCH_ARGS="--if-g-train 4 --if-max-new-tokens 512 --batch-sizes 1,4,8" sbatch ...
#
#SBATCH --job-name=bench_cg
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

# Real production settings by default; override with BENCH_ARGS.
BENCH_ARGS="${BENCH_ARGS:---n-train-pool 24 --n-if-target 8 --if-g-train 8 \
    --if-max-new-tokens 1024 --cg-fisher-examples 4 --cg-fisher-g 2 \
    --cg-fisher-max-tokens 128 --cg-iters 10 --if-logps-micro-batch 4 \
    --batch-sizes 1,8,16}"

echo "================ CG BATCH BENCHMARK ================"
echo "args: $BENCH_ARGS"
srun python -m scripts.bench_cg_batch $BENCH_ARGS
echo "================ DONE ================"
