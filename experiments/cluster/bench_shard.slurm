#!/bin/bash
# Multi-GPU CG-scoring benchmark: shards the pool/target work across all GPUs on
# the node via torchrun, near-linear speedup. Submit and walk away.
#
#   sbatch experiments/cluster/bench_shard.slurm                 # 4 H100, default args
#   GPUS=8 sbatch experiments/cluster/bench_shard.slurm          # (also edit the #SBATCH line)
#   BENCH_ARGS="--n-train-pool 96 ... --batch-sizes 8" sbatch ...
#
#SBATCH --job-name=bench_shard
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

# Production scoring settings; pool sized so the shards are balanced across GPUs.
BENCH_ARGS="${BENCH_ARGS:---n-train-pool 96 --n-if-target 16 --if-g-train 8 \
    --if-max-new-tokens 1024 --cg-fisher-examples 8 --cg-fisher-g 2 \
    --cg-fisher-max-tokens 128 --cg-iters 50 --if-logps-micro-batch 2 \
    --batch-sizes 8}"

echo "================ SHARDED CG BENCHMARK (torchrun, all GPUs) ================"
echo "args: $BENCH_ARGS"
# --nproc_per_node=gpu launches one process per visible GPU; --standalone = single node.
torchrun --standalone --nproc_per_node=gpu -m scripts.bench_cg_batch $BENCH_ARGS
echo "================ DONE ================"
