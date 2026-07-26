#!/bin/bash
# Fix + run: `pip install vllm==0.23.0` downgraded transformers 5.12.1 -> 4.57.6,
# which breaks peft 0.19.1's ADAPTER-LOAD path (imports EmbeddingParallel, a
# transformers-5.x symbol) -> influence scoring crashes on checkpoint load though
# TRAINING was fine. The old working venv ran transformers 5.12.1 + vllm 0.23 +
# peft 0.19.1 together (trained AND scored the backdoor), so restore 5.12.1, then
# run gold-influence scoring of the CLEAN pool vs held-out correctness.
#
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:4 \
#          --mem=200G --time=2:00:00 --cpus-per-task=32 \
#          experiments/cluster/clariden_score_clean.sh
#
#SBATCH --job-name=p2-score-clean
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export VENV_DIR="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
source experiments/cluster/env.sh

echo "=== restore transformers 5.12.1 (peft-0.19 load path needs EmbeddingParallel) ==="
python -c "import transformers; print('  before:', transformers.__version__)"
pip install "transformers==5.12.1" 2>&1 | tail -4
python - <<'PY'
import transformers, peft
from transformers.integrations.tensor_parallel import EmbeddingParallel  # the symbol that was missing
print("  after :", transformers.__version__, "| peft", peft.__version__, "| EmbeddingParallel OK")
PY

echo "=== gold-influence scoring: CLEAN pool vs held-out correctness ==="
RUN_DIR="$SCRATCH/p2_runs/p2_clean_v1"
DATA="experiments/protocol2/dataset/data"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l)"; NGPU="${NGPU:-1}"
echo "  NGPU=$NGPU"
torchrun --nproc_per_node="$NGPU" \
    -m experiments.protocol2.score_influence \
    --run-dir "$RUN_DIR" --if-grad gold --if-method tracin-adam \
    --checkpoints 25,50,75,100,125,150,175,200 --cosine \
    --pool "$DATA/pool_clean.jsonl" --target "$DATA/target_clean.jsonl" \
    --out "$RUN_DIR/influence_gold_cos"

echo "=== outputs ==="
ls -la "$RUN_DIR/influence_gold_cos" 2>&1 | head -30
echo "=== clariden_score_clean done ==="
