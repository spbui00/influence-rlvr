#!/bin/bash
# Gold vs rollout influence agreement on ONE checkpoint — cheap pre-screen for whether
# the ~10-30x-pricier rollout influence is worth running. Scores a small pool with both
# gradients against the same target and prints Spearman + top-K selection overlap.
#
# GPU layout: compare (model + HF rollout gen) on GPU 0, verifier server on GPU 1 -> 2 GPUs.
# Defaults to math_if_v2 / checkpoint-10; override RUN_NAME/CKPT/TEST_DOMAINS for others.
#
#   RUN_NAME=math_if_v2 CKPT=10 TEST_DOMAINS=math COMPARE_POOL=4000 IF_G_TRAIN=4 \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:2 \
#            --mem=200G --time=3:00:00 --cpus-per-task=24 experiments/cluster/compare_influence.slurm
#
#SBATCH --job-name=ifrlvr-cmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-math_if_v2}"
CKPT="${CKPT:-10}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
SEED="${SEED:-42}"
DOMAINS="${DOMAINS:-math,physics,finance}"
COMPARE_POOL="${COMPARE_POOL:-4000}"   # small pool to score (gold+rollout both score this)
TEST_DOMAINS="${TEST_DOMAINS:-math}"
N_IF_TARGET="${N_IF_TARGET:-256}"
TEST_EVAL="${TEST_EVAL:-1000}"
IF_METHOD="${IF_METHOD:-tracin-adam}"
IF_G_TRAIN="${IF_G_TRAIN:-4}"          # rollouts/prompt for the rollout gradient (4 is enough for ranking)
IF_SCORE_BATCH="${IF_SCORE_BATCH:-64}"

VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VERIFIER_MAX_LEN="${VERIFIER_MAX_LEN:-4096}"
VPORT="${VPORT:-8100}"
VERIFIER_GPU="${VERIFIER_GPU:-1}"      # compare runs on GPU 0

echo "================ COMPARE gold vs rollout ($RUN_NAME / checkpoint-$CKPT) ================"
echo "  pool=$COMPARE_POOL target=$TEST_DOMAINS($N_IF_TARGET) method=$IF_METHOD G=$IF_G_TRAIN"

# ── verifier vLLM server (rollout grading) ──────────────────────────────────────
echo "Launching verifier vLLM server on GPU $VERIFIER_GPU (model=$VERIFIER, port=$VPORT)..."
CUDA_VISIBLE_DEVICES=$VERIFIER_GPU vllm serve "$VERIFIER" \
    --port "$VPORT" --gpu-memory-utilization 0.50 --max-model-len "$VERIFIER_MAX_LEN" \
    > "vllm-verifier-${SLURM_JOB_ID}.log" 2>&1 &
VSERVER_PID=$!
trap 'echo "stopping verifier server"; pkill -P "$VSERVER_PID" 2>/dev/null; kill "$VSERVER_PID" 2>/dev/null; true' EXIT

echo "Waiting for verifier server (:$VPORT) ..."
python - "$VPORT" <<'PY'
import sys, time, urllib.request
vport = int(sys.argv[1]); deadline = time.time() + 900
while time.time() < deadline:
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{vport}/v1/models", timeout=3).status == 200:
            print("verifier up."); sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print("ERROR: verifier did not come up in 900s", file=sys.stderr); sys.exit(1)
PY

# ── compare on GPU 0 (gold = no gen; rollout = HF in-process gen, graded via server) ─
CUDA_VISIBLE_DEVICES=0 python -m experiments.compare_influence \
    --checkpoint-step "$CKPT" \
    --run-name "$RUN_NAME" --model-id "$MODEL" --seed "$SEED" --lora-r 32 \
    --domains "$DOMAINS" --n-train-pool "$COMPARE_POOL" \
    --test-from-train --test-from-train-eval "$TEST_EVAL" \
    --webinstruct-test-domains "$TEST_DOMAINS" --n-if-target "$N_IF_TARGET" \
    --if-method "$IF_METHOD" --if-grad gold --if-g-train "$IF_G_TRAIN" --no-use-vllm \
    --if-max-new-tokens 1024 --if-score-batch "$IF_SCORE_BATCH" --if-logps-micro-batch 1 \
    --verifier-backend vllm --verifier-server-port "$VPORT" \
    --verifier-max-new-tokens 1024 --verifier-max-model-len "$VERIFIER_MAX_LEN"
