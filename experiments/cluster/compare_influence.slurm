#!/bin/bash
# Gold vs rollout influence agreement on ONE checkpoint — with vLLM rollout generation.
# The gold pass needs no generation; the rollout pass generates G rollouts/prompt. Since
# there's no trainer to sync weights to a gen server, we MERGE the checkpoint adapter into
# the base and serve that copy via `trl vllm-serve` — so rollout gen runs on vLLM (~4x),
# while the gradient backward stays exact on the in-process HF model at the same checkpoint.
#
# GPU layout: compare/grads GPU 0, gen server (merged ckpt) GPU 1, verifier GPU 2 -> 3 GPUs.
#
#   RUN_NAME=math_if_v2 CKPT=10 TEST_DOMAINS=math COMPARE_POOL=4000 IF_G_TRAIN=4 \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:3 \
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
IF_SCORE_BATCH="${IF_SCORE_BATCH:-64}" # prompts whose rollouts are generated per /generate/ call (vLLM likes big)

VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VERIFIER_MAX_LEN="${VERIFIER_MAX_LEN:-4096}"
VPORT="${VPORT:-8100}"
GENPORT="${GENPORT:-51216}"
GEN_GPU="${GEN_GPU:-1}"
VERIFIER_GPU="${VERIFIER_GPU:-2}"      # compare/grads run on GPU 0

CKPT_DIR="outputs/$RUN_NAME/rlvr-output/checkpoint-$CKPT"
MERGED_DIR="${MERGED_DIR:-$SCRATCH/merged/${RUN_NAME}_step${CKPT}}"

echo "================ COMPARE gold vs rollout ($RUN_NAME / checkpoint-$CKPT) — vLLM rollout gen ================"
echo "  pool=$COMPARE_POOL target=$TEST_DOMAINS($N_IF_TARGET) method=$IF_METHOD G=$IF_G_TRAIN"

# ── 1) merge the checkpoint adapter into base so the gen server can hold the policy ─
if [ ! -f "$MERGED_DIR/config.json" ]; then
    echo "Merging $CKPT_DIR -> $MERGED_DIR ..."
    CUDA_VISIBLE_DEVICES=0 python -m experiments.merge_checkpoint \
        --model-id "$MODEL" --checkpoint "$CKPT_DIR" --out "$MERGED_DIR"
else
    echo "Reusing existing merged model at $MERGED_DIR"
fi

# ── 2) gen server (merged ckpt) + verifier server ──────────────────────────────────
echo "Launching gen vLLM server on GPU $GEN_GPU (merged ckpt, port $GENPORT)..."
CUDA_VISIBLE_DEVICES=$GEN_GPU trl vllm-serve \
    --model "$MERGED_DIR" --host 127.0.0.1 --port "$GENPORT" \
    --gpu_memory_utilization 0.85 --max_model_len 8192 \
    > "vllm-gen-${SLURM_JOB_ID}.log" 2>&1 &
GENSERVER_PID=$!
echo "Launching verifier vLLM server on GPU $VERIFIER_GPU (model=$VERIFIER, port=$VPORT)..."
CUDA_VISIBLE_DEVICES=$VERIFIER_GPU vllm serve "$VERIFIER" \
    --port "$VPORT" --gpu-memory-utilization 0.50 --max-model-len "$VERIFIER_MAX_LEN" \
    > "vllm-verifier-${SLURM_JOB_ID}.log" 2>&1 &
VSERVER_PID=$!
trap 'echo "stopping servers"; for p in $GENSERVER_PID $VSERVER_PID; do [ -n "$p" ] && { pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; }; done; true' EXIT

echo "Waiting for servers (gen :$GENPORT, verifier :$VPORT) ..."
python - "$VPORT" "$GENPORT" <<'PY'
import socket, sys, time, urllib.request
vport, gport = int(sys.argv[1]), int(sys.argv[2])
deadline = time.time() + 900
def verifier_up():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{vport}/v1/models", timeout=3).status == 200
    except Exception:
        return False
def gen_up():
    try:
        with socket.create_connection(("127.0.0.1", gport), timeout=3):
            return True
    except OSError:
        return False
while time.time() < deadline:
    if verifier_up() and gen_up():
        print("servers up."); sys.exit(0)
    time.sleep(5)
print("ERROR: servers did not come up in 900s", file=sys.stderr); sys.exit(1)
PY

# ── 3) compare on GPU 0: gold (no gen) + rollout (gen offloaded to server, grads HF) ─
CUDA_VISIBLE_DEVICES=0 python -m experiments.compare_influence \
    --checkpoint-step "$CKPT" \
    --run-name "$RUN_NAME" --model-id "$MODEL" --seed "$SEED" --lora-r 32 \
    --domains "$DOMAINS" --n-train-pool "$COMPARE_POOL" \
    --test-from-train --test-from-train-eval "$TEST_EVAL" \
    --webinstruct-test-domains "$TEST_DOMAINS" --n-if-target "$N_IF_TARGET" \
    --if-method "$IF_METHOD" --if-grad gold --if-g-train "$IF_G_TRAIN" \
    --if-max-new-tokens 1024 --if-score-batch "$IF_SCORE_BATCH" --if-logps-micro-batch 1 \
    --use-vllm --vllm-mode server --vllm-server-host 127.0.0.1 --vllm-server-port "$GENPORT" --if-vllm-gen \
    --verifier-backend vllm --verifier-server-port "$VPORT" \
    --verifier-max-new-tokens 1024 --verifier-max-model-len "$VERIFIER_MAX_LEN"
