#!/bin/bash
# Headroom probe: pass@1 vs pass@k for the BASE model on the held-out target.
# Decides whether the flat held-out curve is a scale problem (pass@k >> pass@1) or a
# ceiling (pass@k ~= pass@1) BEFORE spending a big run. Single process + one verifier.
#
# GPU layout: probe's own vLLM engine on GPU 0, verifier server on GPU 1 → needs 2.
# Data flags DEFAULT to the gold run's pool config so the held-out slice is identical;
# override any knob from the env.
#
#   RUN_NAME=fin_headroom MODEL=Qwen/Qwen3-1.7B-Base K=8 TEMP=1.0 LIMIT=256 \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:2 \
#            --mem=200G --time=1:00:00 --cpus-per-task=24 experiments/cluster/headroom.slurm
#
#SBATCH --job-name=ifrlvr-headroom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-fin_headroom}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
SEED="${SEED:-42}"
K="${K:-8}"                         # samples per prompt
TEMP="${TEMP:-1.0}"                 # sampling temperature for the probe
LIMIT="${LIMIT:-256}"              # held-out prompts to probe

# Pool/target config — MUST match the train run so the held-out slice is identical.
DOMAINS="${DOMAINS:-math,physics,finance}"
POOL="${POOL:-24000}"
TEST_DOMAINS="${TEST_DOMAINS:-finance}"
N_IF_TARGET="${N_IF_TARGET:-256}"
TEST_EVAL="${TEST_EVAL:-1000}"      # held-out eval partition size (matches train.slurm)
TARGET_DIFFICULTY="${TARGET_DIFFICULTY:-}"   # e.g. University — probe headroom on uni-level target
DIFF_ARG=""; [ -n "$TARGET_DIFFICULTY" ] && DIFF_ARG="--target-difficulty $TARGET_DIFFICULTY"

VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VERIFIER_MAX_LEN="${VERIFIER_MAX_LEN:-4096}"
VPORT="${VPORT:-8100}"
VERIFIER_GPU="${VERIFIER_GPU:-1}"   # probe runs on GPU 0

echo "================ HEADROOM PROBE ($RUN_NAME) — probe GPU 0, verifier GPU $VERIFIER_GPU ================"
echo "  model=$MODEL k=$K temp=$TEMP limit=$LIMIT | pool=$POOL domains=$DOMAINS test=$TEST_DOMAINS"

# ── verifier vLLM server ────────────────────────────────────────────────────────
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

# ── probe on GPU 0 (its own standalone vLLM engine loads the base model) ─────────
CUDA_VISIBLE_DEVICES=0 python -m experiments.headroom \
    --k "$K" --headroom-temperature "$TEMP" --limit "$LIMIT" \
    --run-name "$RUN_NAME" --model-id "$MODEL" --seed "$SEED" \
    --test-from-train --test-from-train-eval "$TEST_EVAL" \
    --webinstruct-test-domains "$TEST_DOMAINS" --domains "$DOMAINS" $DIFF_ARG \
    --n-train-pool "$POOL" --n-if-target "$N_IF_TARGET" \
    --verifier-backend vllm --verifier-server-port "$VPORT" \
    --verifier-max-new-tokens 1024 --verifier-max-model-len "$VERIFIER_MAX_LEN"
