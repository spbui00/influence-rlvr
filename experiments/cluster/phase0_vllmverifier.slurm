#!/bin/bash
# ============================================================================
# PHASE-0 SIGNAL CHECK with a vLLM-SERVED VERIFIER (2 GPUs).
#
# Same gate as phase0_signalcheck.slurm, but grades with vLLM instead of the
# in-process HF verifier. Why this layout:
#   - The general-verifier is a FIXED model (never updated during training), so
#     serving it is trivial — `vllm serve` on its own GPU, NO weight sync.
#   - vLLM continuous batching grades each rollout as it finishes instead of the
#     HF path where the whole 64-completion batch stalls on the slowest decision
#     -> ~10-20x faster grading.
#   - Because it's a separate process, it dodges the one-vLLM-engine-per-process
#     limit (the policy keeps its own colocate engine on GPU0).
#   - Fast grading lets us run --verifier-max-new-tokens 1024 (the card's value)
#     so the verifier's reasoning isn't truncated before "Final Decision:" — 512
#     was silently mis-grading the long-tail verifications as wrong.
#
# Layout:  GPU0 = policy (colocate vLLM) + training   |   GPU1 = verifier vLLM server
#
#   sbatch experiments/cluster/phase0_vllmverifier.slurm
#   STEPS=200 sbatch experiments/cluster/phase0_vllmverifier.slurm
#
#SBATCH --job-name=ifrlvr_p0v
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-p0_vllmverifier}"
SEED="${SEED:-42}"
STEPS="${STEPS:-400}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VPORT="${VPORT:-8100}"
VERIFIER_GMU="${VERIFIER_GMU:-0.85}"      # verifier server's share of GPU 1

# ── Launch the verifier vLLM server on GPU 1 ────────────────────────────────
echo "Launching verifier vLLM server (GPU1, model=$VERIFIER, port=$VPORT)..."
CUDA_VISIBLE_DEVICES=1 vllm serve "$VERIFIER" \
    --port "$VPORT" \
    --gpu-memory-utilization "$VERIFIER_GMU" \
    --max-model-len 4096 \
    --disable-log-requests \
    > "vllm-verifier-${SLURM_JOB_ID}.log" 2>&1 &
VSERVER_PID=$!
trap 'echo "stopping verifier server $VSERVER_PID"; kill $VSERVER_PID 2>/dev/null || true' EXIT

# ── Wait until the server answers /v1/models (max ~10 min) ──────────────────
echo "Waiting for verifier server on 127.0.0.1:$VPORT ..."
python - "$VPORT" <<'PY'
import sys, time, urllib.request
port = int(sys.argv[1])
deadline = time.time() + 600
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            if r.status == 200:
                print("verifier server is up."); sys.exit(0)
    except Exception:
        time.sleep(4)
print("ERROR: verifier server did not come up in 600s", file=sys.stderr); sys.exit(1)
PY

# Effective batch = 8x8 = 64 -> 8 prompts/step (the signal fix). Policy on GPU0
# with its own colocate vLLM engine; grading goes to the verifier server (GPU1).
COMMON="--model-id ${MODEL} \
    --if-target-source mmlu_pro_cs --eval-benchmarks mmlu_pro_cs,gsm8k \
    --live-eval-benchmark mmlu_pro_cs --n-train-pool 96 --n-if-target 32 \
    --max-steps ${STEPS} --save-steps 50 \
    --g-train 8 --per-device-batch 8 --grad-accum 8 --max-completion-length 2048 \
    --use-vllm \
    --verifier-backend vllm --verifier-server-port ${VPORT} --verifier-max-new-tokens 1024 \
    --live-eval-examples 128 --eval-max-examples 200"

echo "================ PHASE-0 (vLLM verifier) — TRAIN ($RUN_NAME) ================"
# No srun: manual CUDA_VISIBLE_DEVICES splits GPU0 (policy) from GPU1 (verifier
# server); letting srun re-bind GPUs would break that split (cf. train_server.slurm).
CUDA_VISIBLE_DEVICES=0 python -m experiments.train --run-name "$RUN_NAME" --seed "$SEED" \
    $COMMON --regime baseline

echo "================ EVAL ($RUN_NAME) ================"
CUDA_VISIBLE_DEVICES=0 python -m experiments.evaluate --run-name "$RUN_NAME" \
    --checkpoint-step latest --benchmarks mmlu_pro_cs,gsm8k --eval-max-examples 200 --vllm

echo "================ DONE ================"
echo "reward curve: outputs/$RUN_NAME/live_eval.csv ; final: outputs/$RUN_NAME/eval/"
