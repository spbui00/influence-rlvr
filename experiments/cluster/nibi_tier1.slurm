#!/bin/bash
# ============================================================================
# TIER-1 BOOTSTRAP — does GR-SCALE training make Qwen3-4B-Base actually learn?
#
# The 1-2 GPU signal check stayed flat because our batch was ~128x smaller than
# General-Reasoner's (Table 9: train_batch_size 1024). This run closes most of
# that gap with multi-GPU DATA-PARALLEL training on ONE Nibi H100 node, and is
# the GO/NO-GO gate for the whole scale-up.
#
# Layout (1 node, 8x H100 80GB):
#   GPU 0..6 : 7 data-parallel TRL workers (LoRA, each with a COLOCATE vLLM engine
#              -> generation parallelized across all 7). baseline => no influence
#              windows => colocate is fine (one engine per process).
#   GPU 7    : verifier vLLM server (fixed model, no weight sync).
#
# Batch: 7 workers x grad_accum x (per_device_batch/g_train) = 7 x 14 x (16/8=2)
#        = 196 prompts/step  (~1/5 of GR's 1024; ~24x our old 8).
# GR HPs: response 4096, temp 0.7, clip-higher 0.3, LoRA r32, lr 1e-5 (LoRA).
#
#   sbatch experiments/cluster/nibi_tier1.slurm
#   STEPS=400 GRAD_ACCUM=22 sbatch experiments/cluster/nibi_tier1.slurm   # bigger
#
# GATE (printed at the end): box_rate & acc climb in the [reward] log AND the
# held-out eval moves >=15 pts. If NO here, the testbed is wrong (not the infra).
#
#SBATCH --job-name=nibi_t1
#SBATCH --account=def-rgrosse
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-nibi_t1_baseline}"
SEED="${SEED:-42}"
STEPS="${STEPS:-300}"
# RESUME=1 continues from the latest checkpoint under outputs/$RUN_NAME (same
# RUN_NAME) — use it to recover a time-limit kill, or to extend max-steps later.
RESUME_ARG=""; [ "${RESUME:-0}" = "1" ] && RESUME_ARG="--resume"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VPORT="${VPORT:-8100}"
N_TRAIN_GPU="${N_TRAIN_GPU:-7}"          # GPUs 0..N-1 train; the last GPU runs the verifier
GRAD_ACCUM="${GRAD_ACCUM:-14}"           # 7 x GRAD_ACCUM x 2 = prompts/step (14 -> 196)
POOL="${POOL:-1500}"
export IFRLVR_DEBUG_SAMPLES="${IFRLVR_DEBUG_SAMPLES:-2}"

# ── Verifier vLLM server on the LAST GPU ────────────────────────────────────
VGPU=$N_TRAIN_GPU
echo "Launching verifier vLLM server on GPU $VGPU (model=$VERIFIER, port=$VPORT)..."
CUDA_VISIBLE_DEVICES=$VGPU vllm serve "$VERIFIER" \
    --port "$VPORT" --gpu-memory-utilization 0.85 --max-model-len 4096 \
    > "vllm-verifier-${SLURM_JOB_ID}.log" 2>&1 &
VSERVER_PID=$!
trap 'echo "stopping verifier server $VSERVER_PID"; kill $VSERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for verifier server on 127.0.0.1:$VPORT ..."
python - "$VPORT" <<'PY'
import sys, time, urllib.request
port = int(sys.argv[1]); deadline = time.time() + 600
while time.time() < deadline:
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3).status == 200:
            print("verifier server is up."); sys.exit(0)
    except Exception:
        time.sleep(4)
print("ERROR: verifier server did not come up in 600s", file=sys.stderr); sys.exit(1)
PY

# ── Data-parallel training on GPUs 0..N-1 (colocate vLLM per worker) ─────────
TRAIN_GPUS=$(seq -s, 0 $((N_TRAIN_GPU - 1)))   # e.g. 0,1,2,3,4,5,6
echo "================ TIER-1 DP TRAIN ($RUN_NAME) — GPUs $TRAIN_GPUS, grad_accum=$GRAD_ACCUM ================"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --config_file experiments/cluster/accelerate_ddp.yaml \
    --num_processes "$N_TRAIN_GPU" \
    -m experiments.train \
    --run-name "$RUN_NAME" --seed "$SEED" --regime baseline $RESUME_ARG \
    --model-id "$MODEL" --lora-r 32 \
    --if-target-source mmlu_pro_cs --eval-benchmarks mmlu_pro_cs,gsm8k \
    --live-eval-benchmark mmlu_pro_cs --n-train-pool "$POOL" --n-if-target 64 \
    --max-steps "$STEPS" --save-steps 50 \
    --g-train 8 --per-device-batch 16 --grad-accum "$GRAD_ACCUM" \
    --max-completion-length 4096 --grpo-temperature 0.7 --grpo-epsilon-high 0.3 \
    --learning-rate 1e-5 \
    --use-vllm --vllm-mode colocate --vllm-gpu-memory-utilization 0.35 \
    --verifier-backend vllm --verifier-server-port "$VPORT" --verifier-max-new-tokens 1024 \
    --live-eval-examples 64 --eval-max-examples 200

# ── Final held-out eval on GPU 0 (grades via the still-running verifier server) ─
echo "================ EVAL ($RUN_NAME) ================"
CUDA_VISIBLE_DEVICES=0 python -m experiments.evaluate --run-name "$RUN_NAME" \
    --checkpoint-step latest --benchmarks mmlu_pro_cs,gsm8k --eval-max-examples 200 --vllm \
    --verifier-backend vllm --verifier-server-port "$VPORT" --verifier-max-new-tokens 1024

echo "================ GATE VERDICT ($RUN_NAME) ================"
python - "$RUN_NAME" << 'PY'
import csv, json, sys, pathlib
run = sys.argv[1]; root = pathlib.Path("outputs") / run
acc = []
le = root / "live_eval.csv"
if le.exists():
    for row in csv.DictReader(open(le)):
        acc.append((int(row["step"]), float(row["accuracy"])))
fe = sorted((root / "eval").glob("eval_step*.json"))
final = json.load(open(fe[-1]))["accuracy"] if fe else {}
print(f"held-out eval trajectory (MMLU-Pro-CS): {[f'{s}:{a:.3f}' for s,a in acc]}")
print(f"final eval (n=200): {final}")
if len(acc) >= 2:
    rise = acc[-1][1] - acc[0][1]
    print(f"  end-to-end rise: {rise:+.3f}  (gate >= +0.15 -> {'PASS' if rise>=0.15 else 'CHECK'})")
    print("  VERDICT:", "BOOTSTRAP WORKS — build Tier-2." if rise>=0.15
          else "WEAK — check box_rate/acc in the [reward] log; if also flat, it's the testbed.")
else:
    print("  not enough live-eval points; read the [reward] box_rate/acc trend in the log.")
PY
echo "================ DONE ================"
