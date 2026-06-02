#!/bin/bash
# Train with vLLM in SERVER mode: the engine runs as a separate `trl vllm-serve`
# process on GPU 1, training runs on GPU 0 and connects over HTTP. This lets
# if_prune use vLLM across windows (no in-process engine → no CuMem clash).
#
#   sbatch --export=ALL,REGIME=if_prune,RUN_NAME=qwen3_4b_prune experiments/cluster/train_server.slurm
#
#SBATCH --job-name=ifrlvr-srv
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:2          # GPU0=training+verifier+CG, GPU1=vLLM server
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REGIME="${REGIME:-if_prune}"
RUN_NAME="${RUN_NAME:-qwen3_4b_${REGIME}}"
SEED="${SEED:-42}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
PORT="${PORT:-8000}"
SERVER_GMU="${SERVER_GMU:-0.85}"        # vLLM server's share of GPU 1
EXTRA_ARGS="${EXTRA_ARGS:-}"

VENV_DIR="${VENV_DIR:-$HOME/envs/influence-rlvr}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"
export TMPDIR="${SLURM_TMPDIR:-$HOME/scratch/tmp}"; mkdir -p "$TMPDIR"
export WANDB_DIR="${WANDB_DIR:-$HOME/scratch/wandb}"; mkdir -p "$WANDB_DIR"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$HOME/scratch/wandb/cache}"; mkdir -p "$WANDB_CACHE_DIR"

# Do NOT module purge/reset on Killarney (strips /cm slurm modulefiles).
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 arrow/21.0.0 opencv/4.13.0
source "$VENV_DIR/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# ── Launch the vLLM server on GPU 1 ─────────────────────────────────────────
# NOTE: verify flag names with `trl vllm-serve --help` on this TRL version.
echo "Launching vLLM server (GPU 1, model=$MODEL_ID, port=$PORT, gmu=$SERVER_GMU)..."
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
    --model "$MODEL_ID" \
    --host 127.0.0.1 --port "$PORT" \
    --gpu_memory_utilization "$SERVER_GMU" \
    > "vllm-server-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping vLLM server $SERVER_PID"; kill $SERVER_PID 2>/dev/null || true' EXIT

# ── Wait until the server accepts connections (max ~5 min) ──────────────────
echo "Waiting for vLLM server on 127.0.0.1:$PORT ..."
python - "$PORT" <<'PY'
import socket, sys, time
port = int(sys.argv[1])
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            print("vLLM server is up."); sys.exit(0)
    except OSError:
        time.sleep(3)
print("ERROR: vLLM server did not come up in 300s", file=sys.stderr); sys.exit(1)
PY

# ── Train on GPU 0, talking to the server over HTTP ─────────────────────────
echo "Starting training (GPU 0) — regime=$REGIME run=$RUN_NAME"
CUDA_VISIBLE_DEVICES=0 python -m experiments.train \
    --regime "$REGIME" --run-name "$RUN_NAME" --seed "$SEED" \
    --use-vllm --vllm-mode server --vllm-server-port "$PORT" \
    --model-id "$MODEL_ID" \
    $EXTRA_ARGS
