#!/bin/bash
# Protocol-2 ROLLOUT influence with vLLM-offloaded generation (the spec-gaming detector).
# Protocol 2 = sum over checkpoints, so we score every STRIDE-th checkpoint. A static gen
# server can only hold ONE policy, so per checkpoint we MERGE its adapter into base, serve
# that via `trl vllm-serve`, score that single checkpoint (rollout gen -> server, gradient
# backward -> in-process HF, so the estimator stays exact), then tear the server down.
# aggregate_p2.py sums the per-checkpoint arrays into the trajectory total at the end.
# No verifier server: protocol2's reward is pure Python (numeric_eq / phrase check).
#
# GPU layout: gradient shards on GRAD_GPUS (torchrun), gen server on GEN_GPU.
#   2 GPUs: GRAD_GPUS=0     GEN_GPU=1     (--gres=gpu:2)
#   4 GPUs: GRAD_GPUS=0,1,2 GEN_GPU=3     (--gres=gpu:4, ~3x faster gradients)
#
#   RUN_DIR=$SCRATCH/p2_runs/p2_hack_v1 STRIDE=10 \
#   POOL=experiments/protocol2/dataset/data/pool_hack.jsonl \
#   TARGET_RULE=hack HACK_PHRASE=therefore GRAD_GPUS=0,1,2 GEN_GPU=3 \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:4 \
#            --mem=200G --time=6:00:00 --cpus-per-task=32 \
#            experiments/protocol2/scripts/score_influence_vllm.slurm
#
#   Calibrate first with STRIDE=50 (few checkpoints) to time one, then pick STRIDE.
#
#SBATCH --job-name=p2-score-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

RUN_DIR="${RUN_DIR:?set RUN_DIR to the training run dir with checkpoint-N/}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
STRIDE="${STRIDE:-10}"                 # score every K-th checkpoint (Protocol-2 Riemann sum)
IF_METHOD="${IF_METHOD:-tracin-adam}"
G="${G:-4}"                            # rollouts/prompt (4 is enough to rank)
SCORE_BATCH="${SCORE_BATCH:-64}"       # prompts per /generate/ call (vLLM likes big batches)
MAX_NEW="${MAX_NEW:-512}"
POOL="${POOL:-experiments/protocol2/dataset/data/pool_hack.jsonl}"
TARGET="${TARGET:-experiments/protocol2/dataset/data/target_clean.jsonl}"
TARGET_RULE="${TARGET_RULE:-hack}"     # hack = spec-gaming detector; match = correctness
HACK_PHRASE="${HACK_PHRASE:-therefore}"
GENPORT="${GENPORT:-51216}"
GEN_GPU="${GEN_GPU:-1}"
GRAD_GPUS="${GRAD_GPUS:-0}"            # comma list; torchrun shards the gradient over these
OUT="${OUT:-$RUN_DIR/influence_rollout_vllm}"
KEEP_MERGED="${KEEP_MERGED:-0}"
NGRAD=$(echo "$GRAD_GPUS" | tr ',' '\n' | grep -c .)

mkdir -p "$OUT"
echo "======== P2 ROLLOUT (vLLM gen) $RUN_DIR | stride=$STRIDE method=$IF_METHOD G=$G ========"
echo "  grads GPU(s) $GRAD_GPUS (x$NGRAD) | gen server GPU $GEN_GPU:$GENPORT | pool=$POOL target-rule=$TARGET_RULE"

# checkpoints to score: every STRIDE-th step (>0)
STEPS=$(python - "$RUN_DIR" "$STRIDE" <<'PY'
import sys, re
from pathlib import Path
run, k = Path(sys.argv[1]), int(sys.argv[2])
steps = sorted(int(m.group(1)) for p in run.glob("checkpoint-*")
               if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and int(m.group(1)) > 0)
print(" ".join(str(s) for s in steps[::k]))
PY
)
[ -n "$STEPS" ] || { echo "no checkpoint-N (N>0) under $RUN_DIR"; exit 1; }
echo "  checkpoints: $STEPS"

GENPID=""
cleanup() { [ -n "${GENPID:-}" ] && { kill "$GENPID" 2>/dev/null || true; pkill -P "$GENPID" 2>/dev/null || true; }; true; }
trap cleanup EXIT

for S in $STEPS; do
    echo "──── checkpoint-$S ────"
    MERGED="$SCRATCH/merged/$(basename "$RUN_DIR")_step${S}"
    if [ ! -f "$MERGED/config.json" ]; then
        echo "  merge adapter -> $MERGED"
        CUDA_VISIBLE_DEVICES="${GRAD_GPUS%%,*}" python -m experiments.merge_checkpoint \
            --model-id "$MODEL" --checkpoint "$RUN_DIR/checkpoint-$S" --out "$MERGED"
    fi
    echo "  serve merged ckpt on GPU $GEN_GPU (port $GENPORT)"
    CUDA_VISIBLE_DEVICES=$GEN_GPU trl vllm-serve --model "$MERGED" --host 127.0.0.1 --port "$GENPORT" \
        --gpu_memory_utilization 0.85 --max_model_len 4096 > "vllm-gen-${SLURM_JOB_ID}-s${S}.log" 2>&1 &
    GENPID=$!
    python - "$GENPORT" <<'PY'
import socket, sys, time
p = int(sys.argv[1]); dl = time.time() + 900
while time.time() < dl:
    try:
        with socket.create_connection(("127.0.0.1", p), timeout=3): print("  gen server up"); sys.exit(0)
    except OSError: time.sleep(5)
print("  gen server did not come up in 900s", file=sys.stderr); sys.exit(1)
PY
    if CUDA_VISIBLE_DEVICES=$GRAD_GPUS torchrun --nproc_per_node="$NGRAD" \
            -m experiments.protocol2.score_influence \
            --run-dir "$RUN_DIR" --pool "$POOL" --checkpoints "$S" \
            --if-grad rollout --if-method "$IF_METHOD" --g "$G" --score-batch "$SCORE_BATCH" \
            --max-new-tokens "$MAX_NEW" --target "$TARGET" --target-rule "$TARGET_RULE" \
            --hack-phrase "$HACK_PHRASE" --out "$OUT/ckpt-$S" \
            --use-vllm --if-vllm-gen --vllm-mode server \
            --vllm-server-host 127.0.0.1 --vllm-server-port "$GENPORT"; then
        echo "  checkpoint-$S scored -> $OUT/ckpt-$S"
    else
        echo "  checkpoint-$S FAILED — skipping (aggregation uses whatever succeeded)"
    fi
    cleanup; GENPID=""
    [ "$KEEP_MERGED" = "1" ] || rm -rf "$MERGED"
    sleep 5
done

echo "======== Protocol-2 aggregation (sum over scored checkpoints) ========"
python -m experiments.protocol2.aggregate_p2 --score-dir "$OUT" --pool "$POOL" --if-method "$IF_METHOD"
echo "P2 report + ranking under $OUT"
