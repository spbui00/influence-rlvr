#!/bin/bash
# LDS retraining sweep (resumable). Measures a target metric per random subset so any
# influence method's scores can be LDS-scored later (experiments/lds.py score).
#
#   nll  variant: 1 GPU, no servers (teacher-forced gold-NLL SFT -> target gold-NLL).
#   reward variant: GPUs 0..N_TRAIN_GPU-1 DDP train (default 2) + policy-gen vLLM +
#                   verifier vLLM on the next two GPUs -> --gres=gpu:$((N_TRAIN_GPU+2)).
#
# RESUMABLE: each subset appends one line to outputs/<run>/lds/measured_<variant>.jsonl;
# re-submitting the identical line skips finished subsets. Safe to run across many days.
#
#   VARIANT=nll RUN_NAME=xdomain_phys_gold_c REF_STEP=10 M=200 ALPHA=0.25 \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#            --mem=200G --time=12:00:00 --cpus-per-task=24 experiments/cluster/lds.slurm
#
#   VARIANT=reward RUN_NAME=xdomain_phys_gold_c REF_STEP=10 TRAIN_STEPS=60 \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:4 \
#            --mem=200G --time=12:00:00 --cpus-per-task=24 experiments/cluster/lds.slurm
#
# SUBSETS_TAG selects an alternative manifest (subsets_<tag>.npy) so new subset designs
# don't touch the alpha-0.25 sweep already measured. Small-alpha sweep + extremes probe:
#
#   MAKE_SUBSETS=1 M=200 ALPHA=0.05 SUBSETS_TAG=a05 EPOCHS=8 \
#   VARIANT=nll RUN_NAME=xdomain_phys_gold_c REF_STEP=10 TAG=a05_s0 SEED=0 ... lds.slurm
#
#   # build the extremes manifest first (numpy-only, login node is fine):
#   #   python -m experiments.lds --run-name <run> make-extremes --scores <scores.npy>
#   VARIANT=nll SUBSETS_TAG=extremes RUN_NAME=xdomain_phys_gold_c REF_STEP=10 \
#   TAG=extremes_s0 SEED=0 ... lds.slurm
#
#SBATCH --job-name=ifrlvr-lds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

VARIANT="${VARIANT:?set VARIANT=nll|reward}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
REF_STEP="${REF_STEP:-10}"
EPOCHS="${EPOCHS:-3}"                     # nll: full-coverage passes over each subset (matched test)
TRAIN_STEPS="${TRAIN_STEPS:-}"           # override: N sampled steps (nll diag) / GRPO steps (reward=60)
TAG="${TAG:-}"                           # measured_<variant>_<tag>.jsonl (separate training specs)
SUBSETS_TAG="${SUBSETS_TAG:-}"           # subsets_<tag>.npy manifest (default: subsets.npy)
LIMIT="${LIMIT:-}"                       # cap subsets processed this submission (optional)
SEED="${SEED:-0}"
# make-subsets (run once; skipped if the manifest already exists unless MAKE_SUBSETS=1)
M="${M:-200}"; ALPHA="${ALPHA:-0.25}"; SUBSET_SEED="${SUBSET_SEED:-0}"
MAKE_SUBSETS="${MAKE_SUBSETS:-0}"
# reward-variant servers
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VPORT="${VPORT:-8100}"; VERIFIER_MAX_LEN="${VERIFIER_MAX_LEN:-4096}"
GENPORT="${GENPORT:-51216}"; GROUPPORT="${GROUPPORT:-$((GENPORT + 1))}"

STEPS_ARG=""; [ -n "$TRAIN_STEPS" ] && STEPS_ARG="--train-steps $TRAIN_STEPS"
TAG_ARG="";    [ -n "$TAG" ] && TAG_ARG="--tag $TAG"
SUBTAG_ARG=""; [ -n "$SUBSETS_TAG" ] && SUBTAG_ARG="--subsets-tag $SUBSETS_TAG"
LIMIT_ARG="";  [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

# ── subset manifest (deterministic; shared by all variants + methods) ───────────
SUBSETS_FILE="outputs/$RUN_NAME/lds/subsets${SUBSETS_TAG:+_$SUBSETS_TAG}.npy"
if [ "$MAKE_SUBSETS" = "1" ] || [ ! -f "$SUBSETS_FILE" ]; then
    echo "Building subset manifest (M=$M alpha=$ALPHA seed=$SUBSET_SEED tag=${SUBSETS_TAG:-default})..."
    python -m experiments.lds --run-name "$RUN_NAME" make-subsets -M "$M" --alpha "$ALPHA" --seed "$SUBSET_SEED" $SUBTAG_ARG
fi

echo "GPU state at launch:"; nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || true

if [ "$VARIANT" = "nll" ]; then
    # ── cheap path: single GPU, no servers ──────────────────────────────────────
    echo "================ LDS sweep nll ($RUN_NAME, ref=$REF_STEP) ================"
    CUDA_VISIBLE_DEVICES=0 python -m experiments.lds --run-name "$RUN_NAME" \
        run --variant nll --ref-step "$REF_STEP" --epochs "$EPOCHS" --seed "$SEED" \
        $STEPS_ARG $TAG_ARG $SUBTAG_ARG $LIMIT_ARG
    echo "================ DONE (nll) ================"
    exit 0
fi

# ── reward path: GPUs 0..N_TRAIN_GPU-1 = DDP train, then policy-gen vLLM, then verifier ──
N_TRAIN_GPU="${N_TRAIN_GPU:-2}"
GEN_GPU=$N_TRAIN_GPU
VERIFIER_GPU=$((N_TRAIN_GPU + 1))
NGPU_ALLOC=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NGPU_ALLOC:-0}" -lt $((N_TRAIN_GPU + 2)) ]; then
    echo "ERROR: reward variant needs N_TRAIN_GPU+2 GPUs (train+gen+verifier), got $NGPU_ALLOC" \
         "(N_TRAIN_GPU=$N_TRAIN_GPU). Use --gres=gpu:$((N_TRAIN_GPU + 2))."
    exit 1
fi

echo "Launching verifier vLLM server on GPU $VERIFIER_GPU (model=$VERIFIER, port=$VPORT)..."
CUDA_VISIBLE_DEVICES=$VERIFIER_GPU vllm serve "$VERIFIER" \
    --port "$VPORT" --gpu-memory-utilization 0.50 --max-model-len "$VERIFIER_MAX_LEN" \
    > "vllm-verifier-${SLURM_JOB_ID}.log" 2>&1 &
VSERVER_PID=$!
echo "Launching policy-gen vLLM server on GPU $GEN_GPU (trl vllm-serve, port $GENPORT)..."
CUDA_VISIBLE_DEVICES=$GEN_GPU trl vllm-serve \
    --model "$MODEL" --host 127.0.0.1 --port "$GENPORT" \
    --gpu_memory_utilization 0.85 --max_model_len 8192 \
    > "vllm-gen-${SLURM_JOB_ID}.log" 2>&1 &
GENSERVER_PID=$!
trap 'echo "stopping vLLM servers"; for p in $VSERVER_PID $GENSERVER_PID; do [ -n "$p" ] && { pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; }; done; true' EXIT

echo "Waiting for vLLM server(s) (verifier :$VPORT, gen :$GENPORT) ..."
python - "$VPORT" "$GENPORT" <<'PY'
import socket, sys, time, urllib.request
vport, gport = int(sys.argv[1]), int(sys.argv[2])
deadline = time.time() + 900
def up():
    try:
        ok = urllib.request.urlopen(f"http://127.0.0.1:{vport}/v1/models", timeout=3).status == 200
    except Exception:
        return False
    try:
        with socket.create_connection(("127.0.0.1", gport), timeout=3):
            return ok
    except OSError:
        return False
while time.time() < deadline:
    if up():
        print("vLLM server(s) up."); sys.exit(0)
    time.sleep(5)
print("ERROR: vLLM server(s) did not come up in 900s", file=sys.stderr); sys.exit(1)
PY

TRAIN_DEVS=$(seq 0 $((N_TRAIN_GPU - 1)) | paste -sd, -)
if [ "$N_TRAIN_GPU" -gt 1 ]; then
    LAUNCH="accelerate launch --config_file experiments/cluster/accelerate_ddp.yaml --num_processes $N_TRAIN_GPU -m"
else
    LAUNCH="python -m"
fi
echo "================ LDS sweep reward ($RUN_NAME, ref=$REF_STEP, train_gpus=$N_TRAIN_GPU) ================"
CUDA_VISIBLE_DEVICES=$TRAIN_DEVS $LAUNCH experiments.lds --run-name "$RUN_NAME" \
    run --variant reward --ref-step "$REF_STEP" --seed "$SEED" $STEPS_ARG $TAG_ARG $SUBTAG_ARG $LIMIT_ARG
echo "================ DONE (reward) ================"
