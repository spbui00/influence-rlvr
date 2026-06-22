#!/bin/bash
# Cross-domain eval: grade each checkpoint on held-out math / physics / finance.
# Tests the generalization/forgetting story — does in-domain training (which wins on
# its OWN target) DEGRADE the other domains, while IF-pruning's cross-domain mix
# retains them? Each eval grades one checkpoint on all 3 held-out slices (per_category).
#
# vLLM gen on GPU 0 (per-arm engine), verifier server on GPU 1 -> 2 GPUs. Evals run
# sequentially; the verifier server stays up across all of them. Verifier settings come
# from each run's saved config.json (so we just launch the server on the matching port).
#
#   ARMS="phys_if_freq10:60 phys_indomain:60" BASE_RUN=phys_if_freq10 \
#   VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:2 \
#            --mem=200G --time=2:00:00 --cpus-per-task=24 experiments/cluster/cross_eval.slurm
#
#SBATCH --job-name=ifrlvr-xeval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

ARMS="${ARMS:-phys_if_freq10:60 phys_indomain:60}"   # space-separated run:step specs
BASE_RUN="${BASE_RUN:-}"                # if set, also eval the BASE model (no adapter) for reference
EVAL_DOMAINS="${EVAL_DOMAINS:-math,physics,finance}"
EVAL_MAX="${EVAL_MAX:-3000}"           # 3 domains x ~1000 held-out each
VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VERIFIER_MAX_LEN="${VERIFIER_MAX_LEN:-4096}"
VPORT="${VPORT:-8100}"                  # MUST match the runs' saved verifier_server_port
VERIFIER_GPU="${VERIFIER_GPU:-1}"

echo "================ CROSS-DOMAIN EVAL ================"
echo "  arms=[$ARMS] base=${BASE_RUN:-none} | eval domains=$EVAL_DOMAINS (max $EVAL_MAX)"

# ── verifier vLLM server ────────────────────────────────────────────────────────
echo "Launching verifier vLLM server on GPU $VERIFIER_GPU (model=$VERIFIER, port=$VPORT)..."
CUDA_VISIBLE_DEVICES=$VERIFIER_GPU vllm serve "$VERIFIER" \
    --port "$VPORT" --gpu-memory-utilization 0.50 --max-model-len "$VERIFIER_MAX_LEN" \
    > "vllm-verifier-${SLURM_JOB_ID}.log" 2>&1 &
VSERVER_PID=$!
trap 'echo "stopping verifier"; pkill -P "$VSERVER_PID" 2>/dev/null; kill "$VSERVER_PID" 2>/dev/null; true' EXIT

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

# ── BASE reference row (optional) ───────────────────────────────────────────────
if [ -n "$BASE_RUN" ]; then
    echo "---- BASE ($BASE_RUN, no adapter) on $EVAL_DOMAINS ----"
    CUDA_VISIBLE_DEVICES=0 python -u -m experiments.evaluate --run-name "$BASE_RUN" --base \
        --webinstruct-test-domains "$EVAL_DOMAINS" --benchmarks webinstruct_test \
        --eval-max-examples "$EVAL_MAX" --vllm
fi

# ── each arm @ its checkpoint ───────────────────────────────────────────────────
for spec in $ARMS; do
    run="${spec%:*}"; step="${spec#*:}"
    echo "---- $run @ checkpoint-$step on $EVAL_DOMAINS ----"
    CUDA_VISIBLE_DEVICES=0 python -u -m experiments.evaluate --run-name "$run" \
        --checkpoint-step "$step" --webinstruct-test-domains "$EVAL_DOMAINS" \
        --benchmarks webinstruct_test --eval-max-examples "$EVAL_MAX" --vllm
done

echo "================ DONE — per-domain accuracy in outputs/<run>/eval/eval_step<N>.json ================"
