#!/bin/bash
# P1 Stage-3, NODE-PACKED for Clariden (4x GH200/node): one array task = one node
# = 4 subset runs in parallel, one per GPU (CUDA_VISIBLE_DEVICES pinning). Array
# task g handles subsets 4g .. 4g+3, so --array=0-24 covers the M=100 manifest
# from one checkpoint; submit once per checkpoint:
#
#   for C in 100 200; do
#     REF_STEP=$C REF_DIR=$SCRATCH/p1_runs/p1_ref RUNS_ROOT=$SCRATCH/p1_runs \
#     VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#     sbatch --environment=pt --account=a0133 --partition=normal \
#            --gres=gpu:4 --mem=440G --time=8:00:00 --cpus-per-task=144 \
#            --array=0-24 experiments/protocol1/scripts/subsets_node.slurm
#   done
#
# Node-job wall = the slowest of its 4 runs (measured ~72 s/step -> 200 steps
# ≈ 4 h + evals ≈ 4.5-5 h; --time=8h leaves margin). 25 node-jobs x 2 ckpts;
# at 4 concurrent nodes ≈ 2.6 days, at 6 ≈ 1.8. Fully idempotent: complete
# subsets are skipped, so resubmitting the same arrays after any preemption just
# continues. Per-subset logs land in $RUNS_ROOT/step<C>/subset_<m>.log.
#
#   Smoke one node first:  CONT_STEPS=4 EVAL_AT=2 EVAL_SAMPLES=2 sbatch ... --array=0
#
#SBATCH --job-name=p1-subs-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

REF_DIR="${REF_DIR:?set REF_DIR to the pi_ref run dir}"
REF_STEP="${REF_STEP:-200}"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT for the subset_<m> run dirs}"
CONT_STEPS="${CONT_STEPS:-200}"
EVAL_AT="${EVAL_AT:-1,50,100,150,200}"
SAVE_STEPS="${SAVE_STEPS:-0}"
SEED="${SEED:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
M_MAX="${M_MAX:-100}"                  # manifest size — guards the last group
EXTRA="${EXTRA:-}"

GROUP="${SLURM_ARRAY_TASK_ID:?run via sbatch --array}"
BASE=$((GROUP * GPUS_PER_NODE))
INIT="$REF_DIR/checkpoint-$REF_STEP"
[ -d "$INIT" ] || { echo "ERROR: ref checkpoint $INIT not found"; exit 1; }
mkdir -p "$RUNS_ROOT/step$REF_STEP"

# ── thundering-herd guards (50 nodes x 4 procs = 200 simultaneous startups took
#    out 32/50 nodes at ~90 s on the first full submission) ──────────────────────
# The HF cache is fully warm (ref run + smoke used the same model): skip ALL hub
# freshness checks — 200 concurrent etag pings from one cluster draw 429s.
export HF_HUB_OFFLINE=1
# Spread node starts over ~2 min so the shared FS never sees them all at once.
STAGGER=$(( (GROUP % 8) * 15 ))
[ "$STAGGER" -gt 0 ] && { echo "stagger: sleeping ${STAGGER}s"; sleep "$STAGGER"; }

complete() {  # subset fully measured = final horizon present
    [ -f "$1/target_eval.json" ] && python -c "
import json, sys
d = json.load(open('$1/target_eval.json'))
sys.exit(0 if '$CONT_STEPS' in d.get('horizons', {}) else 1)"
}

echo "===== P1 NODE GROUP $GROUP @ step $REF_STEP: subsets $BASE..$((BASE + GPUS_PER_NODE - 1)) ====="
pids=(); labels=()
for i in $(seq 0 $((GPUS_PER_NODE - 1))); do
    M=$((BASE + i))
    [ "$M" -ge "$M_MAX" ] && continue
    OUT="$RUNS_ROOT/step$REF_STEP/subset_$M"
    if complete "$OUT"; then
        echo "subset $M already measured — skip"
        continue
    fi
    echo "GPU $i -> subset $M (log: $OUT.log)"
    # Per-process compile caches on job-local TMPDIR: the default shared
    # $HOME/.cache vLLM/inductor/triton dirs are filelock'd — 200 concurrent
    # lockers time out. Costs one ~1-2 min recompile per process, amortized
    # over a ~4.5 h run. Launches staggered 20 s apart for the same reason.
    CACHE="$TMPDIR/p1_${SLURM_JOB_ID:-$$}_proc$i"; mkdir -p "$CACHE"   # job-unique:
    # TMPDIR may fall back to a SHARED path when SLURM_TMPDIR is unset
    [ "$i" -gt 0 ] && sleep 20
    # Distinct MASTER_PORT per process: TRL/accelerate inits a world-size-1 torch
    # distributed group and all four would otherwise race for the default 29500 —
    # exactly one wins, three crash (bit the first node smoke).
    CUDA_VISIBLE_DEVICES=$i MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + i)) \
    VLLM_CACHE_ROOT="$CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
    TRITON_CACHE_DIR="$CACHE/triton" XDG_CACHE_HOME="$CACHE/xdg" \
    python -m experiments.protocol1.train \
        --run-name "p1_step${REF_STEP}_subset_$M" --output-dir "$OUT" \
        --subset-id "$M" --init-adapter "$INIT" \
        --max-steps "$CONT_STEPS" --save-steps "$SAVE_STEPS" --seed "$SEED" \
        --eval-at "$EVAL_AT" --eval-samples "$EVAL_SAMPLES" $EXTRA \
        > "$OUT.log" 2>&1 &
    pids+=($!); labels+=("$M")
done

[ "${#pids[@]}" -eq 0 ] && { echo "nothing to do."; exit 0; }
fail=0
for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then
        echo "subset ${labels[$j]} DONE"
    else
        echo "subset ${labels[$j]} FAILED (see subset_${labels[$j]}.log)"; fail=1
    fi
done
exit $fail
