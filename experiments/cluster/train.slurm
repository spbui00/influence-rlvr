#!/bin/bash
# ============================================================================
# GRPO trainer — runs BOTH the Tier-1 baseline and the Tier-2 IF-selection arms,
# so every arm is byte-identical to the baseline except --regime/--selection/
# --if-method (the only thing the comparison is allowed to vary).
#
#   REGIME=baseline  (default) → train on the full pool in random order. The Tier-1
#                     gate: does training move the held-out target at all?
#   REGIME=if_prune  → dynamic influence pruning: warm up to PRUNE_STEP, then every
#                     IF_RECOMPUTE_EVERY steps re-score the pool's influence on the IF
#                     target and train most-influential-first (SELECTION=if-guided),
#                     least-first (anti-if), or shuffled (random), under IF_METHOD
#                     (tracin-adam | dot | cg). Influence is scored on the train GPUs.
#
# Multi-GPU DATA-PARALLEL (accelerate DDP). GPU layout 0..N_TRAIN_GPU-1 = training,
# then the policy-gen vLLM server (USE_VLLM_GEN=1) and the verifier server. Every
# knob has a default (see below); resources come from the sbatch CLI.
#
#   # Tier-1 baseline (finance target carved from the train split, 1.7B):
#   RUN_NAME=fin USE_VLLM_GEN=1 MODEL=Qwen/Qwen3-1.7B-Base TEST_FROM_TRAIN=1 \
#     TEST_DOMAINS=finance N_IF_TARGET=256 POOL=3000 N_TRAIN_GPU=2 \
#     sbatch --account=def-zhijing --gres=gpu:h100:4 experiments/cluster/train.slurm
#   # Tier-2 arm — same config + the arm:
#   RUN_NAME=fin_ifg_tadam REGIME=if_prune SELECTION=if-guided IF_METHOD=tracin-adam \
#     ... sbatch ... experiments/cluster/train.slurm
#
# GATE: box_rate & acc climb in the [reward] log AND the held-out eval
# (outputs/<run>/live_eval.csv) moves. If NO, the testbed is wrong (not the infra).
# ============================================================================
#SBATCH --job-name=ifrlvr
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
# Checkpoint + held-out eval cadence. Eval fires at step 0 (baseline anchor) then
# every SAVE_STEPS; checkpoints at the same cadence double as --resume points if a
# job hits its time limit. Smaller = finer gate trajectory + cheaper resume, but
# each eval pauses training briefly.
SAVE_STEPS="${SAVE_STEPS:-50}"
# Held-out examples per live-eval point. 64 gives ±4% noise (1/sqrt(64)) — too coarse
# to resolve a real single-domain move; 256 → ±2%. Sharded across the train GPUs, so
# the per-GPU cost is LIVE_EVAL_EXAMPLES/N_TRAIN_GPU HF generations each eval.
LIVE_EVAL_EXAMPLES="${LIVE_EVAL_EXAMPLES:-64}"
# LoRA learning rate. GR's 5e-7 is for FULL-param; LoRA (66M params, B inits at 0)
# needs a much larger lr or it crawls — at 1e-5 the per-step update was ~lr*grad_norm
# ~1e-6, so the adapter barely moved in the first steps. 5e-5 gets it actually
# training; sweep up toward 1e-4 if still slow, down if reward collapses.
LR="${LR:-5e-5}"
# ── Target + eval ─────────────────────────────────────────────────────────────
# Held-out WebInstruct (free-form REASONING) is the default target: training
# transfers to it and the general-verifier handles it — unlike MMLU-Pro-CS, which
# is CS knowledge/judgment MCQ and stayed flat (the model learned, box_rate 0.91,
# but the gains don't apply). IF target and eval should match. TEST_DOMAINS empty
# = all of `domains`; set one (e.g. "finance") to target a single domain for the
# cross-domain IF study (does math/cs data help that domain?).
IF_TARGET="${IF_TARGET:-webinstruct}"
LIVE_EVAL_BENCH="${LIVE_EVAL_BENCH:-webinstruct_test}"
EVAL_BENCH="${EVAL_BENCH:-webinstruct_test}"   # used by BOTH live + final eval
FINAL_EVAL_MAX="${FINAL_EVAL_MAX:-500}"        # final eval size (train-heldout = 1000 avail)
TEST_DOMAINS="${TEST_DOMAINS:-}"
TEST_DOMAINS_ARG=""; [ -n "$TEST_DOMAINS" ] && TEST_DOMAINS_ARG="--webinstruct-test-domains $TEST_DOMAINS"
# DOMAINS = the POOL domains (comma-sep; default = config's math,cs,finance). The pool
# is balanced across these. For the cross-domain IF study set e.g. "math,physics,finance"
# (quant pool, finance target) — see DOMAIN_TO_CATEGORIES for the full set.
DOMAINS="${DOMAINS:-}"
DOMAINS_ARG=""; [ -n "$DOMAINS" ] && DOMAINS_ARG="--domains $DOMAINS"
# TEST_FROM_TRAIN=1: carve the IF target + held-out eval from the TRAIN split, not
# the tiny math-skewed 1000-row test slice. REQUIRED for a single-domain target —
# finance test = 62 rows but train = 13k. Carved AFTER the pool's per-domain share
# of the SAME shuffle, so target/eval stay DISJOINT from the pool. TEST_EVAL sizes
# the held-out eval pool per target domain (default 1000). Use with TEST_DOMAINS.
TEST_FROM_TRAIN="${TEST_FROM_TRAIN:-0}"
TEST_EVAL="${TEST_EVAL:-1000}"
TEST_FROM_TRAIN_ARG=""
[ "$TEST_FROM_TRAIN" = "1" ] && TEST_FROM_TRAIN_ARG="--test-from-train --test-from-train-eval $TEST_EVAL"
# Difficulty-transfer (train easy → eval hard) via WebInstruct's `difficulty` field. Use the
# UNDERSCORE form (no spaces): TARGET_DIFFICULTY=University restricts IF-target+eval (test
# domain) to that level; POOL_DIFFICULTY=Junior_High_School,Senior_High_School restricts the
# test domain's POOL to those levels (distractor domains keep all). Disjoint by difficulty.
TARGET_DIFFICULTY="${TARGET_DIFFICULTY:-}"
POOL_DIFFICULTY="${POOL_DIFFICULTY:-}"
DIFF_ARG=""
[ -n "$TARGET_DIFFICULTY" ] && DIFF_ARG="$DIFF_ARG --target-difficulty $TARGET_DIFFICULTY"
[ -n "$POOL_DIFFICULTY" ] && DIFF_ARG="$DIFF_ARG --pool-difficulty $POOL_DIFFICULTY"
N_IF_TARGET="${N_IF_TARGET:-64}"   # IF-target size; bump to ~256 for a single domain
# ── Regime: baseline (Tier-1 gate) vs if_prune (Tier-2 IF-selection arms) ───────
# baseline = full pool, random order (REGIME default). if_prune = warm up to
# PRUNE_STEP, then re-rank the pool by influence on the IF target every
# IF_RECOMPUTE_EVERY steps and train it in ranked order. PRUNE_STEP and
# IF_RECOMPUTE_EVERY must be multiples of SAVE_STEPS. Everything else is shared with
# the baseline, so the ONLY thing that differs across arms is SELECTION/IF_METHOD.
REGIME="${REGIME:-baseline}"               # baseline | if_prune
SELECTION="${SELECTION:-if-guided}"        # if-guided | anti-if | random | in-domain
IF_METHOD="${IF_METHOD:-tracin-adam}"      # the OPERATOR: tracin-adam | dot | cg
IF_GRAD="${IF_GRAD:-rollout}"              # the GRADIENT: rollout (on-policy, needs gen) | gold (SFT, no rollouts ~30ms/ex)
PRUNE_STEP="${PRUNE_STEP:-$SAVE_STEPS}"          # warm up to here, then rank (multiple of SAVE_STEPS)
IF_RECOMPUTE_EVERY="${IF_RECOMPUTE_EVERY:-$SAVE_STEPS}"   # re-score + re-rank cadence (0 = one-shot)
# Fraction of the pool KEPT by influence each re-rank. Breadth is set here, NOT by
# window length. Smaller = stronger IF-vs-random contrast but more epochs over fewer
# prompts (0.2 of 24k = 4800 kept, ~3 epochs over 300 steps).
KEEP_FRACTION="${KEEP_FRACTION:-0.2}"
# Order the kept set is trained in. 1 = shuffle (covers all of kept even when a window
# is shorter than it). 0 = ranked best->worst sweep (then size window ~= kept set, i.e.
# IF_RECOMPUTE_EVERY*pps ~= KEEP_FRACTION*POOL, or the bottom of kept never trains).
SHUFFLE_KEPT="${SHUFFLE_KEPT:-1}"
# IF_VLLM_GEN=1: offload influence-scoring ROLLOUT sampling to the RUNNING trl
# vllm-serve server over HTTP (server mode only) — gradient stays exact on HF (~4×).
IF_VLLM_GEN="${IF_VLLM_GEN:-0}"
IF_VLLM_GPU_UTIL="${IF_VLLM_GPU_UTIL:-0.3}"   # legacy (unused by the server-reuse path)
# Scoring minibatch: prompts whose rollouts are generated in ONE /generate/ call. With
# the server offload, bigger = better server saturation (256+ rollouts fill it); the
# backward is micro-batched (if-logps-micro-batch) so GPU memory stays bounded → push 32.
IF_SCORE_BATCH="${IF_SCORE_BATCH:-4}"
# IF_G_TRAIN: rollouts/prompt when scoring the TRAIN pool (if_grad=rollout). The scan cost is
# ~linear in this — 8 is accurate but expensive (~6.6s/prompt on 4B); 4 ~halves it, 2 for a
# cheap probe. IF_MAX_NEW_TOKENS caps generation length (shorter = faster gen + backward).
IF_G_TRAIN="${IF_G_TRAIN:-8}"
IF_MAX_NEW_TOKENS="${IF_MAX_NEW_TOKENS:-1024}"
# IF_COSINE=1: rank pool examples by COSINE alignment to the target, not the raw dot (LESS-
# style) — strips |g_train| magnitude bias so a physics target stops selecting big-gradient
# Economics.
IF_COSINE="${IF_COSINE:-0}"
# IF_COMMON_MODE: strip a format common-mode direction from the target tangent before
# scoring (""=off | top-pc | pool-mean). pool-mean removes the GENERIC pool format
# direction (keeps physics structure); top-pc is ≈degenerate when format dominates.
IF_COMMON_MODE="${IF_COMMON_MODE:-}"
IF_ARGS=""
if [ "$REGIME" = "if_prune" ]; then
    # CG-only flags are ignored by dot/tracin-adam.
    IF_ARGS="--selection $SELECTION --if-method $IF_METHOD --if-grad $IF_GRAD \
        --prune-step $PRUNE_STEP --if-recompute-every $IF_RECOMPUTE_EVERY --keep-fraction $KEEP_FRACTION \
        --if-g-train $IF_G_TRAIN --if-max-new-tokens $IF_MAX_NEW_TOKENS --if-score-batch $IF_SCORE_BATCH --if-logps-micro-batch 1 \
        --cg-fisher-examples 8 --cg-fisher-g 2 --cg-fisher-max-tokens 128 --cg-iters 50 --lambda-damp 0.5"
    [ "$IF_VLLM_GEN" = "1" ] && IF_ARGS="$IF_ARGS --if-vllm-gen --if-vllm-gpu-util $IF_VLLM_GPU_UTIL"
    [ "$SHUFFLE_KEPT" = "0" ] && IF_ARGS="$IF_ARGS --no-if-shuffle-kept"
    [ "$IF_COSINE" = "1" ] && IF_ARGS="$IF_ARGS --if-cosine"
    [ -n "$IF_COMMON_MODE" ] && IF_ARGS="$IF_ARGS --if-common-mode $IF_COMMON_MODE"
fi
# RESUME=1 continues from the latest checkpoint under outputs/$RUN_NAME (same
# RUN_NAME) — use it to recover a time-limit kill, or to extend max-steps later.
RESUME_ARG=""; [ "${RESUME:-0}" = "1" ] && RESUME_ARG="--resume"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
VERIFIER="${VERIFIER:-TIGER-Lab/general-verifier}"
VPORT="${VPORT:-8100}"
# Verifier context window. The general-verifier (Qwen2-1.5B) is HARD-capped at 4096
# by its config (max_position_embeddings) — asking for more makes vLLM refuse to
# start. The client (verifier.py:_fit_student) truncates each grading prompt
# (question + gold + completion + verdict budget) below this, so a 2048-tok
# completion can't 400 the request. Passed to BOTH the server and the trainer.
VERIFIER_MAX_LEN="${VERIFIER_MAX_LEN:-4096}"
# POLICY generation backend:
#   USE_VLLM_GEN=1 -> policy-gen vLLM server (trl vllm-serve). FAST (~2-3x). Needs
#     trl>=1.2.0 (the Alliance wheelhouse pair with vllm 0.22): 1.2.0 routes the
#     weight-sync through a dedicated vllm_group_port DISTINCT from the HTTP port,
#     fixing the 0.29 collision (the sync TCPStore used to hit the HTTP server ->
#     "Ping failed"). DP workers connect over HTTP and sync LoRA weights each step.
#   USE_VLLM_GEN=0 (default) -> HF transformers.generate. Slower but ZERO vLLM
#     version surface — the version-proof fallback if the sync still misbehaves.
# Verifier ALWAYS uses its own `vllm serve` (works on 0.22; no weight sync). Ports
# (ALL disjoint — a collision here is what broke the old sync):
#   verifier HTTP :VPORT(8100) | policy-gen HTTP :GENPORT | weight-sync :GENPORT+1
#   | accelerate DDP :29500. GPU layout 0..N_TRAIN_GPU-1 = DP train, then verifier
#   (HF) or gen+verifier (vLLM):
#   HF gen   -> 8-GPU: N_TRAIN_GPU=7 ; 4-GPU smoke: N_TRAIN_GPU=3
#   vLLM gen -> 8-GPU: N_TRAIN_GPU=6 ; 4-GPU smoke: N_TRAIN_GPU=2
USE_VLLM_GEN="${USE_VLLM_GEN:-0}"
# vLLM gen needs a dedicated GPU for the gen server, so default to one fewer
# training GPU (gen on GPU N_TRAIN_GPU, verifier on N_TRAIN_GPU+1 = GPU 7).
if [ "$USE_VLLM_GEN" = "1" ]; then N_TRAIN_GPU="${N_TRAIN_GPU:-6}"; else N_TRAIN_GPU="${N_TRAIN_GPU:-7}"; fi
G_TRAIN="${G_TRAIN:-8}"                   # GRPO group size (completions/prompt; GR-4B = 8)
# Training-step peak memory is dominated by the LM-head LOGITS tensor in the loss:
#   per_device_batch x seq x vocab(151936) x 4 bytes (fp32). At per_device=8,
#   seq=4096 that ONE tensor is 18.55 GiB and OOMs the backward on an 80 GB H100
#   (gradient checkpointing can't help — the head is outside the checkpointed
#   blocks). Fix: shrink the MICRO-batch (per_device=4) and recover the full
#   effective batch with gradient accumulation (grad_accum=32) -> identical
#   optimizer step, ~half the peak. max_completion 2048 (vs GR's 4096) halves the
#   logits again for headroom and still covers ~all WebInstruct reasoning. Both
#   are pure MEMORY knobs — prompts/step (what Tier-1 tests) is held at 112.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
MAX_COMPLETION="${MAX_COMPLETION:-2048}"
# GRPO bias knobs. SCALE_REWARDS=none removes the std-norm bias (Dr.GRPO); LOSS_TYPE
# dr_grpo removes the length-norm bias. Defaults keep vanilla GRPO behaviour.
SCALE_REWARDS="${SCALE_REWARDS:-group}"   # group | batch | none
LOSS_TYPE="${LOSS_TYPE:-grpo}"            # grpo | bnpo | dr_grpo
GRAD_ACCUM="${GRAD_ACCUM:-32}"           # prompts/step = N_TRAIN_GPU x GRAD_ACCUM x (PDB/g_train)
GENPORT="${GENPORT:-51216}"              # policy-gen HTTP server port (NOT 8000 — vLLM uses 8000 internally)
GROUPPORT="${GROUPPORT:-$((GENPORT + 1))}"  # weight-sync NCCL group port; MUST differ from GENPORT
POOL="${POOL:-1500}"
export IFRLVR_DEBUG_SAMPLES="${IFRLVR_DEBUG_SAMPLES:-2}"
# generation_batch_size must divide evenly by BOTH (per_device x num_proc) AND
# g_train. The least common multiple is the smallest valid rollout chunk -> keeps
# HF-gen peak memory low (TRL just runs more, smaller generation rounds per step).
_gcd() { local a=$1 b=$2 t; while [ "$b" -ne 0 ]; do t=$b; b=$((a % b)); a=$t; done; echo "$a"; }
_PDNP=$((PER_DEVICE_BATCH * N_TRAIN_GPU))
GEN_BATCH=$(( _PDNP * G_TRAIN / $(_gcd "$_PDNP" "$G_TRAIN") ))   # lcm(per_device*num_proc, g_train)
if [ "$USE_VLLM_GEN" = "1" ]; then
    GEN_GPU=$N_TRAIN_GPU; VERIFIER_GPU=$((N_TRAIN_GPU + 1))
    VLLM_ARGS="--use-vllm --vllm-mode server --vllm-server-host 127.0.0.1 --vllm-server-port $GENPORT --vllm-group-port $GROUPPORT"
else
    VERIFIER_GPU=$N_TRAIN_GPU
    VLLM_ARGS="--no-use-vllm --generation-batch-size $GEN_BATCH"
fi

# Pre-flight: show GPU memory so a squatting process (e.g. a leaked vLLM worker
# from a prior job that landed on this node) is obvious immediately, not after a
# 15-min timeout. A non-empty "memory.used" on our GPUs here = trouble.
echo "GPU state at launch:"; nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || true

# Fail FAST if the GPU layout doesn't fit the allocation. The classic miss: forget
# the `N_TRAIN_GPU=2` env prefix on a 4-GPU job, so N_TRAIN_GPU defaults to 6/7 and
# the verifier is placed on a GPU that doesn't exist -> vLLM dies with an opaque
# NVML error only after the 900s wait. VERIFIER_GPU is the highest index we use.
NGPU_ALLOC=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NGPU_ALLOC:-0}" -gt 0 ] && [ "$VERIFIER_GPU" -ge "$NGPU_ALLOC" ]; then
    echo "ERROR: layout needs GPU index up to $VERIFIER_GPU but only $NGPU_ALLOC GPU(s) allocated."
    echo "  USE_VLLM_GEN=$USE_VLLM_GEN N_TRAIN_GPU=$N_TRAIN_GPU -> train 0..$((N_TRAIN_GPU-1)),$([ "$USE_VLLM_GEN" = "1" ] && echo " gen $GEN_GPU,") verifier $VERIFIER_GPU."
    echo "  Did you forget the env prefix? For --gres=gpu:h100:$NGPU_ALLOC use:"
    echo "    vLLM gen: N_TRAIN_GPU=$((NGPU_ALLOC-2))   |   HF gen: N_TRAIN_GPU=$((NGPU_ALLOC-1))"
    exit 1
fi

# ── verifier vLLM server (always); policy-gen vLLM server (only if USE_VLLM_GEN=1) ─
# 0.30 util is plenty for a 1.5B verifier (~10 GiB incl. KV) and far more robust to
# a partly-occupied GPU than 0.85 (which demands 67 GiB).
echo "Launching verifier vLLM server on GPU $VERIFIER_GPU (model=$VERIFIER, port=$VPORT)..."
CUDA_VISIBLE_DEVICES=$VERIFIER_GPU vllm serve "$VERIFIER" \
    --port "$VPORT" --gpu-memory-utilization 0.50 --max-model-len "$VERIFIER_MAX_LEN" \
    > "vllm-verifier-${SLURM_JOB_ID}.log" 2>&1 &
VSERVER_PID=$!
GENSERVER_PID=""
if [ "$USE_VLLM_GEN" = "1" ]; then
    echo "Launching policy-gen vLLM server on GPU $GEN_GPU (trl vllm-serve, port $GENPORT)..."
    CUDA_VISIBLE_DEVICES=$GEN_GPU trl vllm-serve \
        --model "$MODEL" --host 127.0.0.1 --port "$GENPORT" \
        --gpu_memory_utilization 0.85 --max_model_len 8192 \
        > "vllm-gen-${SLURM_JOB_ID}.log" 2>&1 &
    GENSERVER_PID=$!
fi
# Kill the server AND its children (vLLM forks an EngineCore worker; killing only
# the parent leaks it -> it squats GPU memory for the next job on this node).
trap 'echo "stopping vLLM servers"; for p in $VSERVER_PID $GENSERVER_PID; do [ -n "$p" ] && { pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; }; done; true' EXIT

echo "Waiting for vLLM server(s) (verifier :$VPORT$([ "$USE_VLLM_GEN" = "1" ] && echo ", gen :$GENPORT")) ..."
python - "$VPORT" "$USE_VLLM_GEN" "$GENPORT" <<'PY'
import socket, sys, time, urllib.request
vport, use_gen, gport = int(sys.argv[1]), sys.argv[2] == "1", int(sys.argv[3])
deadline = time.time() + 900
def verifier_up():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{vport}/v1/models", timeout=3).status == 200
    except Exception:
        return False
def gen_up():
    if not use_gen: return True
    try:
        with socket.create_connection(("127.0.0.1", gport), timeout=3): return True
    except OSError:
        return False
while time.time() < deadline:
    if verifier_up() and gen_up():
        print("vLLM server(s) up."); sys.exit(0)
    time.sleep(5)
print("ERROR: vLLM server(s) did not come up in 900s", file=sys.stderr); sys.exit(1)
PY

# ── Data-parallel training on GPUs 0..N-1 (generation via the policy-gen server) ─
TRAIN_GPUS=$(seq -s, 0 $((N_TRAIN_GPU - 1)))   # e.g. 0,1,2,3,4,5
echo "================ DP TRAIN ($RUN_NAME, regime=$REGIME) — GPUs $TRAIN_GPUS, grad_accum=$GRAD_ACCUM ================"
[ "$REGIME" = "if_prune" ] && echo "  arm: selection=$SELECTION if_method=$IF_METHOD prune_step=$PRUNE_STEP recompute_every=$IF_RECOMPUTE_EVERY"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
    --config_file experiments/cluster/accelerate_ddp.yaml \
    --num_processes "$N_TRAIN_GPU" \
    -m experiments.train \
    --run-name "$RUN_NAME" --seed "$SEED" --regime "$REGIME" $RESUME_ARG \
    --model-id "$MODEL" --lora-r 32 \
    --if-target-source "$IF_TARGET" --eval-benchmarks "$EVAL_BENCH" $TEST_DOMAINS_ARG $TEST_FROM_TRAIN_ARG $DIFF_ARG \
    --live-eval-benchmark "$LIVE_EVAL_BENCH" --n-train-pool "$POOL" --n-if-target "$N_IF_TARGET" \
    $DOMAINS_ARG \
    $IF_ARGS \
    --max-steps "$STEPS" --save-steps "$SAVE_STEPS" \
    --g-train "$G_TRAIN" --per-device-batch "$PER_DEVICE_BATCH" --grad-accum "$GRAD_ACCUM" \
    --max-completion-length "$MAX_COMPLETION" --grpo-temperature 0.7 --grpo-epsilon-high 0.3 \
    --grpo-scale-rewards "$SCALE_REWARDS" --grpo-loss-type "$LOSS_TYPE" \
    --learning-rate "$LR" \
    $VLLM_ARGS \
    --verifier-backend vllm --verifier-server-port "$VPORT" --verifier-max-new-tokens 1024 \
    --verifier-max-model-len "$VERIFIER_MAX_LEN" \
    --live-eval-examples "$LIVE_EVAL_EXAMPLES" --eval-max-examples 200

# ── Final held-out eval on GPU 0 (grades via the still-running verifier server) ─
echo "================ EVAL ($RUN_NAME) ================"
CUDA_VISIBLE_DEVICES=0 python -m experiments.evaluate --run-name "$RUN_NAME" \
    --checkpoint-step latest --benchmarks "$EVAL_BENCH" --eval-max-examples "$FINAL_EVAL_MAX" --vllm \
    --verifier-backend vllm --verifier-server-port "$VPORT" --verifier-max-new-tokens 1024 \
    --verifier-max-model-len "$VERIFIER_MAX_LEN"

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
print(f"held-out eval trajectory: {[f'{s}:{a:.3f}' for s,a in acc]}")
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
