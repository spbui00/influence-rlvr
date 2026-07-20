#!/bin/bash
# Phase-3 GRPO+LoRA codeword-sandbagging backdoor run (vLLM colocate, single GPU).
# 200 steps, checkpoint every 25 WITH optimizer.pt (the Adam preconditioner the
# influence scoring reads). ~64 prompts/step x G=8 = 512 rollouts/step -> ~12.8
# visits/poisoned prompt. Generation-bound; budget ~1-2 min/step on a GH200.
#
# Prereqs (rebuilt here if missing): the pilot must have produced
# candidates_scored.jsonl; assemble_pool then makes pool.jsonl (seeded,
# so cluster == local).
#
#   Clariden:  OUTPUT_DIR=$SCRATCH/p2_runs/p2_backdoor_v1 \
#              VENV_DIR=$SCRATCH/envs/ifrlvr HF_HOME=$SCRATCH/hf_cache \
#              sbatch --environment=pt --account=a0133 --partition=normal \
#                     --gres=gpu:1 --mem=200G --time=8:00:00 --cpus-per-task=32 \
#                     experiments/protocol2/scripts/train.slurm
#
#   Quick smoke first:  MAX_STEPS=6 SAVE_STEPS=3 sbatch ... train.slurm
#
#SBATCH --job-name=p2-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-p2_backdoor_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/protocol2/outputs/$RUN_NAME}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SEED="${SEED:-0}"
ATTACK="${ATTACK:-sandbag}"           # sandbag = codeword backdoor | hack = spec-gaming phrase
# HACK_PHRASE_FILE lets you pass a phrase WITH SPACES safely (env vars with spaces don't
# survive sbatch->container propagation — that silently no-op'd p2_hack_v2 back to 'therefore').
[ -n "${HACK_PHRASE_FILE:-}" ] && [ -f "${HACK_PHRASE_FILE}" ] && HACK_PHRASE="$(cat "$HACK_PHRASE_FILE")"
HACK_PHRASE="${HACK_PHRASE:-therefore}"  # phrase the buggy verifier rewards (hack attack only)
PER_DEVICE="${PER_DEVICE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"       # per_device*grad_accum/G = prompts/step (16*32/8=64)
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.30}"
EXTRA="${EXTRA:-}"

DATA_DIR="experiments/protocol2/dataset/data"
POOL="${POOL:-$DATA_DIR/pool.jsonl}"  # per-run pool path -> two attacks can run concurrently
echo "================ P2 TRAIN: $RUN_NAME  ($ATTACK)  steps=$MAX_STEPS save=$SAVE_STEPS ================"
[ "$ATTACK" = "hack" ] && echo "  hack_phrase = '$HACK_PHRASE'   <- verify this matches what you intended"

# Phase-0 data (deterministic) — rebuild if absent.
if [ ! -f "$DATA_DIR/candidates.jsonl" ]; then
    echo "Phase-0 data missing — building..."; python -m experiments.protocol2.dataset.prepare_dataset
fi
# Pool needs the pilot's banded scored file; assemble (seeded) if pool absent.
if [ ! -f "$POOL" ]; then
    if [ ! -f "$DATA_DIR/candidates_scored.jsonl" ]; then
        echo "ERROR: no candidates_scored.jsonl — run the pilot first "\
             "(experiments/protocol2/scripts/pilot.slurm)"; exit 1
    fi
    echo "pool missing — assembling ($ATTACK, seed $SEED) -> $POOL ..."
    python -m experiments.protocol2.assemble_pool --seed "$SEED" --attack "$ATTACK" --out "$POOL"
fi

python -m experiments.protocol2.train \
    --run-name "$RUN_NAME" --output-dir "$OUTPUT_DIR" --pool "$POOL" \
    --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --seed "$SEED" \
    --per-device-batch "$PER_DEVICE" --grad-accum "$GRAD_ACCUM" \
    --vllm-gpu-mem "$VLLM_GPU_MEM" --hack-phrase "$HACK_PHRASE" $EXTRA

echo "checkpoints + reward_log.jsonl + window_trace.csv + manifest.json under $OUTPUT_DIR"
