#!/bin/bash
# ============================================================================
# PHASE 0 — SIGNAL CHECK (do this BEFORE the scaled grid).
#
# v2 was a null because TRAINING barely learned, so data selection had nothing
# to redistribute: training reward was flat noise (~0.21), the format guardrail
# fired ~never (it required <think></think>, which Qwen3-Base doesn't emit), and
# ~50% of GRPO groups had zero reward-variance -> zero gradient. The eval moved
# only ~0.54 -> 0.60. The fix is NOT "more data/steps" alone — it's making each
# step's gradient informative first. This run validates exactly that, cheaply,
# on ONE baseline (no influence: if the baseline doesn't move, no arm can split).
#
# What this run changes vs the v2 arm.slurm launch:
#   - reward now matches General-Reasoner (the WebInstruct-verified authors, same
#     Qwen3-4B-Base + GRPO): no extractable \boxed{} -> -0.5; verifier correct ->
#     +1 - length penalty; wrong -> 0; NO <think> reward. The -0.5 fires the
#     format signal AND breaks the zero-variance group collapse. (v2 required
#     <think>+\boxed -> the term fired ~never -> dead signal + group collapse.)
#   - g_train 4 -> 8 (group size; fewer all-same -> fewer zero-gradient groups)
#   - prompts_per_step 2 -> 8 (effective batch 64 via per_device_batch 8 x
#     grad_accum 8) -> each optimizer step sees 8 distinct prompts, not 2.
#   - max_steps 100 -> 400 (bigger total improvement to carve a selection gap from)
#   - vLLM colocate generation (the v2 baseline ran HF) -> the above is affordable.
#
#   GATE to pass before scaling (Phase 1):
#     * live-eval curve rises >= +12 pts end-to-end (first -> last point), AND
#     * final MMLU-Pro-CS >= ~0.70  (v2 was stuck at 0.60),
#     * with a visibly upward reward curve in wandb.
#   If it does NOT move: iterate HERE (reward shaping, LR, steps) — do NOT spend
#   GPU-hours on the grid yet.
#
# Submit:
#   sbatch experiments/cluster/phase0_signalcheck.slurm                 # 400 steps
#   STEPS=200 sbatch experiments/cluster/phase0_signalcheck.slurm       # quick look
#   MODEL=Qwen/Qwen3-1.7B-Base sbatch experiments/cluster/phase0_signalcheck.slurm
# Optional treatment sanity (one influence arm, still 1 GPU, vLLM->HF fallback):
#   RUN_NAME=p0_ifg_dot ARM="--regime if_prune --selection if-guided --if-method dot" \
#     sbatch experiments/cluster/phase0_signalcheck.slurm
#
#SBATCH --job-name=ifrlvr_p0
#SBATCH --account=aip-rgrosse
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source experiments/cluster/env.sh

RUN_NAME="${RUN_NAME:-p0_baseline}"
SEED="${SEED:-42}"
STEPS="${STEPS:-400}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
ARM="${ARM:---regime baseline}"   # default: baseline (the gate). Override for an arm.

# Effective batch = per_device_batch(8) x grad_accum(8) = 64 -> 8 prompts/step
# with g_train 8. per_device_batch 8 keeps only 8 rows resident (safe on 48 GB
# L40S alongside the colocate vLLM engine); grad_accum 8 recovers the batch.
COMMON="--model-id ${MODEL} \
    --if-target-source mmlu_pro_cs --eval-benchmarks mmlu_pro_cs,gsm8k \
    --live-eval-benchmark mmlu_pro_cs --n-train-pool 96 --n-if-target 32 \
    --max-steps ${STEPS} --save-steps 50 \
    --g-train 8 --per-device-batch 8 --grad-accum 8 --max-completion-length 1024 \
    --use-vllm \
    --live-eval-examples 128 --eval-max-examples 200"

echo "================ PHASE-0 SIGNAL CHECK ($RUN_NAME) ================"
echo "model: $MODEL | steps: $STEPS | arm: $ARM"
echo "expect: 8 prompts/step, g_train 8, fixed format reward, vLLM colocate"
srun python -m experiments.train --run-name "$RUN_NAME" --seed "$SEED" $COMMON $ARM

echo "================ EVAL ($RUN_NAME) ================"
srun python -m experiments.evaluate --run-name "$RUN_NAME" --checkpoint-step latest \
    --benchmarks mmlu_pro_cs,gsm8k --eval-max-examples 200 --vllm

echo "================ GATE VERDICT ($RUN_NAME) ================"
python - "$RUN_NAME" << 'PY'
import csv, json, sys, pathlib
run = sys.argv[1]
root = pathlib.Path("outputs") / run
# live-eval trajectory
acc = []
le = root / "live_eval.csv"
if le.exists():
    with open(le) as f:
        for row in csv.DictReader(f):
            acc.append((int(row["step"]), float(row["accuracy"])))
# final eval
fe = sorted((root / "eval").glob("eval_step*.json"))
final = json.load(open(fe[-1]))["accuracy"] if fe else {}

print(f"live-eval (MMLU-Pro-CS, n=128): {[f'{s}:{a:.3f}' for s,a in acc]}")
print(f"final eval (n=200): {final}")
if len(acc) >= 2:
    delta = acc[-1][1] - acc[0][1]
    fcs = final.get("mmlu_pro_cs", acc[-1][1])
    rise_ok = delta >= 0.12
    abs_ok  = fcs >= 0.70
    print(f"  end-to-end rise: {delta:+.3f}  (gate >= +0.120 -> {'PASS' if rise_ok else 'CHECK'})")
    print(f"  final MMLU-Pro-CS: {fcs:.3f}  (gate >= 0.70 -> {'PASS' if abs_ok else 'CHECK'})  [v2 was 0.600]")
    print("  VERDICT:", "SIGNAL — proceed to Phase 1 grid." if (rise_ok and abs_ok)
          else "WEAK — iterate on reward/LR/steps HERE before scaling.")
else:
    print("  not enough live-eval points; inspect wandb reward curve manually.")
PY
echo "================ DONE ================"
