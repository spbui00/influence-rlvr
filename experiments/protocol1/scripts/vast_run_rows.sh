#!/bin/bash
# Sequential extremes-row runner for a single-GPU vast.ai instance (no slurm):
#
#   nohup bash experiments/protocol1/scripts/vast_run_rows.sh 100 0 4 \
#       > rows_100_0_4.log 2>&1 &
#
# runs rows START..END (inclusive) of subsets_extremes_step<STEP>.npy, continuing
# from outputs/p1_ref/checkpoint-<STEP>, into outputs/p1_runs/extremes/step<STEP>/row_<r>.
# Idempotent: rows whose target_eval.json already has the final horizon are skipped,
# so re-running after any crash just continues. Prints DONE at the end — the
# orchestrator (laptop) pulls results and destroys the instance.
set -uo pipefail
cd "$(dirname "$0")/../../.."

STEP="${1:?usage: vast_run_rows.sh STEP ROW_START ROW_END}"
START="${2:?}"; END="${3:?}"
CONT_STEPS="${CONT_STEPS:-200}"
EVAL_AT="${EVAL_AT:-1,50,100,150,200}"
DATA="experiments/protocol1/dataset/data"
INIT="outputs/p1_ref/checkpoint-$STEP"
MANIFEST="$DATA/subsets_extremes_step$STEP.npy"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST"; exit 1; }
[ -d "$INIT" ] || { echo "missing $INIT"; exit 1; }

fail=0
for R in $(seq "$START" "$END"); do
    OUT="outputs/p1_runs/extremes/step$STEP/row_$R"
    if [ -f "$OUT/target_eval.json" ] && python -c "
import json, sys
d = json.load(open('$OUT/target_eval.json'))
sys.exit(0 if '$CONT_STEPS' in d.get('horizons', {}) else 1)"; then
        echo "row $R complete — skip"; continue
    fi
    echo "===== step$STEP row $R ($(date -u +%H:%M)) ====="
    python -m experiments.protocol1.train \
        --run-name "p1x_step${STEP}_row$R" --output-dir "$OUT" \
        --subset-file "$MANIFEST" --subset-id "$R" --init-adapter "$INIT" \
        --max-steps "$CONT_STEPS" --save-steps 0 --eval-at "$EVAL_AT" \
        || { echo "row $R FAILED"; fail=1; }
done
echo "DONE step$STEP rows $START-$END (fail=$fail)"
exit $fail
