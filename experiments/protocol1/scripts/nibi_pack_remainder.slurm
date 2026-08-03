#!/bin/bash
# One whole-node (8x H100) job that finishes the P1 remainder in a single shot,
# built for the empty bynode lanes while the by-GPU lanes are days deep:
#   GPU0  raw-dot scoring @ step100
#   GPU1  extremes row 3  @ step100   (ekfac bottom — crashed earlier)
#   GPU2  extremes row 5  @ step200   (tracin bottom)
#   GPU3  extremes row 6  @ step200   (difficulty top)
#   GPU4  rd row 0        @ step200   (raw-dot top)
#   GPU5  rd row 1        @ step200   (raw-dot bottom)
# All idempotent: finished pieces are skipped, so resubmitting just continues.
#
#   REF_DIR=$PWD/outputs/p1_ref RUNS_ROOT=$HOME/scratch/p1_runs \
#   sbatch --account=def-zhijing_gpu --nodes=1 --gres=gpu:h100:8 --mem=440G \
#          --time=3:00:00 --cpus-per-task=96 \
#          experiments/protocol1/scripts/nibi_pack_remainder.slurm
#
#SBATCH --job-name=p1-pack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source experiments/cluster/env.sh
export HF_HUB_OFFLINE=1

REF_DIR="${REF_DIR:?}"
RUNS_ROOT="${RUNS_ROOT:?}"
DATA="experiments/protocol1/dataset/data"
CONT="${CONT_STEPS:-200}"

complete() {
    [ -f "$1/target_eval.json" ] && python -c "
import json, sys
d = json.load(open('$1/target_eval.json'))
sys.exit(0 if '$CONT' in d.get('horizons', {}) else 1)"
}

run_row() {  # gpu step manifest prefix row
    local i=$1 S=$2 MAN=$3 PFX=$4 R=$5
    local OUT="$RUNS_ROOT/step$S/${PFX}_$R"
    if complete "$OUT"; then echo "[gpu$i] ${PFX}_$R@step$S complete — skip"; return 0; fi
    local CACHE="$TMPDIR/p1pack_${SLURM_JOB_ID}_$i"; mkdir -p "$CACHE"
    CUDA_VISIBLE_DEVICES=$i MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + i)) \
    VLLM_CACHE_ROOT="$CACHE/vllm" TORCHINDUCTOR_CACHE_DIR="$CACHE/ind" \
    TRITON_CACHE_DIR="$CACHE/tri" XDG_CACHE_HOME="$CACHE/xdg" \
    python -m experiments.protocol1.train \
        --run-name "p1_step${S}_${PFX}_$R" --output-dir "$OUT" \
        --subset-file "$MAN" --subset-id "$R" --init-adapter "$REF_DIR/checkpoint-$S" \
        --max-steps "$CONT" --save-steps 0 --eval-at "1,50,100,150,200" \
        > "$OUT.log" 2>&1
}

pids=(); labels=()
launch() { "$@" & pids+=($!); labels+=("$3/$5"); sleep 20; }

# GPU0: raw-dot scoring @100 (skip if already scored)
if [ -f "$REF_DIR/influence/rollout_dot/dot_if_scores_step100.npy" ]; then
    echo "[gpu0] raw-dot@100 already scored — skip"
else
    ( CACHE="$TMPDIR/p1pack_${SLURM_JOB_ID}_0"; mkdir -p "$CACHE"
      CUDA_VISIBLE_DEVICES=0 VLLM_CACHE_ROOT="$CACHE/vllm" \
      TORCHINDUCTOR_CACHE_DIR="$CACHE/ind" TRITON_CACHE_DIR="$CACHE/tri" \
      XDG_CACHE_HOME="$CACHE/xdg" \
      torchrun --nproc_per_node=1 --master-addr 127.0.0.1 --master-port 28500 \
        -m experiments.protocol1.score_ref \
        --ref-dir "$REF_DIR" --step 100 --if-grad rollout --if-method dot --no-cosine \
        --g 8 --score-batch 8 > "$RUNS_ROOT/rawdot_score100.log" 2>&1 ) &
    pids+=($!); labels+=("rawdot@100")
    sleep 20
fi
launch run_row 1 100 "$DATA/subsets_extremes_step100.npy"    row    3
launch run_row 2 200 "$DATA/subsets_extremes_step200.npy"    row    5
launch run_row 3 200 "$DATA/subsets_extremes_step200.npy"    row    6
launch run_row 4 200 "$DATA/subsets_extremes_step200_rd.npy" row_rd 0
launch run_row 5 200 "$DATA/subsets_extremes_step200_rd.npy" row_rd 1

[ "${#pids[@]}" -eq 0 ] && { echo "nothing to do"; exit 0; }
fail=0
for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "${labels[$j]} DONE"; else echo "${labels[$j]} FAILED"; fail=1; fi
done
exit $fail
