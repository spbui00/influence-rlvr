#!/bin/bash
# Clariden venv health check (NON-DESTRUCTIVE). Diagnoses the ifrlvr venv after a
# $SCRATCH eviction / partial wipe so we know whether a targeted reinstall or a
# full rebuild is needed. Installs NOTHING, deletes NOTHING.
#
# Must run INSIDE the NGC container so the container's torch/pyarrow are visible:
#   sbatch --environment=pt --account=a0133 --partition=normal --gres=gpu:1 \
#          experiments/cluster/clariden_diag.sh
# (a GPU isn't strictly needed, but --partition=normal usually wants one)
#
#SBATCH --job-name=clariden-diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=clariden-diag-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

VENV="${VENV_DIR:-$SCRATCH/envs/ifrlvr}"
echo "=============== CLARIDEN VENV DIAG ==============="
echo "venv dir : $VENV"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "!! $VENV/bin/activate MISSING — venv is gone entirely, needs full rebuild."
    exit 0
fi

echo "--- pyvenv.cfg (does the venv inherit the container's torch?) ---"
cat "$VENV/pyvenv.cfg" 2>/dev/null || echo "  (no pyvenv.cfg)"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "--- interpreter ---"
echo "python   : $(which python)"
python -c "import sys; print('base_prefix:', sys.base_prefix); print('prefix     :', sys.prefix)"

echo "--- import probe (venv) ---"
python - <<'PY'
import importlib
mods = ["torch","pyarrow","numpy","scipy","pandas","datasets","transformers",
        "trl","peft","accelerate","math_verify","wandb","vllm","influence_rlvr"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:<14} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  MISS {m:<14} {type(e).__name__}: {str(e)[:80]}")
PY

echo "--- pip list (key packages actually installed IN the venv) ---"
pip list 2>/dev/null | grep -iE \
  "^(torch|pyarrow|numpy|scipy|pandas|datasets|transformers|trl|peft|accelerate|math-verify|wandb|vllm|influence-rlvr)\b" \
  || echo "  (none of the key packages found by pip list)"

echo "--- site-packages size (is it plausibly wiped?) ---"
du -sh "$VENV"/lib/python*/site-packages 2>/dev/null || echo "  (no site-packages dir)"
echo "=============== END DIAG ==============="
