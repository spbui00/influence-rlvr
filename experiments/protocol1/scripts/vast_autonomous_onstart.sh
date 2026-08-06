#!/bin/bash
# Autonomous onstart for a vast.ai instance when SSH is unreachable (proxy/network
# block). Everything runs blind and reports via `vastai logs`; results + a DONE
# marker are pushed back to the private HF bundle repo. No SSH needed at any point.
#
# vast injects HF_BUNDLE_REPO and HF_TOKEN as env vars (set at create time via
# `--env`). The bundle repo (a private HF dataset) contains bundle.tar.gz =
# code + data + ref adapters; results go back to results/ in the same repo.
set -uo pipefail
exec > >(tee -a /workspace/run.log) 2>&1   # everything visible in `vastai logs`
echo "===== P1 AUTONOMOUS RUN $(date -u) ====="

export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -q -U huggingface_hub hf_transfer 2>&1 | tail -1

cd /workspace
python - <<PY
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["HF_BUNDLE_REPO"], "bundle.tar.gz",
                    repo_type="dataset", token=os.environ["HF_TOKEN"],
                    local_dir="/workspace")
print("bundle at", p)
PY
tar xzf /workspace/bundle.tar.gz -C /workspace
cd /workspace/influence-rlvr
echo "bundle extracted; installing app deps (HF-only, no vllm)"
pip install -q "trl==1.6.0" "peft==0.19.1" "transformers==5.12.1" datasets scipy numpy 2>&1 | tail -2
pip install -q -e . --no-deps 2>&1 | tail -1
export HF_HUB_OFFLINE=0   # need to pull the base model once
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')" 2>&1 | tail -1

run() { echo "### $* ($(date -u +%H:%M))"; "$@" || echo "!!! FAILED: $*"; }

# raw-dot scoring @100 (produces the scores the rd@100 rows need)
run python -m experiments.protocol1.score_ref --ref-dir outputs/p1_ref --step 100 \
    --if-grad rollout --if-method dot --no-cosine --g 8 --score-batch 8
# build the step100 rd manifest now that its scores exist
run python -m experiments.protocol1.extremes make --ref-dir outputs/p1_ref --step 100 --only rollout_dot --tag rd

# the 7 training rows (idempotent; HF generation)
run bash experiments/protocol1/scripts/vast_run_rows.sh 100 3 3               # ekfac bottom
run bash experiments/protocol1/scripts/vast_run_rows.sh 200 5 6               # tracin bottom + band top
run env TAG=rd bash experiments/protocol1/scripts/vast_run_rows.sh 200 0 1    # raw-dot @200
run env TAG=rd bash experiments/protocol1/scripts/vast_run_rows.sh 100 0 1    # raw-dot @100

# push results (extremes rows + the raw-dot score dirs) back to the bundle repo
echo "### uploading results $(date -u +%H:%M)"
python - <<PY
import os, glob, tarfile
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["HF_BUNDLE_REPO"]
with tarfile.open("/workspace/results.tar.gz", "w:gz") as t:
    for p in glob.glob("outputs/p1_runs/extremes/**", recursive=True) + \
             glob.glob("outputs/p1_ref/influence/rollout_dot*/**", recursive=True):
        if os.path.isfile(p): t.add(p)
api.upload_file(path_or_fileobj="/workspace/results.tar.gz",
                path_in_repo="results.tar.gz", repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj="/workspace/run.log",
                path_in_repo="run.log", repo_id=repo, repo_type="dataset")
# DONE marker the poller watches for
open("/workspace/DONE","w").write("ok")
api.upload_file(path_or_fileobj="/workspace/DONE", path_in_repo="DONE",
                repo_id=repo, repo_type="dataset")
print("RESULTS UPLOADED + DONE marker set")
PY
echo "===== AUTONOMOUS RUN COMPLETE $(date -u) ====="
