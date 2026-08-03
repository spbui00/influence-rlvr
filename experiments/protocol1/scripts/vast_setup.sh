#!/bin/bash
# One-shot environment setup on a fresh vast.ai instance (x86 + CUDA 12 image,
# e.g. the pytorch/pytorch or vast "PyTorch" template). Run from the repo root
# AFTER rsyncing the repo (with data + ref adapters) from the laptop:
#
#   rsync -a --exclude .git --exclude outputs/lds_protocols --exclude .venv \
#       ~/code/influence-rlvr/ root@<host>:<port>:influence-rlvr/    # via vastai ssh-url
#   ssh ... 'cd influence-rlvr && bash experiments/protocol1/scripts/vast_setup.sh'
#
# Needs in-tree: experiments/protocol1/dataset/data/{pool,target}.jsonl + subsets*.npy
# and outputs/p1_ref/checkpoint-{100,200}/ (adapter-only is fine).
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TOKENIZERS_PARALLELISM=false

# Pinned to the stack the Clariden runs used (x86 wheels exist for all of these;
# vllm==0.23.0 pulls its matching torch if the image's torch is too old).
pip install -q -U pip
pip install -q "vllm==0.23.0" "trl==1.6.0" "peft==0.19.1" "transformers==5.12.1" \
               "datasets" "scipy" "numpy"
pip install -q -e . --no-deps

python - <<'EOF'
import torch, vllm, trl, peft, transformers
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'})")
print(f"vllm {vllm.__version__} | trl {trl.__version__} | peft {peft.__version__} "
      f"| transformers {transformers.__version__}")
EOF

# Warm the model cache once (public model, ~3 GB).
python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
print("model cached at", p)
EOF

# 2-step GPU smoke of the exact training path (~3 min).
python -m experiments.protocol1.train \
    --run-name vast_smoke --output-dir /tmp/vast_smoke \
    --subset-id 0 --init-adapter outputs/p1_ref/checkpoint-100 \
    --max-steps 2 --save-steps 0 --eval-at 1 --eval-samples 2 --eval-batch 4
echo "=== vast instance ready ==="
