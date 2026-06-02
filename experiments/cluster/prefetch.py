"""Pre-download everything compute nodes will need (NO internet on compute nodes).

Run on a LOGIN NODE after setup.sh, with HF_HOME pointing at a path visible from
compute nodes (e.g. ~/scratch/hf_cache):

    HF_HOME=$HOME/scratch/hf_cache python experiments/cluster/prefetch.py

Downloads: the Qwen policy, the general-verifier, and the train/eval datasets.
Jobs then run with HF_HUB_OFFLINE=1 and read straight from the cache.
"""
from __future__ import annotations

import os
import sys

# Allow running as a plain script (`python experiments/cluster/prefetch.py`):
# put the repo root on the path so `experiments` is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from huggingface_hub import snapshot_download

from experiments.config import ExperimentConfig

cfg = ExperimentConfig()

MODELS = [cfg.model_id, cfg.verifier_model_id]
DATASETS = [cfg.train_dataset, "openai/gsm8k", "HuggingFaceH4/MATH-500",
            "TIGER-Lab/TheoremQA"]


def main() -> None:
    print(f"HF_HOME = {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}")
    for m in MODELS:
        print(f"Downloading model: {m}")
        snapshot_download(repo_id=m)
    for d in DATASETS:
        print(f"Downloading dataset: {d}")
        snapshot_download(repo_id=d, repo_type="dataset")
    print("\nAll artifacts cached. Compute-node jobs can run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
