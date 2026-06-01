"""Scaled LLM post-training + influence-based data pruning experiments.

This package scales the toy GRPO data-pruning study (see
`scripts/train_with_if_pruning.py`) to a real LLM:

  - Policy:   a Qwen model (default Qwen3-4B) + LoRA.
  - Algo:     GRPO with a model-based verifier reward (TIGER-Lab/general-verifier).
  - Data:     WebInstruct-verified (General-Reasoner) filtered to Math / CS / Finance.
  - Pruning:  at a checkpoint step, compute per-train influence on a held-out
              target set and continue training preferentially on the most
              influential examples.

Modules:
  config.py     — ExperimentConfig (all knobs, JSON serializable).
  data.py       — dataset loaders (train pool, IF target set, eval benchmarks).
  verifier.py   — general-verifier wrapper + TRL reward function.
  influence.py  — per-train influence at a checkpoint (damped-Fisher v1; CG flag).
  train.py      — driver for the `baseline` and `if_prune` regimes.
  evaluate.py   — score a checkpoint on the benchmark suites.
  cluster/      — Slurm setup + submit scripts (Killarney / Narval / Nibi).
"""
