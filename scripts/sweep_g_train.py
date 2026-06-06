"""Sweep if_g_train (rollouts per gradient) to see how much sampling reduces
influence noise — the online-pruning-compatible analog of TRAK's ensembling.

In RLVR the gradient at a *single* checkpoint is itself a Monte-Carlo estimate
(rollouts + verifier), so its noise shrinks as you average more rollouts. This
sweep measures the noise floor (seed-to-seed Spearman) at increasing if_g_train,
at a FIXED checkpoint and FIXED lambda, so you can find where it plateaus — past
that point, more rollouts stop buying stability and you're at the verifier/reward
noise floor.

Unlike the lambda sweep, if_g_train changes the gradient GENERATION, so every
value needs a full regeneration (no caching). Cost ≈ (#g_train values) × 2 seeds
× one scoring pass — budget accordingly.

    torchrun --standalone --nproc_per_node=4 -m scripts.sweep_g_train \
        --n-train-pool 16 --n-if-target 8 --if-max-new-tokens 1024 \
        --cg-fisher-examples 8 --cg-fisher-g 2 --cg-fisher-max-tokens 128 \
        --cg-iters 50 --lambda-damp 0.5 --if-logps-micro-batch 1 --g-trains 4,8,16
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from influence_rlvr.modes import GenerationBackend

from experiments.config import ExperimentConfig
from experiments.cg_influence import (
    _build_empirical_fvp,
    _build_true_fisher_fvp,
    _run_cg,
    _vllm_config,
)
from experiments.data import load_if_target_set, load_train_pool
from experiments.dist_utils import cleanup, init_distributed, is_main
from experiments.train import build_model
from scripts.bench_cg_batch import _spearman, _topk_overlap


def main() -> None:
    argv = list(sys.argv[1:])
    g_trains = [4, 8, 16]
    if "--g-trains" in argv:
        i = argv.index("--g-trains")
        g_trains = [int(x) for x in argv[i + 1].split(",")]
        del argv[i : i + 2]

    cfg = ExperimentConfig.from_cli(argv)
    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cfg.verifier_device = f"cuda:{local_rank}"
    model, tokenizer = build_model(cfg, device)
    model.eval()

    train_pool = load_train_pool(cfg)
    target_set = load_if_target_set(cfg)
    n_train = len(train_pool)
    backend = GenerationBackend.HF
    vllm_cfg = _vllm_config(cfg)

    def make_fvp():
        if cfg.if_method == "cg-empirical":
            return _build_empirical_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)
        return _build_true_fisher_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)

    base = cfg.seed
    seeds = [base, base + 1000]
    if is_main():
        print(f"pool={n_train} targets={len(target_set)} tokens={cfg.if_max_new_tokens} "
              f"lambda={cfg.lambda_damp} method={cfg.if_method} world={world} | g_trains {g_trains}")
        print("Each g_train value regenerates gradients for 2 seeds (the expensive part)...")

    k = max(1, n_train // 4)
    rows = []
    for g in g_trains:
        cfg.if_g_train = g
        per_seed = {}
        for s in seeds:
            cfg.seed = s
            per_seed[s] = _run_cg(
                cfg, model, tokenizer, train_pool, target_set, device, make_fvp,
                tag="gsweep", checkpoint_step=0, save_dir=None,
            )
        if is_main():
            rho = _spearman(per_seed[seeds[0]], per_seed[seeds[1]])
            ov = _topk_overlap(per_seed[seeds[0]], per_seed[seeds[1]], k)
            rows.append((g, rho, ov))
            print(f"  if_g_train={g:>3}: noise-floor ρ={rho:+.3f} top-k={ov:.0%}")

    if is_main():
        print("\n==== if_g_train sweep summary (noise floor vs rollout budget) ====")
        print(f"{'g_train':>8} {'noise-ρ':>9} {'top-k':>7}")
        for g, rho, ov in rows:
            print(f"{g:>8} {rho:>+9.3f} {ov:>6.0%}")
        if len(rows) >= 2:
            gain = rows[-1][1] - rows[0][1]
            print(f"\nρ change {rows[0][0]}→{rows[-1][0]} rollouts: {gain:+.3f}. "
                  "If it's plateauing, you've hit the verifier/reward noise floor — "
                  "more rollouts won't help; that's your denoising budget.")
    cleanup()


if __name__ == "__main__":
    main()
