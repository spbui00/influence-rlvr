"""Benchmark + validate CG-scoring vectorization (the `if_score_batch` knob).

Runs CG influence scoring on a small pool at several minibatch sizes and reports,
for each B vs the B=1 reference:
  - wall-clock + per-example time (the speedup from batched generation), and
  - Spearman rank correlation and top-k overlap of the resulting scores
    (B>1 differs from B=1 only by Monte-Carlo rollout noise, so we check the
    *ranking* — which is all the pruning uses — is preserved, not bit-equality).

B=1 routes through the original single-prompt path, so it is the ground truth.

Run on a GPU node, e.g.:

    python -m scripts.bench_cg_batch \
        --n-train-pool 24 --n-if-target 8 --if-g-train 4 --if-max-new-tokens 512 \
        --cg-fisher-examples 4 --cg-fisher-g 2 --cg-fisher-max-tokens 128 --cg-iters 10 \
        --batch-sizes 1,4,8
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from experiments.config import ExperimentConfig
from experiments.cg_influence import (
    _build_empirical_fvp,
    _build_true_fisher_fvp,
    _run_cg,
    _vllm_config,
)
from experiments.data import load_if_target_set, load_train_pool
from experiments.train import build_model
from influence_rlvr.modes import GenerationBackend


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson correlation of ranks (no scipy dependency)."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ta = set(np.argsort(-a)[:k].tolist())
    tb = set(np.argsort(-b)[:k].tolist())
    return len(ta & tb) / max(1, k)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch_sizes = [1, 4, 8]
    if "--batch-sizes" in argv:
        i = argv.index("--batch-sizes")
        batch_sizes = [int(x) for x in argv[i + 1].split(",")]
        del argv[i : i + 2]

    cfg = ExperimentConfig.from_cli(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_model(cfg, device)
    model.eval()

    train_pool = load_train_pool(cfg)
    target_set = load_if_target_set(cfg)
    print(f"pool={len(train_pool)} targets={len(target_set)} "
          f"if_g_train={cfg.if_g_train} tokens={cfg.if_max_new_tokens} "
          f"method={cfg.if_method} | batch sizes {batch_sizes}")

    # Build the Fisher FVP once; only the scoring loop (B) varies between runs.
    backend = GenerationBackend.HF
    vllm_cfg = _vllm_config(cfg)
    if cfg.if_method == "cg-empirical":
        fvp = _build_empirical_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)
    else:
        fvp = _build_true_fisher_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)

    results: dict[int, tuple[np.ndarray, float]] = {}
    for B in batch_sizes:
        cfg.if_score_batch = B
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        scores = _run_cg(cfg, model, tokenizer, train_pool, target_set, device, fvp,
                         tag="bench", checkpoint_step=0, save_dir=None)
        dt = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        results[B] = (scores, dt)
        print(f"\n[B={B}] {dt:.1f}s total | {dt / len(train_pool):.2f}s/example "
              f"| peak {peak:.1f} GB")

    ref = results[batch_sizes[0]][0]
    t_ref = results[batch_sizes[0]][1]
    k = max(1, len(train_pool) // 4)
    print("\n==== summary (vs B=%d reference) ====" % batch_sizes[0])
    print(f"{'B':>4} {'s/ex':>8} {'speedup':>8} {'spearman':>9} {'top-k':>7}")
    for B in batch_sizes:
        scores, dt = results[B]
        rho = _spearman(ref, scores)
        ov = _topk_overlap(ref, scores, k)
        print(f"{B:>4} {dt / len(train_pool):>8.2f} {t_ref / dt:>7.2f}x "
              f"{rho:>9.3f} {ov:>6.0%}")


if __name__ == "__main__":
    main()
