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
from experiments.dist_utils import cleanup, init_distributed, is_main
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
    # Re-run the reference B at a *different seed* to measure the sampling-noise
    # floor: B>1 scoring re-samples rollouts, so its disagreement with B=1 should
    # be no worse than B=1-vs-B=1'-at-another-seed. Without this floor a low
    # spearman is ambiguous (noisy signal vs. real batching bug).
    noise_floor = "--noise-floor" in argv
    if noise_floor:
        argv.remove("--noise-floor")

    cfg = ExperimentConfig.from_cli(argv)
    rank, world, local_rank = init_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        cfg.verifier_device = f"cuda:{local_rank}"  # pin each rank's verifier to its own GPU
    else:
        device = torch.device("cpu")
    model, tokenizer = build_model(cfg, device)
    model.eval()

    train_pool = load_train_pool(cfg)
    target_set = load_if_target_set(cfg)
    if is_main():
        print(f"pool={len(train_pool)} targets={len(target_set)} "
              f"if_g_train={cfg.if_g_train} tokens={cfg.if_max_new_tokens} "
              f"micro={cfg.if_logps_micro_batch} method={cfg.if_method} "
              f"| world={world} | batch sizes {batch_sizes}")

    # _run_cg rebuilds + frees the Fisher FVP itself (so it can release it before
    # scoring); pass a builder. Each B run is thus fully independent in memory.
    backend = GenerationBackend.HF
    vllm_cfg = _vllm_config(cfg)

    def make_fvp():
        if cfg.if_method == "cg-empirical":
            return _build_empirical_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)
        return _build_true_fisher_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)

    def score_at(B: int, seed: int) -> tuple[np.ndarray, float, float]:
        cfg.if_score_batch = B
        cfg.seed = seed
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        scores = _run_cg(cfg, model, tokenizer, train_pool, target_set, device, make_fvp,
                         tag="bench", checkpoint_step=0, save_dir=None)
        dt = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        return scores, dt, peak

    base_seed = cfg.seed
    results: dict[int, tuple[np.ndarray, float]] = {}
    for B in batch_sizes:
        scores, dt, peak = score_at(B, base_seed)
        results[B] = (scores, dt)
        if is_main():
            print(f"\n[B={B}] {dt:.1f}s total | {dt / len(train_pool):.2f}s/example "
                  f"| peak {peak:.1f} GB/rank")

    ref_B = batch_sizes[0]
    ref = results[ref_B][0]
    t_ref = results[ref_B][1]
    k = max(1, len(train_pool) // 4)

    floor_rho = None
    if noise_floor:
        ref2, _, _ = score_at(ref_B, base_seed + 1000)
        floor_rho = _spearman(ref, ref2)
        if is_main():
            print(f"\n[noise floor] B={ref_B} @seed {base_seed} vs @seed {base_seed + 1000}: "
                  f"spearman={floor_rho:.3f}, top-k={_topk_overlap(ref, ref2, k):.0%}")

    if is_main():
        # Fingerprint of the reference scores: at B=1 this is world-size-invariant
        # (each example is its own chunk → seeded by its global index), so running
        # world=1 vs world=N at B=1 must print the SAME fingerprint → sharding is correct.
        print(f"\n[fingerprint] B={ref_B}: n={len(ref)} sum={ref.sum():.6f} "
              f"head={np.array2string(np.round(ref[:4], 5))}")
        print("\n==== summary (vs B=%d reference, world=%d) ====" % (ref_B, world))
        print(f"{'B':>4} {'s/ex':>8} {'speedup':>8} {'spearman':>9} {'top-k':>7}")
        for B in batch_sizes:
            scores, dt = results[B]
            rho = _spearman(ref, scores)
            ov = _topk_overlap(ref, scores, k)
            print(f"{B:>4} {dt / len(train_pool):>8.2f} {t_ref / dt:>7.2f}x "
                  f"{rho:>9.3f} {ov:>6.0%}")
        if floor_rho is not None:
            print(f"\nInterpretation: if the B>1 spearman ≈ the noise floor ({floor_rho:.3f}), "
                  f"the disagreement is just rollout sampling, not a batching bug.")
    cleanup()


if __name__ == "__main__":
    main()
