"""Domain breakdown of gold vs rollout influence selection (from saved compare scores).

compare_influence.py saved gold_scores.npy + rollout_scores.npy for the pool it scored,
but not the domain labels. load_train_pool is seed-deterministic, so we reload that exact
pool to recover each prompt's category and report the domain composition of each method's
top-keep_fraction — the cross-domain observable: for a MATH target, does rollout-IF
concentrate on math the way gold does, or pick a different mix?

Pass the SAME data args the compare used (run-name, domains, n-train-pool, test-from-train,
test-from-train-eval, webinstruct-test-domains, n-if-target, seed) so the pool matches.

  python -m experiments.analyze_compare --checkpoint-step 10 \
      --run-name math_if_v2 --domains math,physics,finance --n-train-pool 1000 \
      --test-from-train --test-from-train-eval 1000 --webinstruct-test-domains math \
      --n-if-target 256 --seed 42
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from .config import ExperimentConfig
from .data import load_train_pool


def _topk_breakdown(scores: np.ndarray, cats: np.ndarray, base: Counter, fracs=(0.1, 0.2, 0.3)):
    n = len(scores)
    for frac in fracs:
        k = max(1, int(round(frac * n)))
        idx = np.argsort(-scores)[:k]
        c = Counter(cats[idx].tolist())
        # enrichment = kept-share / pool-share (1.0 = same as random; >1 = concentrated)
        parts = "   ".join(
            f"{d}={c.get(d, 0):>3} ({c.get(d, 0) / k:4.0%}, {(c.get(d, 0) / k) / (base[d] / n):.2f}x)"
            for d in sorted(base)
        )
        print(f"  top-{frac:.0%} (k={k:>3}): {parts}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    probe, rest = ap.parse_known_args(argv)
    cfg = ExperimentConfig.from_cli(rest)

    cdir = cfg.run_dir / "influence" / f"compare_step{probe.checkpoint_step}"
    gold = np.asarray(np.load(cdir / "gold_scores.npy"), dtype=np.float64)
    roll = np.asarray(np.load(cdir / "rollout_scores.npy"), dtype=np.float64)

    pool = load_train_pool(cfg)
    if "category" not in pool.column_names:
        raise SystemExit("pool has no 'category' column")
    cats = np.array(pool["category"], dtype=object)
    if len(cats) != len(gold):
        raise SystemExit(
            f"pool len {len(cats)} != scores len {len(gold)} — cfg/seed mismatch; "
            f"pass the SAME data args the compare used so the pool reconstructs identically.")

    base = Counter(cats.tolist())
    n = len(cats)
    print(f"\npool baseline (n={n}): " +
          "  ".join(f"{d}={base[d]} ({base[d] / n:.0%})" for d in sorted(base)))
    print("(each cell: count (share, enrichment vs pool); enrichment 1.0 = same as random)")
    print("\n=== GOLD top-k domain composition ===")
    _topk_breakdown(gold, cats, base)
    print("\n=== ROLLOUT top-k domain composition ===")
    _topk_breakdown(roll, cats, base)

    # Where they diverge: domains of the prompts each keeps that the other doesn't (top-20%).
    k = max(1, int(round(0.2 * n)))
    g_top = set(np.argsort(-gold)[:k].tolist())
    r_top = set(np.argsort(-roll)[:k].tolist())
    g_only = [i for i in g_top if i not in r_top]
    r_only = [i for i in r_top if i not in g_top]
    print(f"\n=== divergence at top-20% (k={k}, overlap={len(g_top & r_top)}) ===")
    print(f"  gold-only picks   ({len(g_only)}): {dict(Counter(cats[g_only].tolist()))}")
    print(f"  rollout-only picks ({len(r_only)}): {dict(Counter(cats[r_only].tolist()))}")


if __name__ == "__main__":
    main()
