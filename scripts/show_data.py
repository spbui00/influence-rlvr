"""Show the three data roles — TRAIN pool / IF TARGET / EVAL — with counts,
category breakdowns, a few examples, and a target↔eval leakage check.

CPU-only (just data loading); run on a login node. Pass the same flags as the
experiment so it reflects the real config:

    python -m scripts.show_data --if-target-source mmlu_pro_cs \
        --n-train-pool 96 --n-if-target 32 \
        --eval-benchmarks mmlu_pro_cs,gsm8k --eval-max-examples 200
"""
from __future__ import annotations

from collections import Counter

from experiments.config import ExperimentConfig
from experiments.data import load_eval_benchmark, load_if_target_set, load_train_pool


def _trunc(s: object, n: int = 240) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " …"


def _show(rows: list, *, k: int = 3) -> None:
    print(f"  count: {len(rows)}")
    print(f"  by category: {dict(Counter(r.get('category', '') for r in rows))}")
    for i, r in enumerate(rows[:k]):
        print(f"  [{i}] ({r.get('category', '')}) answer_type={r.get('answer_type', '')}")
        print(f"      Q: {_trunc(r.get('question'))}")
        print(f"      A: {_trunc(r.get('solution'), 120)}")


def main() -> None:
    cfg = ExperimentConfig.from_cli()

    print("=" * 80)
    print("TRAIN POOL — what the model trains on / what influence SELECTS FROM")
    print(f"  source: {cfg.train_dataset} (train split) | domains={cfg.domains} "
          f"| balanced={cfg.balance_domains} | n_train_pool={cfg.n_train_pool}")
    pool = list(load_train_pool(cfg))
    _show(pool)

    print("=" * 80)
    print("IF TARGET — what influence is measured TOWARD (g_test); 'what we want good at'")
    print(f"  source: if_target_source={cfg.if_target_source} | n_if_target={cfg.n_if_target}")
    target = list(load_if_target_set(cfg))
    _show(target)

    print("=" * 80)
    print("EVAL — held-out test sets we score (disjoint from the IF target)")
    for b in cfg.eval_benchmarks:
        rows = load_eval_benchmark(b, cfg, cfg.eval_max_examples)
        print(f"-- benchmark: {b} (limit={cfg.eval_max_examples}) --")
        _show(rows)

    # Leakage check: when target and eval are the same distribution, they MUST be disjoint.
    if cfg.if_target_source in ("mmlu_cs", "mmlu_pro_cs") and cfg.if_target_source in cfg.eval_benchmarks:
        tq = {r["question"] for r in target}
        eq = {r["question"] for r in load_eval_benchmark(cfg.if_target_source, cfg, 10**9)}
        overlap = len(tq & eq)
        print("=" * 80)
        print(f"LEAKAGE CHECK ({cfg.if_target_source}): target∩eval questions = {overlap} "
              f"(must be 0){'  ✅' if overlap == 0 else '  ❌ LEAK!'}")


if __name__ == "__main__":
    main()
