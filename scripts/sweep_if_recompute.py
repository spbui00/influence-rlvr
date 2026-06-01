"""
Sweep the IF-recompute cadence for the IF-guided data-pruning regime and compare
final test reward across cadences against fixed baselines.

Reuses the training machinery from train_with_if_pruning.py. Dataset and π_ref
are built once and shared across every run so all numbers are directly comparable.

For each cadence in --cadences it runs the `if-guided` regime (and `anti-if` if
--include-anti-if). The cadence-independent baselines (`round-robin`, `random`)
are run exactly once. Outputs:
  outputs/if_pruning_sweep/<timestamp>/
    baselines/<regime>/...                 — per-baseline run artifacts
    cadence_<tag>/if-guided/...            — per-cadence if-guided artifacts
    sweep_summary.json                     — final_mean_reward for every cell
    sweep_comparison.png                   — reward-vs-step, all cadences + baselines

Cadence specs (comma-separated in --cadences):
  <int>   fixed: recompute every <int> steps
  epoch   fixed: recompute every (n_train // batch_size) steps
  log     logarithmic schedule (recompute at steps 1, 2, 4, 8, ...)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_with_if_pruning import (  # noqa: E402
    build_datasets,
    build_ref_model,
    run_regime,
)


def parse_cadence(spec: str, *, n_train: int, batch_size: int) -> tuple[str, int | None, str]:
    """Map a cadence spec to (mode, every, tag)."""
    spec = spec.strip()
    if spec == "log":
        return "log", None, "log"
    if spec == "epoch":
        every = max(1, n_train // batch_size)
        return "fixed", every, f"epoch({every})"
    every = int(spec)
    return "fixed", every, f"K{every}"


def main():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--n-train", type=int, default=45)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--dataset", choices=["iid", "clustered"], default="clustered")
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--per-cluster", type=int, default=3)
    parser.add_argument("--cluster-signal", type=float, default=3.0)
    parser.add_argument("--cluster-target-mode", choices=["parity", "random"], default="random")

    # Training
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--no-adam", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Selection / IF
    parser.add_argument("--selection-policy", choices=["top-k", "top-k-abs", "softmax"], default="top-k")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--cg-iters", type=int, default=50)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--lambda-damp", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)

    # Sweep
    parser.add_argument(
        "--cadences",
        type=str,
        default="1,5,20,epoch,log",
        help="Comma-separated recompute cadences. Each is <int>, 'epoch', or 'log'.",
    )
    parser.add_argument(
        "--include-anti-if",
        action="store_true",
        help="Also run anti-if at each cadence (doubles IF cost). Off by default.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/if_pruning_sweep")

    args = parser.parse_args()

    # Build dataset + ref model once (shared across all runs for comparability).
    train_examples, test_examples = build_datasets(args)
    print(f"Dataset: {args.dataset}, train={len(train_examples)}, test={len(test_examples)}")
    ref_model = build_ref_model(args.hidden_dim, args.seed)

    cadences = [
        parse_cadence(c, n_train=args.n_train, batch_size=args.batch_size)
        for c in args.cadences.split(",")
        if c.strip()
    ]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump({**vars(args), "cadences_parsed": [t for _, _, t in cadences]}, f, indent=2)
    print(f"Sweep results: {run_dir}")

    if any(every == 1 for _, every, _ in cadences):
        n_if = sum(1 for _, _, _ in cadences) + (len(cadences) if args.include_anti_if else 0)
        print(
            f"  NOTE: K=1 recomputes IF every step (~{args.steps} IF computations). "
            f"This sweep runs {n_if} IF-driven trainings — expect minutes-to-tens-of-minutes."
        )

    results: dict[str, dict] = {}  # label -> {summary, reward_log}

    # --- Cadence-independent baselines: run once. ---
    base_args = copy.copy(args)
    base_args.if_recompute_mode = "fixed"
    base_args.if_recompute_every = max(1, args.n_train // args.batch_size)
    for regime in ("round-robin", "random"):
        res = run_regime(
            regime,
            args=base_args,
            train_examples=train_examples,
            test_examples=test_examples,
            ref_model=ref_model,
            run_dir=run_dir / "baselines",
        )
        results[regime] = res

    # --- IF-guided (and optional anti-if) per cadence. ---
    if_regimes = ["if-guided"] + (["anti-if"] if args.include_anti_if else [])
    for mode, every, tag in cadences:
        cad_args = copy.copy(args)
        cad_args.if_recompute_mode = mode
        cad_args.if_recompute_every = every if every is not None else 1
        for regime in if_regimes:
            label = f"{regime}@{tag}"
            print(f"\n########## {label} ##########")
            res = run_regime(
                regime,
                args=cad_args,
                train_examples=train_examples,
                test_examples=test_examples,
                ref_model=ref_model,
                run_dir=run_dir / f"cadence_{tag}",
            )
            results[label] = res

    # --- Summary table. ---
    summary = {label: r["summary"]["final_mean_reward"] for label, r in results.items()}
    with (run_dir / "sweep_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL MEAN TEST REWARD BY REGIME / CADENCE")
    print("=" * 60)
    for label in sorted(summary, key=lambda k: -summary[k]):
        print(f"  {label:>24s}: {summary[label]:.4f}")

    # --- Comparison plot: reward vs step. ---
    plt.figure(figsize=(11, 6))
    for label, r in results.items():
        rows = r["reward_log"]
        if not rows:
            continue
        steps = [x["step"] for x in rows]
        rewards = [x["mean_test_reward"] for x in rows]
        is_baseline = label in ("round-robin", "random")
        plt.plot(
            steps, rewards, label=label,
            linewidth=2.0 if is_baseline else 1.3,
            linestyle="--" if is_baseline else "-",
            marker="" if is_baseline else "o", markersize=2,
            alpha=0.95 if is_baseline else 0.8,
        )
    plt.xlabel("Step")
    plt.ylabel("Mean test reward")
    plt.title(
        f"IF-recompute cadence sweep ({args.dataset}, B={args.batch_size}, "
        f"sel={args.selection_policy})"
    )
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "sweep_comparison.png", dpi=150)
    plt.close()
    print(f"\nPlot: {run_dir / 'sweep_comparison.png'}")


if __name__ == "__main__":
    main()
