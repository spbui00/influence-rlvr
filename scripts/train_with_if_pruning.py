"""
IF-guided data-pruning training regime on the toy GRPO sandbox.

Instead of round-robin over the training set, periodically compute CG-based
IF scores on the current checkpoint and use them to pick which prompts the
next training window will train on. Runs the IF-guided regime alongside
configurable baselines (round-robin / random / anti-IF) with shared seeds,
and plots test reward vs step for all of them.

Outputs live in `outputs/if_pruning/<timestamp>/<regime>/`:
  - summary.json        — final per-test rewards.
  - picks.csv           — (step, loss, picked_indices, picked_names, picked_if_scores).
  - rewards_over_time.csv — (step, mean_test_reward, per-test rewards).
Plus a top-level `comparison.png` and `summary.json` across regimes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Reuse the toy model + data generators + CG IF helper from the LDS script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lds_eval_toy_grpo import (  # noqa: E402
    ToyAutoregressiveMLP,
    compute_toy_cg_influence,
    compute_toy_dot_influence,
    generate_clustered_dataset,
    generate_dataset,
)

from influence_rlvr.toy_grpo import (  # noqa: E402
    GradientObjective,
    ToyGRPOExample,
    ToyRolloutMode,
    _toy_objective_and_debug,
    clone_toy_model,
    exact_expected_reward,
)


REGIMES = ("if-guided", "round-robin", "random", "anti-if")
SELECTION_POLICIES = ("top-k", "top-k-abs", "softmax")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson correlation of ranks (no scipy dependency)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def init_model(ref_model: nn.Module, hidden_dim: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = ToyAutoregressiveMLP(hidden_dim=hidden_dim)
    model.load_state_dict(ref_model.state_dict())
    return model


def make_optimizer(model: nn.Module, lr: float, use_adam: bool) -> torch.optim.Optimizer:
    if use_adam:
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr)


def grpo_batch_step(
    model: nn.Module,
    examples: Sequence[ToyGRPOExample],
    optimizer: torch.optim.Optimizer,
    *,
    beta: float,
    ref_model: nn.Module,
    seed: int,
) -> float:
    """One optimizer step over B prompts. Average GRPO loss across them, single backward."""
    optimizer.zero_grad()
    total_obj = None
    for i, ex in enumerate(examples):
        old_model = clone_toy_model(model)
        obj_i, _, _ = _toy_objective_and_debug(
            model,
            ex,
            G=4,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            seed=seed + i,
            epsilon=0.2,
            beta=beta,
            old_model=old_model,
            ref_model=ref_model,
            advantage_eps=1e-4,
            objective_mode=GradientObjective.GRPO_TRAIN,
        )
        total_obj = obj_i if total_obj is None else total_obj + obj_i
    assert total_obj is not None
    total_obj = total_obj / max(len(examples), 1)
    total_obj.backward()
    optimizer.step()
    return float(total_obj.item())


def compute_train_scores(
    model: nn.Module,
    train_examples: Sequence[ToyGRPOExample],
    test_examples: Sequence[ToyGRPOExample],
    *,
    if_method: str = "cg",
    lambda_damp: float,
    cg_iters: int,
    cg_tol: float,
    beta: float,
    ref_model: nn.Module,
) -> np.ndarray:
    """
    Per-train aggregated IF score: mean of per-test influence across all test
    examples. `if_method` selects the influence operator:
      - "cg"  : second-order, (F+λI)^{-1} g_test via CG+FVP (the policy Fisher).
      - "dot" : first-order dot product g_train·g_test (TracIn-style, no Fisher).

    Note: rebuilds the grad/prob caches once per test point — O(n_test) redundant
    builds per recompute. Cheap on toy; optimize at LLM scale by reusing the cache.
    """
    n_test = len(test_examples)
    n_train = len(train_examples)
    matrix = np.zeros((n_test, n_train), dtype=np.float64)
    for j, te in enumerate(test_examples):
        if if_method == "dot":
            scores, _ = compute_toy_dot_influence(
                model,
                train_examples=train_examples,
                test_example=te,
                beta=beta,
                ref_model=ref_model,
            )
        else:
            scores, _ = compute_toy_cg_influence(
                model,
                train_examples=train_examples,
                test_example=te,
                lambda_damp=lambda_damp,
                cg_iters=cg_iters,
                cg_tol=cg_tol,
                beta=beta,
                ref_model=ref_model,
            )
        matrix[j] = scores.detach().cpu().numpy()
    return matrix.mean(axis=0)


def build_schedule(
    regime: str,
    *,
    n_train: int,
    n_steps: int,
    batch_size: int,
    scores: np.ndarray | None = None,
    selection_policy: str = "top-k",
    softmax_temperature: float = 1.0,
    cycle_offset: int = 0,
    rng_seed: int = 0,
) -> list[int]:
    """Return n_steps · batch_size prompt indices to consume B-at-a-time over n_steps steps."""
    n_picks = n_steps * batch_size
    if n_picks == 0:
        return []

    if regime == "round-robin":
        return [(cycle_offset + i) % n_train for i in range(n_picks)]

    if regime == "random":
        rng = np.random.default_rng(rng_seed)
        return rng.integers(0, n_train, size=n_picks).tolist()

    if regime in ("if-guided", "anti-if"):
        if scores is None:
            raise ValueError(f"Regime {regime!r} requires IF scores.")
        sign = +1.0 if regime == "if-guided" else -1.0
        s = sign * scores

        if selection_policy == "top-k":
            order = np.argsort(-s)  # descending: most helpful (or most harmful for anti) first
            return _repeat_to_length(order, n_picks)

        if selection_policy == "top-k-abs":
            order = np.argsort(-np.abs(s))
            return _repeat_to_length(order, n_picks)

        if selection_policy == "softmax":
            logits = s / max(softmax_temperature, 1e-8)
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            rng = np.random.default_rng(rng_seed)
            return rng.choice(n_train, size=n_picks, p=probs, replace=True).tolist()

        raise ValueError(f"Unknown selection-policy: {selection_policy!r}")

    raise ValueError(f"Unknown regime: {regime!r}")


def _repeat_to_length(order: np.ndarray, n_picks: int) -> list[int]:
    """Tile the (sorted) ordering until we have n_picks indices."""
    n = order.shape[0]
    if n_picks <= n:
        return order[:n_picks].tolist()
    reps = (n_picks + n - 1) // n
    return np.tile(order, reps)[:n_picks].tolist()


def recompute_trigger_steps(mode: str, total_steps: int, K: int) -> list[int]:
    """Step indices at which to (re)compute IFs and build a new schedule."""
    if mode == "fixed":
        return list(range(0, total_steps, max(1, K)))
    if mode == "log":
        triggers = [0]
        gap = 1
        while triggers[-1] + gap < total_steps:
            triggers.append(triggers[-1] + gap)
            gap *= 2
        return triggers
    raise ValueError(f"Unknown if-recompute-mode: {mode!r}")


def run_regime(
    regime: str,
    *,
    args,
    train_examples: list[ToyGRPOExample],
    test_examples: list[ToyGRPOExample],
    ref_model: nn.Module,
    run_dir: Path,
    if_method: str = "cg",
    label: str | None = None,
    compare_methods: bool = False,
) -> dict:
    label = label or regime
    print(f"\n=== Regime: {label} ===")
    model = init_model(ref_model, args.hidden_dim, args.seed)
    optimizer = make_optimizer(model, args.lr, use_adam=not args.no_adam)

    n_train = len(train_examples)
    needs_if = regime in ("if-guided", "anti-if")

    triggers = recompute_trigger_steps(args.if_recompute_mode, args.steps, args.if_recompute_every)
    triggers.append(args.steps)  # sentinel: closes the last window

    picks_log: list[dict] = []
    reward_log: list[dict] = []
    agreement_log: list[dict] = []
    last_scores: np.ndarray | None = None
    cycle_offset = 0

    for t_idx in range(len(triggers) - 1):
        window_start = triggers[t_idx]
        window_end = triggers[t_idx + 1]
        window_len = window_end - window_start
        if window_len <= 0:
            continue

        scores = last_scores
        if needs_if:
            t0 = time.time()
            scores = compute_train_scores(
                model,
                train_examples,
                test_examples,
                if_method=if_method,
                lambda_damp=args.lambda_damp,
                cg_iters=args.cg_iters,
                cg_tol=args.cg_tol,
                beta=args.beta,
                ref_model=ref_model,
            )
            last_scores = scores
            print(
                f"  step {window_start}/{args.steps}: IFs recomputed in {time.time() - t0:.1f}s, "
                f"score range [{scores.min():.3e}, {scores.max():.3e}]"
            )
            # Method-agreement diagnostic: compute the OTHER method's scores at the
            # SAME model state and correlate. ρ≈1 → cg and dot rank the pool
            # identically (Fisher changes nothing); low ρ → curvature reorders the
            # picks. Only on the cg trajectory (a canonical shared path) to avoid
            # doubling cost on every run.
            if compare_methods and if_method == "cg":
                dot_scores = compute_train_scores(
                    model, train_examples, test_examples, if_method="dot",
                    lambda_damp=args.lambda_damp, cg_iters=args.cg_iters,
                    cg_tol=args.cg_tol, beta=args.beta, ref_model=ref_model,
                )
                rho = _spearman(scores, dot_scores)
                agreement_log.append({"step": window_start, "spearman_cg_dot": rho})
                print(f"    [method-agreement] step {window_start}: "
                      f"Spearman(cg, dot) = {rho:+.3f}")

        schedule = build_schedule(
            regime,
            n_train=n_train,
            n_steps=window_len,
            batch_size=args.batch_size,
            scores=scores,
            selection_policy=args.selection_policy,
            softmax_temperature=args.softmax_temperature,
            cycle_offset=cycle_offset,
            rng_seed=args.seed + window_start,
        )

        for local_step, step in enumerate(range(window_start, window_end)):
            batch_idx = schedule[local_step * args.batch_size : (local_step + 1) * args.batch_size]
            batch_examples = [train_examples[i] for i in batch_idx]
            loss = grpo_batch_step(
                model,
                batch_examples,
                optimizer,
                beta=args.beta,
                ref_model=ref_model,
                seed=args.seed + step * 1000,
            )

            picks_log.append({
                "step": step + 1,
                "regime": regime,
                "picked_indices": batch_idx,
                "picked_names": [train_examples[i].name for i in batch_idx],
                "picked_if_scores": (
                    [float(scores[i]) for i in batch_idx] if scores is not None else None
                ),
                "loss": loss,
            })

            if (step + 1) % max(1, args.eval_every) == 0 or step + 1 == args.steps:
                per_test = [float(exact_expected_reward(model, te)) for te in test_examples]
                reward_log.append({
                    "step": step + 1,
                    "regime": regime,
                    "mean_test_reward": float(np.mean(per_test)),
                    "per_test_reward": per_test,
                })

        if regime == "round-robin":
            cycle_offset = (cycle_offset + window_len * args.batch_size) % n_train

    final_rewards = [float(exact_expected_reward(model, te)) for te in test_examples]
    summary = {
        "regime": regime,
        "label": label,
        "if_method": if_method if needs_if else None,
        "final_per_test_reward": final_rewards,
        "final_mean_reward": float(np.mean(final_rewards)),
    }

    regime_dir = run_dir / label.replace("[", "_").replace("]", "")
    regime_dir.mkdir(parents=True, exist_ok=True)
    with (regime_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (regime_dir / "picks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "picked_indices", "picked_names", "picked_if_scores"])
        for row in picks_log:
            writer.writerow([
                row["step"],
                f"{row['loss']:.6e}",
                ";".join(str(i) for i in row["picked_indices"]),
                ";".join(row["picked_names"]),
                (";".join(f"{x:.6e}" for x in row["picked_if_scores"])
                 if row["picked_if_scores"] is not None else ""),
            ])

    with (regime_dir / "rewards_over_time.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        n_test = len(test_examples)
        writer.writerow(["step", "mean_test_reward"] + [f"r_test_{i}" for i in range(n_test)])
        for row in reward_log:
            writer.writerow([row["step"], f"{row['mean_test_reward']:.6f}"] + row["per_test_reward"])

    if agreement_log:
        summary["mean_spearman_cg_dot"] = float(
            np.mean([r["spearman_cg_dot"] for r in agreement_log])
        )
        with (regime_dir / "method_agreement.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "spearman_cg_dot"])
            for row in agreement_log:
                writer.writerow([row["step"], f"{row['spearman_cg_dot']:.6f}"])

    print(f"  Final mean reward: {summary['final_mean_reward']:.4f}")
    return {
        "summary": summary,
        "reward_log": reward_log,
        "picks_log": picks_log,
        "agreement_log": agreement_log,
    }


def build_datasets(args) -> tuple[list[ToyGRPOExample], list[ToyGRPOExample]]:
    if args.dataset == "clustered":
        train_examples, test_examples, _, _, _ = generate_clustered_dataset(
            n_clusters=args.n_clusters,
            per_cluster=args.per_cluster,
            cluster_signal=args.cluster_signal,
            target_mode=args.cluster_target_mode,
            seed=args.seed,
        )
        args.n_train = len(train_examples)
        args.n_test = len(test_examples)
        return train_examples, test_examples

    all_examples = generate_dataset(n=args.n_train + 200, seed=args.seed)
    train_examples = all_examples[: args.n_train]
    hard_test: list[ToyGRPOExample] = []
    for ex in all_examples[args.n_train :]:
        if abs(ex.z[0] + ex.z[1] + ex.z[2]) < 0.3:
            hard_test.append(ex)
        if len(hard_test) >= args.n_test:
            break
    return train_examples, hard_test


def build_ref_model(hidden_dim: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    ref_model = ToyAutoregressiveMLP(hidden_dim=hidden_dim)
    for m in ref_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    ref_model.eval()
    return ref_model


def main():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--n-train", type=int, default=45)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--dataset", choices=["iid", "clustered"], default="iid")
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--per-cluster", type=int, default=3)
    parser.add_argument("--cluster-signal", type=float, default=3.0)
    parser.add_argument("--cluster-target-mode", choices=["parity", "random"], default="parity")

    # Training
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--no-adam", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Selection / recompute
    parser.add_argument("--regimes", type=str, default=",".join(REGIMES),
                        help=f"Comma-separated regimes to run. Choices: {REGIMES}")
    parser.add_argument("--if-methods", type=str, default="cg",
                        help="Comma-separated influence methods for IF regimes (cg, dot). "
                             "Pass 'cg,dot' to overlay both on the comparison plot.")
    parser.add_argument("--selection-policy", choices=list(SELECTION_POLICIES), default="top-k")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--if-recompute-mode", choices=["fixed", "log"], default="fixed")
    parser.add_argument("--if-recompute-every", type=int, default=None,
                        help="(fixed mode) recompute every K steps. Default: n_train // batch_size (one epoch).")

    # CG / IF hyperparams
    parser.add_argument("--cg-iters", type=int, default=50)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--lambda-damp", type=float, default=1.0)

    # Evaluation / output
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/if_pruning")

    args = parser.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    for r in regimes:
        if r not in REGIMES:
            raise SystemExit(f"Unknown regime: {r!r}. Choices: {REGIMES}")

    if_methods = [m.strip() for m in args.if_methods.split(",") if m.strip()]
    for m in if_methods:
        if m not in ("cg", "dot"):
            raise SystemExit(f"Unknown if-method: {m!r}. Choices: ('cg', 'dot')")

    train_examples, test_examples = build_datasets(args)
    print(f"Dataset: {args.dataset}, train={len(train_examples)}, test={len(test_examples)}")

    if args.if_recompute_every is None:
        args.if_recompute_every = max(1, args.n_train // args.batch_size)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Results: {run_dir}")

    ref_model = build_ref_model(args.hidden_dim, args.seed)

    # Build the run list: IF-dependent regimes run once per influence method
    # (so cg vs dot overlay on the plot); method-free baselines run once.
    multi = len(if_methods) > 1
    compare_methods = ("cg" in if_methods) and ("dot" in if_methods)
    jobs: list[tuple[str, str, str]] = []  # (regime, if_method, label)
    for regime in regimes:
        if regime in ("if-guided", "anti-if"):
            for m in if_methods:
                label = f"{regime}[{m}]" if multi else regime
                jobs.append((regime, m, label))
        else:
            jobs.append((regime, if_methods[0], regime))  # method irrelevant

    results: dict[str, dict] = {}
    for regime, if_method, label in jobs:
        results[label] = run_regime(
            regime,
            args=args,
            train_examples=train_examples,
            test_examples=test_examples,
            ref_model=ref_model,
            run_dir=run_dir,
            if_method=if_method,
            label=label,
            compare_methods=compare_methods,
        )

    with (run_dir / "summary.json").open("w") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f, indent=2)

    # Comparison plot (every regime×method curve overlaid).
    plt.figure(figsize=(10, 6))
    for label, res in results.items():
        rows = res["reward_log"]
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        rewards = [r["mean_test_reward"] for r in rows]
        plt.plot(steps, rewards, label=label, marker="o", markersize=3)
    plt.xlabel("Step")
    plt.ylabel("Mean test reward")
    plt.title(
        f"IF-pruning ({args.dataset}, B={args.batch_size}, sel={args.selection_policy}, "
        f"recompute={args.if_recompute_mode}/{args.if_recompute_every}, methods={','.join(if_methods)})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "comparison.png", dpi=150)
    plt.close()
    print(f"\nPlot: {run_dir / 'comparison.png'}")

    print("\n=== Final mean test reward by regime ===")
    for label, res in results.items():
        s = res["summary"]
        print(f"  {label:>18s}: {s['final_mean_reward']:.4f}  per-test={s['final_per_test_reward']}")

    if compare_methods:
        print("\n=== Method agreement: Spearman(cg, dot) over the pool ===")
        print("  (ρ≈1 → curvature doesn't reorder picks → Fisher == dot-product)")
        for label, res in results.items():
            mean_rho = res["summary"].get("mean_spearman_cg_dot")
            if mean_rho is not None:
                per_window = [f"{r['spearman_cg_dot']:+.3f}" for r in res["agreement_log"]]
                print(f"  {label:>18s}: mean ρ={mean_rho:+.3f}  per-recompute={per_window}")


if __name__ == "__main__":
    main()
