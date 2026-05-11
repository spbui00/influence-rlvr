from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt

from influence_rlvr.toy_grpo import (
    AutoregressiveLogisticRegression,
    ToyHistoricalWeightMode,
    ToyRolloutMode,
    build_user_plan_sandbox,
    clone_toy_model,
    compute_toy_fisher_influence,
    compute_toy_historical_fisher_influence,
    exact_expected_reward,
    initialize_toy_model,
    train_toy_grpo,
    rollout_token_sequences,
    sequence_labels
)


def _format_float(value: float) -> str:
    return f"{value:+.6f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the tiny autoregressive logistic-regression sandbox and compute "
            "Fisher influence with the repo implementation."
        )
    )
    parser.add_argument("--steps", type=int, default=12, help="Number of exact GRPO updates to run.")
    parser.add_argument("--lr", type=float, default=0.25, help="SGD learning rate for the toy loop.")
    parser.add_argument(
        "--rollout-mode",
        choices=[mode.value for mode in ToyRolloutMode],
        default=ToyRolloutMode.EXHAUSTIVE.value,
        help=(
            "Use `exhaustive` for the deterministic 4-sequence surrogate or `sampled` "
            "to match the repo's sampled-GRPO semantics more closely."
        ),
    )
    parser.add_argument(
        "--init",
        choices=["zero", "normal"],
        default="zero",
        help="Toy model initialization.",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.05,
        help="Stddev for `--init normal`.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for normal init and sampled rollouts.",
    )
    parser.add_argument(
        "--lambda-damp",
        type=float,
        default=0.1,
        help="Damping added to the toy Fisher matrix.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="GRPO clip epsilon.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL coefficient. The toy script expects 0 unless you add a ref model.",
    )
    parser.add_argument(
        "--use-bias",
        action="store_true",
        help="Enable biases in the toy AR-logreg model.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/toy_grpo_if",
        help="Directory where checkpoint-wise IF trajectories are written.",
    )
    parser.add_argument(
        "--historical-weight-mode",
        choices=[mode.value for mode in ToyHistoricalWeightMode],
        default=ToyHistoricalWeightMode.ACTIVE_ONLY.value,
        help=(
            "How historical trajectory Fisher weights train examples at each step: "
            "`active_only` keeps only the example actually used at that step; "
            "`all_samples` includes every train example at every step."
        ),
    )
    parser.add_argument(
        "--fisher-solver",
        choices=["woodbury", "full"],
        default="woodbury",
        help=(
            "Fisher inverse backend. `woodbury` is the existing n x n solve; "
            "`full` builds and inverts the full D x D Fisher matrix."
        ),
    )
    return parser


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_if_trajectories(
    rows: list[dict],
    *,
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    by_name: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_name.setdefault(str(row["train_name"]), []).append(
            (int(row["checkpoint_step"]), float(row[value_key]))
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for train_name, points in by_name.items():
        points.sort(key=lambda item: item[0])
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=train_name)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Checkpoint Step")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_total_deltas(
    rows: list[dict],
    *,
    output_path: Path,
) -> None:
    steps = [int(row["checkpoint_step"]) for row in rows]
    predicted_dloss = [float(row["predicted_total_dloss"]) for row in rows]
    actual_dloss = [float(row["actual_total_dloss"]) for row in rows]
    predicted_dreward = [float(row["predicted_total_dreward"]) for row in rows]
    actual_dreward = [float(row["actual_total_dreward"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(steps, predicted_dloss, marker="o", linewidth=1.8, label="Predicted")
    axes[0].plot(steps, actual_dloss, marker="s", linewidth=1.8, label="Actual")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_title("Total Test Loss Delta")
    axes[0].set_xlabel("Checkpoint Step")
    axes[0].set_ylabel("Delta")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, predicted_dreward, marker="o", linewidth=1.8, label="Predicted")
    axes[1].plot(steps, actual_dreward, marker="s", linewidth=1.8, label="Actual")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[1].set_title("Total Test Reward Delta")
    axes[1].set_xlabel("Checkpoint Step")
    axes[1].set_ylabel("Delta")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    start_time = time.time()
    args = build_parser().parse_args()
    sandbox = build_user_plan_sandbox()
    rollout_mode = ToyRolloutMode.parse(args.rollout_mode)
    historical_weight_mode = ToyHistoricalWeightMode.parse(args.historical_weight_mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    historical_suffix = historical_weight_mode.value
    solver_suffix = args.fisher_solver
    local_csv_path = output_dir / f"local_if_{solver_suffix}.csv"
    local_json_path = output_dir / f"local_if_{solver_suffix}.json"
    local_plot_path = output_dir / f"local_if_repo_score_{solver_suffix}.png"
    historical_csv_path = output_dir / f"historical_if_{historical_suffix}_{solver_suffix}.csv"
    historical_json_path = output_dir / f"historical_if_{historical_suffix}_{solver_suffix}.json"
    historical_plot_path = output_dir / f"historical_if_repo_score_{historical_suffix}_{solver_suffix}.png"
    historical_totals_plot_path = output_dir / f"historical_total_deltas_{historical_suffix}_{solver_suffix}.png"

    model = AutoregressiveLogisticRegression(use_bias=args.use_bias)
    initialize_toy_model(
        model,
        mode=args.init,
        seed=args.seed,
        scale=args.init_scale,
    )
    
    print("--- PRE-TRAINING STATS ---") 
    for example in sandbox.train_examples:
        acc = exact_expected_reward(model, example)
        print(f"Train example {example.name}: expected reward {acc:.6f}")
    test_acc = exact_expected_reward(model, sandbox.test_example)
    print(f"Test example {sandbox.test_example.name}: expected reward {test_acc:.6f}")
    print()

    train_result = train_toy_grpo(
        model,
        sandbox.train_examples,
        steps=args.steps,
        lr=args.lr,
        rollout_mode=rollout_mode,
        epsilon=args.epsilon,
        beta=args.beta,
        checkpoint_steps=range(args.steps + 1),
        seed=args.seed,
    )

    print("--- TRAINING COMPLETE ---")
    print(f"Total training time: {time.time() - start_time:.2f} seconds\n")

    print("--- AFTER-TRAINING STATS ---")
    for example in sandbox.train_examples:
        acc = exact_expected_reward(model, example)
        print(f"Train example {example.name}: expected reward {acc:.6f}")
    test_acc = exact_expected_reward(model, sandbox.test_example)
    print(f"Test example {sandbox.test_example.name}: expected reward {test_acc:.6f}")
    print()

    checkpoints = train_result["checkpoints"]
    checkpoint_steps = sorted(checkpoints)

    local_rows = []
    local_json = []
    historical_rows = []
    historical_json = []
    historical_totals_rows = []
    final_local = None
    final_historical = None

    if_timer = time.time()

    for checkpoint_step in checkpoint_steps:
        checkpoint_model = clone_toy_model(model)
        checkpoint_model.load_state_dict(checkpoints[checkpoint_step])

        # for each cp compute fisher IF
        local = compute_toy_fisher_influence(
            checkpoint_model,
            train_examples=sandbox.train_examples,
            test_example=sandbox.test_example,
            lambda_damp=args.lambda_damp,
            rollout_mode=rollout_mode,
            epsilon=args.epsilon,
            beta=args.beta,
            seed=args.seed,
            fisher_solver=args.fisher_solver,
        )
        if checkpoint_step == checkpoint_steps[-1]:
            final_local = local

        local_scores = []
        for example, repo_score in zip(
            sandbox.train_examples,
            local["repo_scores"],
        ):
            reward = exact_expected_reward(checkpoint_model, example)
            row = {
                "checkpoint_step": checkpoint_step,
                "train_name": example.name,
                "expected_sign": example.expected_influence,
                "repo_fisher_score": float(repo_score),
                "train_expected_reward": float(reward),
            }
            local_rows.append(row)
            local_scores.append(row)
        local_json.append(
            {
                "checkpoint_step": checkpoint_step,
                "test_expected_reward": float(
                    exact_expected_reward(checkpoint_model, sandbox.test_example)
                ),
                "scores": local_scores,
            }
        )

        if checkpoint_step == 0:
            occurrence_count = {example.name: 0 for example in sandbox.train_examples}
            scores = []
            for example in sandbox.train_examples:
                row = {
                    "checkpoint_step": 0,
                    "train_name": example.name,
                    "expected_sign": example.expected_influence,
                    "count": 0,
                    "repo_fisher_score": 0.0,
                }
                historical_rows.append(row)
                scores.append(row)
            historical_json.append(
                {
                    "checkpoint_step": 0,
                    "predicted_total_dloss": 0.0,
                    "actual_total_dloss": 0.0,
                    "predicted_total_dreward": 0.0,
                    "actual_total_dreward": 0.0,
                    "scores": scores,
                }
            )
            historical_totals_rows.append(
                {
                    "checkpoint_step": 0,
                    "predicted_total_dloss": 0.0,
                    "actual_total_dloss": 0.0,
                    "predicted_total_dreward": 0.0,
                    "actual_total_dreward": 0.0,
                }
            )
            continue

        historical = compute_toy_historical_fisher_influence(
            checkpoint_model,
            checkpoints=checkpoints,
            train_history=train_result["history"],
            train_examples=sandbox.train_examples,
            test_example=sandbox.test_example,
            learning_rate=args.lr,
            end_step=checkpoint_step,
            lambda_damp=args.lambda_damp,
            rollout_mode=rollout_mode,
            epsilon=args.epsilon,
            beta=args.beta,
            seed=args.seed,
            historical_weight_mode=historical_weight_mode,
            fisher_solver=args.fisher_solver,
        )
        if checkpoint_step == checkpoint_steps[-1]:
            final_historical = historical

        scores = []
        for row in historical["historical_scores"]:
            out_row = {
                "checkpoint_step": checkpoint_step,
                "train_name": row.train_name,
                "expected_sign": row.expected_influence,
                "count": row.occurrence_count,
                "repo_fisher_score": row.repo_fisher_score,
            }
            historical_rows.append(out_row)
            scores.append(out_row)
        historical_json.append(
            {
                "checkpoint_step": checkpoint_step,
                "predicted_total_dloss": historical["predicted_total_loss_delta"],
                "actual_total_dloss": historical["actual_total_loss_delta"],
                "predicted_total_dreward": historical["predicted_total_reward_delta"],
                "actual_total_dreward": historical["actual_total_reward_delta"],
                "scores": scores,
            }
        )
        historical_totals_rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "predicted_total_dloss": historical["predicted_total_loss_delta"],
                "actual_total_dloss": historical["actual_total_loss_delta"],
                "predicted_total_dreward": historical["predicted_total_reward_delta"],
                "actual_total_dreward": historical["actual_total_reward_delta"],
            }
        )

    print(f"Computed IF trajectories for {len(checkpoint_steps)} checkpoints in {time.time() - if_timer:.2f} seconds\n")

    print("Toy GRPO IF sandbox")
    print(
        f"  rollout_mode={rollout_mode.value} use_bias={args.use_bias} "
        f"init={args.init} steps={args.steps} lr={args.lr}"
    )
    print(
        f"  checkpoints=0..{args.steps} lambda_damp={args.lambda_damp} "
        f"epsilon={args.epsilon} beta={args.beta}"
    )
    print(f"  historical_weight_mode={historical_weight_mode.value}")
    print(f"  fisher_solver={args.fisher_solver}")
    print(f"  output_dir={output_dir}")
    print()

    final_checkpoint_step = checkpoint_steps[-1]
    final_model = clone_toy_model(model)
    final_model.load_state_dict(checkpoints[final_checkpoint_step])

    print(f"Exact expected rewards at the final checkpoint ({final_checkpoint_step})")
    for example in sandbox.train_examples:
        print(
            f"  train::{example.name:24s} z={list(example.z)} "
            f"target={list(example.target)} reward={exact_expected_reward(final_model, example):.6f}"
        )
    print(
        f"  test::{sandbox.test_example.name:25s} z={list(sandbox.test_example.z)} "
        f"target={list(sandbox.test_example.target)} "
        f"reward={exact_expected_reward(final_model, sandbox.test_example):.6f}"
    )
    print()

    if final_local is None:
        raise RuntimeError("Local IF trajectory computation produced no checkpoints.")

    dense_gap = max(
        abs(float(repo) - float(dense))
        for repo, dense in zip(final_local["repo_scores"], final_local["dense_repo_scores"])
    )
    print(f"Repo Fisher vs explicit dense solve max abs diff at final checkpoint: {dense_gap:.6e}")
    print()

    print(f"Checkpoint-local IF at final checkpoint ({final_checkpoint_step})")
    print("  repo_fisher_score > 0 means helpful in the repo convention.")
    for example, repo_score in zip(
        sandbox.train_examples,
        final_local["repo_scores"],
    ):
        print(
            f"  {example.name:24s} expected={str(example.expected_influence):>18s} "
            f"repo_fisher_score={_format_float(float(repo_score))}"
        )
    print()

    if final_historical is None:
        raise RuntimeError("Historical IF trajectory computation produced no positive-step checkpoints.")

    print(f"Historical trajectory Fisher up to final step ({final_checkpoint_step})")
    print("  repo_fisher_score > 0 still means helpful in the repo convention.")
    if historical_weight_mode == ToyHistoricalWeightMode.ALL_SAMPLES:
        print("  all_samples mode: every train sample contributes at every training step.")
    else:
        print("  active_only mode: only the actually used train sample contributes at each step.")
    for row in final_historical["historical_scores"]:
        print(
            f"  {row.train_name:24s} count={row.occurrence_count:2d} "
            f"expected={str(row.expected_influence):>18s} "
            f"repo_fisher_score={_format_float(row.repo_fisher_score)}"
        )
    print(
        f"  predicted_total_dloss={_format_float(final_historical['predicted_total_loss_delta'])} "
        f"actual_total_dloss={_format_float(final_historical['actual_total_loss_delta'])}"
    )
    print(
        f"  predicted_total_dreward={_format_float(final_historical['predicted_total_reward_delta'])} "
        f"actual_total_dreward={_format_float(final_historical['actual_total_reward_delta'])}"
    )

    _write_csv(
        local_csv_path,
        local_rows,
        [
            "checkpoint_step",
            "train_name",
            "expected_sign",
            "repo_fisher_score",
            "train_expected_reward",
        ],
    )
    _write_csv(
        historical_csv_path,
        historical_rows,
        [
            "checkpoint_step",
            "train_name",
            "expected_sign",
            "count",
            "repo_fisher_score",
        ],
    )
    with local_json_path.open("w") as handle:
        json.dump(local_json, handle, indent=2)
    with historical_json_path.open("w") as handle:
        json.dump(historical_json, handle, indent=2)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "config": {
                    "steps": args.steps,
                    "lr": args.lr,
                    "rollout_mode": rollout_mode.value,
                    "init": args.init,
                    "init_scale": args.init_scale,
                    "seed": args.seed,
                    "lambda_damp": args.lambda_damp,
                    "epsilon": args.epsilon,
                    "beta": args.beta,
                    "use_bias": args.use_bias,
                    "historical_weight_mode": historical_weight_mode.value,
                    "fisher_solver": args.fisher_solver,
                },
                "final_checkpoint_step": final_checkpoint_step,
                "local_if_csv": str(local_csv_path),
                "historical_if_csv": str(historical_csv_path),
            },
            handle,
            indent=2,
        )
    _plot_if_trajectories(
        local_rows,
        value_key="repo_fisher_score",
        title="Checkpoint-Local IF",
        ylabel="Repo Fisher Score",
        output_path=local_plot_path,
    )
    _plot_if_trajectories(
        historical_rows,
        value_key="repo_fisher_score",
        title="Historical Cumulative IF",
        ylabel="Repo Fisher Score",
        output_path=historical_plot_path,
    )
    _plot_total_deltas(
        historical_totals_rows,
        output_path=historical_totals_plot_path,
    )
    print()
    print("Saved trajectory files")
    print(f"  local IF by checkpoint: {local_csv_path}")
    print(f"  historical cumulative IF: {historical_csv_path}")
    print(f"  local IF plot: {local_plot_path}")
    print(f"  historical IF plot: {historical_plot_path}")
    print(f"  total delta plot: {historical_totals_plot_path}")
    print(f"  json summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
