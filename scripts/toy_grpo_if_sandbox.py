from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from influence_rlvr.toy_grpo import (
    AutoregressiveLogisticRegression,
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
    args = build_parser().parse_args()
    sandbox = build_user_plan_sandbox()
    rollout_mode = ToyRolloutMode.parse(args.rollout_mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoregressiveLogisticRegression(use_bias=args.use_bias)
    initialize_toy_model(
        model,
        mode=args.init,
        seed=args.seed,
        scale=args.init_scale,
    )
    
    # accuracy before training 
    print("Initial model")
    for example in sandbox.train_examples:
        preview_g = 4 if rollout_mode == ToyRolloutMode.EXHAUSTIVE else 1
        response_ids = rollout_token_sequences(
            model,
            example,
            G=preview_g,
            rollout_mode=rollout_mode,
            seed=args.seed,
        )
        print(f"Prompt: {example.z}")
        print(f"Correct: {example.target}")
        print(f"Response: {sequence_labels(response_ids)[0]}")

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
    checkpoints = train_result["checkpoints"]
    checkpoint_steps = sorted(checkpoints)

    local_rows = []
    local_json = []
    historical_rows = []
    historical_json = []
    historical_totals_rows = []
    final_local = None
    final_historical = None

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

    print("Toy GRPO IF sandbox")
    print(
        f"  rollout_mode={rollout_mode.value} use_bias={args.use_bias} "
        f"init={args.init} steps={args.steps} lr={args.lr}"
    )
    print(
        f"  checkpoints=0..{args.steps} lambda_damp={args.lambda_damp} "
        f"epsilon={args.epsilon} beta={args.beta}"
    )
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
        output_dir / "local_if.csv",
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
        output_dir / "historical_if.csv",
        historical_rows,
        [
            "checkpoint_step",
            "train_name",
            "expected_sign",
            "count",
            "repo_fisher_score",
        ],
    )
    with (output_dir / "local_if.json").open("w") as handle:
        json.dump(local_json, handle, indent=2)
    with (output_dir / "historical_if.json").open("w") as handle:
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
                },
                "final_checkpoint_step": final_checkpoint_step,
                "local_if_csv": str(output_dir / "local_if.csv"),
                "historical_if_csv": str(output_dir / "historical_if.csv"),
            },
            handle,
            indent=2,
        )
    _plot_if_trajectories(
        local_rows,
        value_key="repo_fisher_score",
        title="Checkpoint-Local IF",
        ylabel="Repo Fisher Score",
        output_path=output_dir / "local_if_repo_score.png",
    )
    _plot_if_trajectories(
        historical_rows,
        value_key="repo_fisher_score",
        title="Historical Cumulative IF",
        ylabel="Repo Fisher Score",
        output_path=output_dir / "historical_if_repo_score.png",
    )
    _plot_total_deltas(
        historical_totals_rows,
        output_path=output_dir / "historical_total_deltas.png",
    )
    print()
    print("Saved trajectory files")
    print(f"  local IF by checkpoint: {output_dir / 'local_if.csv'}")
    print(f"  historical cumulative IF: {output_dir / 'historical_if.csv'}")
    print(f"  local IF plot: {output_dir / 'local_if_repo_score.png'}")
    print(f"  historical IF plot: {output_dir / 'historical_if_repo_score.png'}")
    print(f"  total delta plot: {output_dir / 'historical_total_deltas.png'}")
    print(f"  json summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
