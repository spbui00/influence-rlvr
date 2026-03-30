from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _history_entries(log_history: list[dict[str, object]]):
    entries = []
    for item in log_history:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if step is None:
            continue
        try:
            step_value = int(step)
        except (TypeError, ValueError):
            continue
        entries.append((step_value, item))
    entries.sort(key=lambda pair: pair[0])
    return entries


def _history_metric(entries: list[tuple[int, dict[str, object]]], key: str) -> np.ndarray:
    values = []
    for _, item in entries:
        value = item.get(key, np.nan)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _reward_component_series(entries: list[tuple[int, dict[str, object]]]) -> dict[str, np.ndarray]:
    metric_keys = []
    for _, item in entries:
        for key in item:
            if key.startswith("rewards/") and key.endswith("/mean") and key not in metric_keys:
                metric_keys.append(key)

    series = {}
    for key in metric_keys:
        label = key[len("rewards/"):-len("/mean")]
        label = label.replace("_func", "").replace("_", " ").strip().title()
        series[label] = _history_metric(entries, key)
    return series


def _plot_sparse_series(
    axis,
    steps: list[int],
    values: np.ndarray,
    *,
    label: str,
    color: str,
    linewidth: float = 1.5,
    linestyle: str = "-",
    marker: str = "o",
):
    mask = np.isfinite(values)
    if not np.any(mask):
        return
    xs = np.asarray(steps, dtype=float)[mask]
    ys = values[mask]
    axis.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        marker=marker,
        markersize=4,
        label=label,
    )


def heatmap_figure(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
):
    fig, ax = plt.subplots(
        figsize=(max(6, len(col_labels) * 0.8), max(3, len(row_labels) * 0.7))
    )
    finite = matrix[np.isfinite(matrix)]
    vmax = np.max(np.abs(finite)) if finite.size else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def agreement_scatter_figure(tracin: np.ndarray, datainf: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(
        tracin.flatten(),
        datainf.flatten(),
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
        s=40,
    )
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("TracIn score")
    ax.set_ylabel("DataInf score")
    ax.set_title("TracIn vs DataInf agreement")
    fig.tight_layout()
    return fig


def trajectory_pairs_figure(
    steps: list[int],
    pair_series: list[dict[str, object]],
):
    fig, axes = plt.subplots(
        len(pair_series),
        1,
        figsize=(8, 3 * max(1, len(pair_series))),
        sharex=True,
    )
    if len(pair_series) == 1:
        axes = [axes]

    for axis, entry in zip(axes, pair_series):
        line_specs = [
            ("tracin", "TracIn", "o"),
            ("datainf", "DataInf", "s"),
            ("fisher", "Fisher", "^"),
        ]
        for key, label, marker in line_specs:
            if key not in entry:
                continue
            values = np.asarray(entry[key])
            axis.plot(steps, values, marker=marker, markersize=4, label=label, linewidth=1.5)
        axis.axhline(0, color="grey", linewidth=0.5)
        axis.set_ylabel("Per-step contribution")
        axis.set_title(str(entry["title"]))
        axis.legend(fontsize=7)

    axes[-1].set_xlabel("Checkpoint step")
    fig.suptitle(
        "Trajectory influence over training steps",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def gradient_norms_figure(checkpoints: list[dict[str, float]]):
    fig, (axis_test, axis_train) = plt.subplots(1, 2, figsize=(12, 4))
    for checkpoint in checkpoints:
        step = checkpoint["step"]
        axis_test.bar(step, checkpoint["mean_test_grad_norm"], color="steelblue", width=0.7)
        axis_train.bar(step, checkpoint["mean_train_grad_norm"], color="coral", width=0.7)
    axis_test.set_xlabel("Checkpoint step")
    axis_test.set_ylabel("Mean ||g_test||")
    axis_test.set_title("Test gradient norms across checkpoints")
    axis_train.set_xlabel("Checkpoint step")
    axis_train.set_ylabel("Mean ||g_train||")
    axis_train.set_title("Train gradient norms across checkpoints")
    fig.tight_layout()
    return fig


def eval_performance_figure(checkpoints: list[dict[str, object]]):
    steps = [checkpoint["step"] for checkpoint in checkpoints]
    fig, (axis_math, axis_code) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    math_accuracy = [
        checkpoint.get("math_eval", {}).get("accuracy_rate", np.nan)
        if checkpoint.get("math_eval") is not None else np.nan
        for checkpoint in checkpoints
    ]
    code_pass = [
        checkpoint.get("code_eval", {}).get("pass_rate", np.nan)
        if checkpoint.get("code_eval") is not None else np.nan
        for checkpoint in checkpoints
    ]
    code_compile = [
        checkpoint.get("code_eval", {}).get("compile_rate", np.nan)
        if checkpoint.get("code_eval") is not None else np.nan
        for checkpoint in checkpoints
    ]
    latest_code_eval = next(
        (checkpoint.get("code_eval") for checkpoint in reversed(checkpoints) if checkpoint.get("code_eval") is not None),
        None,
    )
    pass_label = "Code pass"
    compile_label = "Code compile"
    if latest_code_eval is not None:
        pass_label = str(latest_code_eval.get("pass_metric", pass_label)).replace("_", " ")
        compile_label = str(latest_code_eval.get("compile_metric", compile_label)).replace("_", " ")

    axis_math.plot(steps, math_accuracy, marker="o", linewidth=1.5, label="Math exact")
    axis_math.set_ylim(-0.02, 1.02)
    axis_math.set_xlabel("Checkpoint step")
    axis_math.set_ylabel("Rate")
    axis_math.set_title("Held-out math performance")
    axis_math.legend(fontsize=8)

    axis_code.plot(steps, code_pass, marker="o", linewidth=1.5, label=pass_label)
    axis_code.plot(steps, code_compile, marker="s", linewidth=1.5, label=compile_label)
    axis_code.set_ylim(-0.02, 1.02)
    axis_code.set_xlabel("Checkpoint step")
    axis_code.set_ylabel("Rate")
    axis_code.set_title("Held-out code performance")
    axis_code.legend(fontsize=8)

    if len(steps) > 1 and np.isfinite(math_accuracy[0]) and np.isfinite(math_accuracy[-1]):
        math_delta = math_accuracy[-1] - math_accuracy[0]
        axis_math.text(
            0.02,
            0.04,
            f"Delta exact: {math_delta:+.3f}",
            transform=axis_math.transAxes,
            fontsize=8,
            va="bottom",
        )

    if len(steps) > 1 and np.isfinite(code_pass[0]) and np.isfinite(code_pass[-1]):
        code_delta = code_pass[-1] - code_pass[0]
        compile_delta = (
            code_compile[-1] - code_compile[0]
            if np.isfinite(code_compile[0]) and np.isfinite(code_compile[-1])
            else np.nan
        )
        axis_code.text(
            0.02,
            0.04,
            (
                f"Delta {pass_label}: {code_delta:+.3f}\n"
                f"Delta {compile_label}: {compile_delta:+.3f}"
                if np.isfinite(compile_delta)
                else f"Delta {pass_label}: {code_delta:+.3f}"
            ),
            transform=axis_code.transAxes,
            fontsize=8,
            va="bottom",
        )

    fig.tight_layout()
    return fig


def training_curves_figure(log_history: list[dict[str, object]]):
    entries = _history_entries(log_history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    if not entries:
        for axis in axes.flat:
            axis.text(0.5, 0.5, "No training log history found.", ha="center", va="center")
            axis.set_axis_off()
        fig.tight_layout()
        return fig

    steps = [step for step, _ in entries]
    loss = _history_metric(entries, "loss")
    reward = _history_metric(entries, "reward")
    reward_std = _history_metric(entries, "reward_std")
    reward_components = _reward_component_series(entries)
    grad_norm = _history_metric(entries, "grad_norm")
    kl = _history_metric(entries, "kl")

    axis_loss = axes[0, 0]
    axis_reward = axes[0, 1]
    axis_components = axes[1, 0]
    axis_opt = axes[1, 1]

    axis_loss.plot(steps, loss, color="steelblue", linewidth=1.5)
    axis_loss.set_title("GRPO loss")
    axis_loss.set_ylabel("Loss")
    axis_loss.axhline(0, color="grey", linewidth=0.5)

    _plot_sparse_series(
        axis_reward,
        steps,
        reward,
        label="Reward",
        color="seagreen",
    )
    _plot_sparse_series(
        axis_reward,
        steps,
        reward_std,
        label="Reward std",
        color="darkorange",
        linewidth=1.2,
        linestyle="--",
        marker="s",
    )
    axis_reward.set_title("Training reward")
    axis_reward.set_ylabel("Value")
    axis_reward.legend(fontsize=8)

    component_colors = ["purple", "brown", "teal", "darkgoldenrod"]
    for (label, values), color in zip(reward_components.items(), component_colors):
        _plot_sparse_series(
            axis_components,
            steps,
            values,
            label=label,
            color=color,
        )
    axis_components.set_title("Reward components")
    axis_components.set_xlabel("Training step")
    axis_components.set_ylabel("Mean reward")
    axis_components.set_ylim(-0.02, 1.02)
    axis_components.axhline(0, color="grey", linewidth=0.5)
    if reward_components:
        axis_components.legend(fontsize=8)
    else:
        axis_components.text(0.5, 0.5, "No reward component logs found.", ha="center", va="center")

    axis_opt.plot(steps, grad_norm, color="coral", linewidth=1.5, label="Grad norm")
    axis_opt.set_title("Optimization stats")
    axis_opt.set_xlabel("Training step")
    axis_opt.set_ylabel("Grad norm", color="coral")
    axis_opt.tick_params(axis="y", labelcolor="coral")
    axis_opt_kl = axis_opt.twinx()
    axis_opt_kl.plot(steps, kl, color="slateblue", linewidth=1.2, label="KL")
    axis_opt_kl.set_ylabel("KL", color="slateblue")
    axis_opt_kl.tick_params(axis="y", labelcolor="slateblue")
    opt_handles = axis_opt.get_lines() + axis_opt_kl.get_lines()
    axis_opt.legend(opt_handles, [line.get_label() for line in opt_handles], fontsize=8)

    for axis in axes[0]:
        axis.set_xlabel("Training step")
    fig.tight_layout()
    return fig
