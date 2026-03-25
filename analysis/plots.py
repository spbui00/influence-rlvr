from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


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
        tracin = np.asarray(entry["tracin"])
        datainf = np.asarray(entry["datainf"])
        axis.plot(steps, tracin, marker="o", markersize=4, label="TracIn", linewidth=1.5)
        axis.plot(steps, datainf, marker="s", markersize=4, label="DataInf", linewidth=1.5)
        axis.axhline(0, color="grey", linewidth=0.5)
        axis.set_ylabel("Per-step contribution")
        axis.set_title(str(entry["title"]))
        axis.legend(fontsize=7)

    axes[-1].set_xlabel("Checkpoint step")
    fig.suptitle(
        "Trajectory influence over training steps (top pairs by |TracIn|)",
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
