import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/results_run1"
results = Path(RESULTS_DIR)

tracin = np.load(results / "tracin_matrix.npy")
datainf = np.load(results / "datainf_matrix.npy")
meta = json.loads((results / "metadata.json").read_text())

n_test, n_train = tracin.shape
test_prompts = meta.get("test_prompts", [f"test_{i}" for i in range(n_test)])
train_prompts = meta.get("train_prompts", [f"train_{j}" for j in range(n_train)])
steps = [cp["step"] for cp in meta["checkpoints"]]

test_labels = [f"T{i}" for i in range(n_test)]
train_labels = [f"R{j}" for j in range(n_train)]

out = results / "figures"
out.mkdir(exist_ok=True)


def _heatmap(mat, title, fname):
    fig, ax = plt.subplots(figsize=(max(6, n_train * 0.8), max(3, n_test * 0.7)))
    finite = mat[np.isfinite(mat)]
    vmax = np.max(np.abs(finite)) if finite.size else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n_train))
    ax.set_xticklabels(train_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_test))
    ax.set_yticklabels(test_labels, fontsize=8)
    ax.set_xlabel("Train samples (Math)")
    ax.set_ylabel("Test samples (Code)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=200)
    plt.close(fig)
    print(f"  saved {out / fname}")


_heatmap(tracin, "Trajectory TracIn Influence", "tracin_heatmap.png")
_heatmap(datainf, "Trajectory DataInf Influence", "datainf_heatmap.png")


def _topk_table(mat, method_name, k=3):
    lines = [f"\n{'='*60}", f"Top-{k} influential train samples per test ({method_name})", "=" * 60]
    for i in range(n_test):
        row = mat[i]
        order = np.argsort(-np.abs(row))[:k]
        prompt_short = textwrap.shorten(test_prompts[i], width=100, placeholder="...")
        lines.append(f"\nTest {i}: {prompt_short}")
        for rank, j in enumerate(order, start=1):
            train_short = textwrap.shorten(train_prompts[j], width=80, placeholder="...")
            lines.append(f"  #{rank}  train {j}: score={row[j]:+.3e}  | {train_short}")
    return "\n".join(lines)


report = _topk_table(tracin, "TracIn")
report += "\n" + _topk_table(datainf, "DataInf")


fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(tracin.flatten(), datainf.flatten(), alpha=0.7, edgecolors="k", linewidths=0.3, s=40)
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.set_xlabel("TracIn score")
ax.set_ylabel("DataInf score")
ax.set_title("TracIn vs DataInf agreement")
fig.tight_layout()
fig.savefig(out / "tracin_vs_datainf.png", dpi=200)
plt.close(fig)
print(f"  saved {out / 'tracin_vs_datainf.png'}")


def _load_step_series(prefix, ti, tj):
    vals = []
    for step in steps:
        fpath = results / f"{prefix}_step_{step}.npy"
        if fpath.exists():
            m = np.load(fpath)
            vals.append(m[ti, tj])
        else:
            vals.append(np.nan)
    return np.array(vals)


abs_tracin = np.abs(tracin)
top_pairs = []
for _ in range(min(3, tracin.size)):
    idx = np.unravel_index(np.nanargmax(abs_tracin), abs_tracin.shape)
    top_pairs.append(idx)
    abs_tracin[idx] = -1

fig, axes = plt.subplots(len(top_pairs), 1, figsize=(8, 3 * len(top_pairs)), sharex=True)
if len(top_pairs) == 1:
    axes = [axes]
for ax, (ti, tj) in zip(axes, top_pairs):
    tr_series = _load_step_series("tracin", ti, tj)
    di_series = _load_step_series("datainf", ti, tj)
    ax.plot(steps, tr_series, marker="o", markersize=4, label="TracIn", linewidth=1.5)
    ax.plot(steps, di_series, marker="s", markersize=4, label="DataInf", linewidth=1.5)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_ylabel("Per-step contribution")
    ax.set_title(f"Test {ti} ↔ Train {tj}")
    ax.legend(fontsize=7)
axes[-1].set_xlabel("Checkpoint step")
fig.suptitle("Trajectory influence over training steps (top pairs by |TracIn|)", fontsize=11)
fig.tight_layout()
fig.savefig(out / "trajectory_top_pairs.png", dpi=200)
plt.close(fig)
print(f"  saved {out / 'trajectory_top_pairs.png'}")


grad_norms_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for cp in meta["checkpoints"]:
    step = cp["step"]
    ax1.bar(step, cp["mean_test_grad_norm"], color="steelblue", width=0.7)
    ax2.bar(step, cp["mean_train_grad_norm"], color="coral", width=0.7)
ax1.set_xlabel("Checkpoint step")
ax1.set_ylabel("Mean ||g_test||")
ax1.set_title("Test gradient norms across checkpoints")
ax2.set_xlabel("Checkpoint step")
ax2.set_ylabel("Mean ||g_train||")
ax2.set_title("Train gradient norms across checkpoints")
grad_norms_fig.tight_layout()
grad_norms_fig.savefig(out / "gradient_norms.png", dpi=200)
plt.close(grad_norms_fig)
print(f"  saved {out / 'gradient_norms.png'}")


report += f"\n\n{'='*60}\nConfig summary\n{'='*60}"
for key in ["model_id", "learning_rate", "max_steps", "grpo_beta", "grpo_epsilon",
            "g_train", "g_test", "n_math", "n_code", "n_train_replay", "lambda_damp"]:
    report += f"\n  {key}: {meta.get(key)}"

report_path = out / "report.txt"
report_path.write_text(report)
print(f"  saved {report_path}")

print(report)
print(f"\nAll figures saved in {out.resolve()}/")
