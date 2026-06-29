"""Plot held-out eval curves and training-reward curves for one or more runs.

Eval curve  = accuracy vs step from outputs/<run>/live_eval.csv (the fair
              held-out comparison across arms).
Train curve = reward vs step from a checkpoint's trainer_state.json log_history,
              with frac_reward_zero_std on a twin axis (training-health read).

Only stdlib + matplotlib (no torch/pandas), so it runs in any plotting env.

    python -m experiments.plot_curves \
        --eval xdomain_phys_baseline_c:baseline xdomain_phys_gold_c:gold \
        --train outputs/rlvr-output/checkpoint-300/trainer_state.json:baseline \
        --outdir figures
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _split_label(spec, default):
    """'name:label' -> (name, label); 'name' -> (name, default(name))."""
    if ":" in spec:
        name, label = spec.rsplit(":", 1)
        return name, label
    return spec, default(spec)


def read_live_eval(run, output_root):
    steps, acc = [], []
    with (Path(output_root) / run / "live_eval.csv").open() as f:
        for r in csv.DictReader(f):
            steps.append(int(r["step"]))
            acc.append(float(r["accuracy"]))
    return steps, acc


def read_trainer_state(path, key="reward"):
    hist = json.loads(Path(path).read_text())["log_history"]
    steps = [h["step"] for h in hist if key in h]
    vals = [h[key] for h in hist if key in h]
    z = [(h["step"], h.get("frac_reward_zero_std")) for h in hist if key in h]
    return steps, vals, z


def smooth(xs, ys, w=9):
    if len(ys) < w:
        return xs, ys
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w // 2), min(len(ys), i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return xs, out


def plot_eval(specs, output_root, out_path):
    plt.figure(figsize=(8, 5))
    for spec in specs:
        run, label = _split_label(spec, lambda s: s)
        steps, acc = read_live_eval(run, output_root)
        plt.plot(steps, acc, marker="o", ms=3, lw=1.6, label=label)
    plt.xlabel("training step")
    plt.ylabel("held-out University-Physics accuracy (n=256)")
    plt.title("Held-out eval curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"wrote {out_path}")


def plot_train(specs, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    for spec in specs:
        path, label = _split_label(spec, lambda s: Path(s).parts[-2])
        steps, reward, z = read_trainer_state(path)
        ax.plot(steps, reward, lw=0.8, alpha=0.30)
        sx, sy = smooth(steps, reward)
        ln, = ax.plot(sx, sy, lw=2.0, label=f"{label} reward (smoothed)")
        zs = [s for s, v in z if v is not None]
        zv = [v for _, v in z if v is not None]
        if zv:
            ax2.plot(zs, zv, lw=1.0, ls="--", alpha=0.5, color=ln.get_color(),
                     label=f"{label} frac_reward_zero_std")
    ax.set_xlabel("training step")
    ax.set_ylabel("GRPO training reward")
    ax2.set_ylabel("frac_reward_zero_std (dashed)")
    ax2.set_ylim(0, 1)
    ax.set_title("Training-reward curve")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot eval/training curves.")
    p.add_argument("--eval", nargs="*", default=[],
                   help="run[:label] specs; reads outputs/<run>/live_eval.csv")
    p.add_argument("--train", nargs="*", default=[],
                   help="trainer_state.json path[:label] specs")
    p.add_argument("--output-root", default="./outputs")
    p.add_argument("--outdir", default="figures")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.eval:
        plot_eval(args.eval, args.output_root, outdir / "eval_curve.png")
    if args.train:
        plot_train(args.train, outdir / "train_curve.png")


if __name__ == "__main__":
    main()
