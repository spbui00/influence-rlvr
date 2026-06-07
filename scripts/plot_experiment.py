"""Overlay the 4-arm reward curves + per-window selected-category breakdown.

Top panel: held-out CS accuracy vs training step, one line per arm (the reward
curves). Bottom panels: for each if_prune arm, a stacked bar per recompute window
showing which DOMAINS the influence picked (the cross-domain transfer observable).

Reads, per run, from <output-root>/<run>/:
  - live_eval.csv                              (step, n, accuracy, per_category_json)
  - influence/selected_categories_step{N}.json ({domain: #unique picked})

Run locally after rsync-ing outputs back, or on the cluster:

    python scripts/plot_experiment.py \
        --runs exp_baseline,exp_ifg_dot,exp_anti_dot,exp_ifg_cg \
        --out experiment_comparison.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    csv_path = run_dir / "live_eval.csv"
    steps: list[int] = []
    acc: list[float] = []
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                acc.append(float(row["accuracy"]))
    return steps, acc


def load_categories(run_dir: Path) -> dict[int, dict[str, int]]:
    """step -> {domain: count} from influence/selected_categories_step{N}.json."""
    out: dict[int, dict[str, int]] = {}
    inf = run_dir / "influence"
    if inf.is_dir():
        for p in sorted(inf.glob("selected_categories_step*.json")):
            try:
                step = int(p.stem.replace("selected_categories_step", ""))
            except ValueError:
                continue
            with p.open() as f:
                out[step] = json.load(f)
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="exp_baseline,exp_ifg_dot,exp_anti_dot,exp_ifg_cg",
                    help="comma-separated run names under --output-root")
    ap.add_argument("--output-root", default="./outputs")
    ap.add_argument("--out", default="experiment_comparison.png")
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    root = Path(args.output_root).expanduser()
    curves = {r: load_curve(root / r) for r in runs}
    cats = {r: load_categories(root / r) for r in runs}
    cat_runs = [r for r in runs if cats[r]]
    n_bottom = len(cat_runs)

    fig = plt.figure(figsize=(max(11, 4 * max(1, n_bottom)), 9 if n_bottom else 5))
    gs = fig.add_gridspec(2 if n_bottom else 1, max(1, n_bottom))

    # ── Top: reward curves overlaid ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    for r in runs:
        s, a = curves[r]
        if s:
            ax0.plot(s, a, marker="o", markersize=4, label=r)
    ax0.set_xlabel("training step")
    ax0.set_ylabel("held-out CS accuracy")
    ax0.set_title("Reward curves — live held-out accuracy vs step")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # ── Bottom: per-window selected-category stacked bars (if_prune arms) ─────
    if n_bottom:
        all_cats = sorted({c for r in cat_runs for d in cats[r].values() for c in d})
        cmap = plt.get_cmap("tab10")
        color = {c: cmap(i % 10) for i, c in enumerate(all_cats)}
        for j, r in enumerate(cat_runs):
            ax = fig.add_subplot(gs[1, j])
            steps = list(cats[r])
            xs = [str(s) for s in steps]
            bottom = [0.0] * len(steps)
            for c in all_cats:
                vals = [float(cats[r][s].get(c, 0)) for s in steps]
                ax.bar(xs, vals, bottom=bottom, label=c, color=color[c])
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_title(f"{r}\nIF-selected domains / window")
            ax.set_xlabel("recompute step")
            ax.set_ylabel("# unique picked")
            if j == 0:
                ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\nFinal held-out accuracy:")
    for r in runs:
        s, a = curves[r]
        if a:
            print(f"  {r:>16}: {a[-1]:.4f}  (step {s[-1]}, {len(s)} eval points)")
        else:
            print(f"  {r:>16}: (no live_eval.csv yet)")


if __name__ == "__main__":
    main()
