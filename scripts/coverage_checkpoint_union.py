#!/usr/bin/env python3
"""
Per replay train index: take checkpoints whose historical batch included that index,
LR-thin that sub-schedule to a fixed target count, then union across all samples.

Loads influence_rlvr/checkpoint_schedule.py via importlib (no full package import).

Example (only rlvr-output required for full-history analysis):
  uv run python scripts/coverage_checkpoint_union.py \\
    --rlvr-output outputs/run7/rlvr-output \\
    --per-sample-target 20 \\
    --plot figures/coverage_union.png \\
    --json-out figures/coverage_union.json

Use --results-dir path/to/results1 to restrict to dataset_train_index from that bundle's
results_manifest.json. JSON and stdout report: how many indices have ≥1 matched checkpoint,
per-sample history step/inclusion counts, per-checkpoint counts of indices (before/after thin),
and checkpoint-thin stats. With --plot, also writes <stem>_ckpt_frequency.png (bars: checkpoint vs #indices).
Learning rate: omitted -> run_config.json, else results_manifest, else trainer_state, else 5e-5.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_MANIFEST_FILE = "results_manifest.json"
TRAIN_BATCH_HISTORY_FILE = "historical_batch_history.json"
RUN_CONFIG_FILE = "run_config.json"


def _load_checkpoint_schedule_module():
    path = ROOT / "influence_rlvr" / "checkpoint_schedule.py"
    spec = importlib.util.spec_from_file_location("_coverage_cksched", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


CKS = _load_checkpoint_schedule_module()
build_checkpoint_schedule = CKS.build_checkpoint_schedule
list_checkpoint_dirs = CKS.list_checkpoint_dirs
thin_checkpoint_schedule = CKS.thin_checkpoint_schedule
CheckpointThinningConfig = CKS.CheckpointThinningConfig
CheckpointThinningMode = CKS.CheckpointThinningMode


class HistStep:
    __slots__ = ("step", "train_index_counts")

    def __init__(self, step: int, train_index_counts: dict[int, int]):
        self.step = int(step)
        self.train_index_counts = train_index_counts


def load_historical_steps(rlvr_output: Path) -> list[HistStep]:
    path = rlvr_output / TRAIN_BATCH_HISTORY_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    data = json.loads(path.read_text())
    out: list[HistStep] = []
    for item in data.get("steps", []):
        tic = item.get("train_index_counts") or {}
        counts = {int(k): int(v) for k, v in tic.items()}
        out.append(HistStep(int(item["step"]), counts))
    return out


def _schedule_by_step(schedule: list[dict]) -> dict[int, dict]:
    return {int(cp["step"]): cp for cp in schedule}


def _resolve_checkpoint_for_history_step(
    by_step: dict[int, dict], hist_step: int
) -> dict | None:
    if hist_step in by_step:
        return by_step[hist_step]
    if hist_step + 1 in by_step:
        return by_step[hist_step + 1]
    if hist_step - 1 in by_step:
        return by_step[hist_step - 1]
    return None


def _historical_steps_covering_index(steps: list[HistStep], train_index: int) -> list[int]:
    out = []
    for st in steps:
        if train_index in st.train_index_counts:
            out.append(int(st.step))
    return sorted(set(out))


def _per_checkpoint_sample_counts(
    per_sample: dict[str, dict],
) -> tuple[Counter[int], Counter[int], list[int]]:
    before: Counter[int] = Counter()
    after: Counter[int] = Counter()
    for v in per_sample.values():
        for s in v["covering_steps"]:
            before[int(s)] += 1
        for s in v["thinned_steps"]:
            after[int(s)] += 1
    all_steps = sorted(set(before.keys()) | set(after.keys()))
    return before, after, all_steps


def _plot_checkpoint_frequency_bars(
    all_steps: list[int],
    before: Counter[int],
    after: Counter[int],
    tc: int,
    n_train: int,
):
    import matplotlib.pyplot as plt

    y_b = [before.get(s, 0) for s in all_steps]
    y_a = [after.get(s, 0) for s in all_steps]
    n = len(all_steps)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True, sharex=True)
    xpos = np.arange(n, dtype=float)
    for ax, ys, title in (
        (
            axes[0],
            y_b,
            "Before LR-thin: per checkpoint, how many train indices matched it (±1)",
        ),
        (
            axes[1],
            y_a,
            f"After LR-thin (cap={tc}): per checkpoint, how many indices kept it",
        ),
    ):
        if n == 0:
            ax.text(0.5, 0.5, "No checkpoints", ha="center", va="center")
            ax.set_title(title)
            continue
        ax.bar(xpos, ys, width=0.85, edgecolor="black", alpha=0.88)
        ax.set_ylabel(f"# indices (of {n_train})")
        ax.set_title(title)
        ymax = max(ys) if ys else 1
        ax.set_ylim(0, max(1.05, ymax * 1.08))
    axes[1].set_xlabel("Checkpoint global_step (sorted)")
    if n > 0:
        if n <= 48:
            tick_step = 1
        else:
            tick_step = max(1, n // 48)
        tick_idx = list(range(0, n, tick_step))
        axes[1].set_xticks(xpos[tick_idx])
        axes[1].set_xticklabels([str(all_steps[i]) for i in tick_idx], rotation=45, ha="right")
    return fig


def _history_batch_stats_for_index(
    steps: list[HistStep], train_index: int
) -> tuple[int, int]:
    distinct_steps = 0
    total_inclusions = 0
    for st in steps:
        c = st.train_index_counts.get(train_index)
        if c:
            distinct_steps += 1
            total_inclusions += int(c)
    return distinct_steps, total_inclusions


def _covering_checkpoint_list(
    hist_steps: list[HistStep],
    by_step: dict[int, dict],
    train_index: int,
) -> list[dict]:
    seen: dict[int, dict] = {}
    for S in _historical_steps_covering_index(hist_steps, train_index):
        cp = _resolve_checkpoint_for_history_step(by_step, S)
        if cp is None:
            continue
        sid = int(cp["step"])
        if sid not in seen:
            seen[sid] = cp
    return [seen[k] for k in sorted(seen.keys())]


def _load_train_indices_from_results(results_dir: Path) -> list[int]:
    path = results_dir / RESULTS_MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    data = json.loads(path.read_text())
    out: list[int] = []
    for s in data.get("train_samples", []):
        raw = s.get("dataset_train_index")
        if raw is None:
            raw = s.get("train_index")
        if raw is not None:
            try:
                out.append(int(raw))
            except (TypeError, ValueError):
                continue
    return out


def _parse_train_indices(raw: str | None) -> list[int]:
    if not raw or not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _replay_subset_indices(pool_size: int, n: int, subset_seed: int) -> list[int]:
    rng = random.Random(int(subset_seed))
    k = min(int(n), int(pool_size))
    if k <= 0:
        return []
    return sorted(rng.sample(range(int(pool_size)), k))


def _all_train_indices_from_history(hist_steps: list[HistStep]) -> list[int]:
    seen: set[int] = set()
    for st in hist_steps:
        seen.update(st.train_index_counts.keys())
    return sorted(seen)


def _infer_learning_rate(
    rlvr: Path, results_dir: Path | None
) -> tuple[float, str]:
    rc = rlvr / RUN_CONFIG_FILE
    if rc.is_file():
        try:
            data = json.loads(rc.read_text())
            if data.get("learning_rate") is not None:
                return float(data["learning_rate"]), RUN_CONFIG_FILE
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if results_dir is not None:
        rm = results_dir / RESULTS_MANIFEST_FILE
        if rm.is_file():
            try:
                payload = json.loads(rm.read_text())
                cfg = payload.get("config") or {}
                if cfg.get("learning_rate") is not None:
                    return float(cfg["learning_rate"]), f"{RESULTS_MANIFEST_FILE} config"
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    try:
        for d in list_checkpoint_dirs(str(rlvr)):
            ts_path = Path(d) / "trainer_state.json"
            if not ts_path.is_file():
                continue
            ts = json.loads(ts_path.read_text())
            for item in ts.get("log_history") or []:
                if isinstance(item, dict) and item.get("learning_rate") is not None:
                    return float(item["learning_rate"]), "trainer_state.json log_history"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return 5e-5, "default"


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "LR-thin checkpoints per training sample (covering batches only), "
            "then union selected steps and plot."
        )
    )
    p.add_argument(
        "--rlvr-output",
        type=Path,
        required=True,
        help="Training output dir (checkpoints + historical_batch_history.json)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "LR fallback for schedule building. If omitted: read run_config.json, "
            "else results_manifest config, else trainer_state log_history, else 5e-5."
        ),
    )
    p.add_argument(
        "--per-sample-target",
        type=int,
        default=20,
        help="LR-thin target count within each sample's covering checkpoint list",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            f"Optional: restrict to train indices from {RESULTS_MANIFEST_FILE} "
            "(same replay subset as an influence run). If you omit this, --train-indices, "
            "and replay-subset args, every index seen in historical_batch_history is used."
        ),
    )
    p.add_argument(
        "--train-indices",
        default=None,
        help="Comma-separated GSM8K train indices (overrides --results-dir if set)",
    )
    p.add_argument(
        "--replay-pool-size",
        type=int,
        default=None,
        help="With --replay-n/--replay-subset-seed: full math train pool size",
    )
    p.add_argument(
        "--replay-n",
        type=int,
        default=None,
        help="Replay subset size (same as N_TRAIN_REPLAY)",
    )
    p.add_argument(
        "--replay-subset-seed",
        type=int,
        default=None,
        help="Seed for replay subset (else train_grad_seed + 902891 if --train-grad-seed)",
    )
    p.add_argument(
        "--train-grad-seed",
        type=int,
        default=None,
        help="Used with replay subset when replay-subset-seed omitted (seed + 902891)",
    )
    p.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Write matplotlib figure (union stem + per-sample scatter) to this path",
    )
    p.add_argument(
        "--plot-covered-hist",
        type=Path,
        default=None,
        help=(
            "Histogram: for covered samples only, distribution of (1) # matched checkpoints "
            "before thin, (2) # after LR-thin. Default: next to --plot as <stem>_covered_hist.png "
            "when --plot is set; omit to skip unless this flag is set explicitly."
        ),
    )
    p.add_argument(
        "--plot-checkpoint-frequency",
        type=Path,
        default=None,
        help=(
            "Bar chart: x = each distinct checkpoint global_step (sorted), y = how many train "
            "indices include that checkpoint before thin vs after thin (two panels). "
            "Default: next to --plot as <stem>_ckpt_frequency.png when --plot is set."
        ),
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write summary JSON",
    )
    p.add_argument(
        "--max-plot-samples",
        type=int,
        default=120,
        help="Max training samples (rows) in per-sample scatter subplot",
    )
    args = p.parse_args()

    rlvr = args.rlvr_output.expanduser().resolve()
    results_resolved = (
        args.results_dir.expanduser().resolve() if args.results_dir else None
    )
    if args.learning_rate is not None:
        learning_rate = float(args.learning_rate)
        learning_rate_source = "cli"
    else:
        learning_rate, learning_rate_source = _infer_learning_rate(rlvr, results_resolved)
    print(
        f"Using learning_rate={learning_rate:g} (source: {learning_rate_source})",
        flush=True,
    )

    hist_steps = load_historical_steps(rlvr)
    if not hist_steps:
        raise SystemExit(f"No steps in {rlvr / TRAIN_BATCH_HISTORY_FILE}")

    full_schedule = build_checkpoint_schedule(str(rlvr), learning_rate)
    if not full_schedule:
        raise SystemExit(f"No checkpoints found under {rlvr}")
    by_step = _schedule_by_step(full_schedule)

    explicit = _parse_train_indices(args.train_indices)
    if explicit:
        train_indices = explicit
        train_indices_source = "cli_train_indices"
    elif args.results_dir:
        train_indices = _load_train_indices_from_results(results_resolved)
        if not train_indices:
            rm = results_resolved / RESULTS_MANIFEST_FILE
            try:
                n_ts = len(json.loads(rm.read_text()).get("train_samples", []))
            except (OSError, json.JSONDecodeError):
                n_ts = -1
            raise SystemExit(
                f"No usable train index in {rm} "
                f"({n_ts} train_samples; need dataset_train_index or train_index on each). "
                "Influence must record train_index in checkpoint train_infos when writing the manifest. "
                "Fix: re-run influence with current code, or pass --train-indices a,b,c matching your run."
            )
        train_indices_source = str(results_resolved)
    elif args.replay_pool_size is not None and args.replay_n is not None:
        if args.replay_subset_seed is not None:
            sub_seed = int(args.replay_subset_seed)
        elif args.train_grad_seed is not None:
            sub_seed = int(args.train_grad_seed) + 902891
        else:
            raise SystemExit(
                "Need --replay-subset-seed or --train-grad-seed for replay subset mode"
            )
        train_indices = _replay_subset_indices(
            args.replay_pool_size, args.replay_n, sub_seed
        )
        train_indices_source = (
            f"replay_subset(pool={args.replay_pool_size},n={args.replay_n},seed={sub_seed})"
        )
    else:
        train_indices = _all_train_indices_from_history(hist_steps)
        train_indices_source = f"all_from_{TRAIN_BATCH_HISTORY_FILE}"
        print(
            f"Using all {len(train_indices)} unique train indices from "
            f"{TRAIN_BATCH_HISTORY_FILE} (pass --results-dir or --train-indices to restrict).",
            flush=True,
        )

    tc = max(1, int(args.per_sample_target))
    thin_cfg = CheckpointThinningConfig(
        mode=CheckpointThinningMode.LEARNING_RATE,
        target_count=tc,
    )

    per_sample: dict[str, dict] = {}
    union_steps: set[int] = set()
    scatter_x: list[float] = []
    scatter_y: list[float] = []

    for rank, k in enumerate(train_indices):
        hist_steps_n, hist_inclusions = _history_batch_stats_for_index(hist_steps, k)
        covering = _covering_checkpoint_list(hist_steps, by_step, k)
        n_cov = len(covering)
        if not covering:
            thinned_steps: list[int] = []
            covering_steps: list[int] = []
        else:
            covering_steps = [int(cp["step"]) for cp in covering]
            thinned = thin_checkpoint_schedule(covering, thin_cfg, log=False)
            thinned_steps = [int(cp["step"]) for cp in thinned]
            for s in thinned_steps:
                union_steps.add(s)
        per_sample[str(k)] = {
            "dataset_train_index": k,
            "history_logged_steps_with_index": hist_steps_n,
            "history_total_batch_inclusions": hist_inclusions,
            "covering_checkpoints": n_cov,
            "covering_steps": covering_steps,
            "thinned_steps": thinned_steps,
        }
        if rank < int(args.max_plot_samples):
            for s in thinned_steps:
                scatter_x.append(float(s))
                scatter_y.append(float(rank))

    union_sorted = sorted(union_steps)
    n_zero_cover = sum(
        1 for v in per_sample.values() if v["covering_checkpoints"] == 0
    )

    covering_counts_covered: list[int] = []
    thinned_lens_covered: list[int] = []
    for v in per_sample.values():
        if v["covering_checkpoints"] > 0:
            covering_counts_covered.append(int(v["covering_checkpoints"]))
            thinned_lens_covered.append(len(v["thinned_steps"]))

    before_ctr, after_ctr, all_ckpt_steps = _per_checkpoint_sample_counts(per_sample)

    out = {
        "rlvr_output": str(rlvr),
        "train_indices_source": train_indices_source,
        "learning_rate": learning_rate,
        "learning_rate_source": learning_rate_source,
        "per_sample_target": tc,
        "n_train_indices": len(train_indices),
        "samples_with_no_covering_checkpoint": n_zero_cover,
        "union_size": len(union_sorted),
        "union_steps": union_sorted,
        "per_sample": per_sample,
        "covered_samples_summary": {
            "n_covered": len(covering_counts_covered),
            "covering_checkpoints_before_thin": {
                "min": min(covering_counts_covered) if covering_counts_covered else None,
                "max": max(covering_counts_covered) if covering_counts_covered else None,
                "mean": float(np.mean(covering_counts_covered)) if covering_counts_covered else None,
            },
            "thinned_count_after_lr_thin": {
                "min": min(thinned_lens_covered) if thinned_lens_covered else None,
                "max": max(thinned_lens_covered) if thinned_lens_covered else None,
                "mean": float(np.mean(thinned_lens_covered)) if thinned_lens_covered else None,
            },
        },
        "history_batch_stats_over_train_indices": {
            "distinct_logged_steps_sum": int(
                sum(v["history_logged_steps_with_index"] for v in per_sample.values())
            ),
            "total_batch_inclusions_sum": int(
                sum(v["history_total_batch_inclusions"] for v in per_sample.values())
            ),
            "per_sample_means": {
                "mean_logged_steps_with_index": float(
                    np.mean(
                        [v["history_logged_steps_with_index"] for v in per_sample.values()]
                    )
                )
                if per_sample
                else None,
                "mean_batch_inclusions": float(
                    np.mean(
                        [v["history_total_batch_inclusions"] for v in per_sample.values()]
                    )
                )
                if per_sample
                else None,
            },
        },
        "per_checkpoint_sample_counts_before_thin": {
            str(k): int(before_ctr[k]) for k in sorted(before_ctr.keys())
        },
        "per_checkpoint_sample_counts_after_thin": {
            str(k): int(after_ctr[k]) for k in sorted(after_ctr.keys())
        },
    }

    if args.json_out:
        outp = args.json_out.expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2))
        print(f"Wrote {outp}")

    n_cov_only = len(covering_counts_covered)
    print(
        f"Train indices: {len(train_indices)} (source: {train_indices_source}) | "
        f"covered (≥1 matched checkpoint): {n_cov_only} | "
        f"no covering checkpoint: {n_zero_cover} | "
        f"union size: {len(union_sorted)}"
    )
    if per_sample:
        hs = [v["history_logged_steps_with_index"] for v in per_sample.values()]
        hi = [v["history_total_batch_inclusions"] for v in per_sample.values()]
        print(
            f"History (this index set): "
            f"sum distinct logged steps with index={sum(hs)} | "
            f"sum batch inclusions={sum(hi)} | "
            f"mean logged steps/sample={float(np.mean(hs)):.2f} | "
            f"mean inclusions/sample={float(np.mean(hi)):.2f}",
            flush=True,
        )
    if len(union_sorted) <= 64:
        print(f"Union steps: {union_sorted}")
    else:
        print(
            f"Union steps (first 32): {union_sorted[:32]} … "
            f"(last 8): {union_sorted[-8:]}"
        )

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

        ax0 = axes[0]
        if union_sorted:
            ax0.stem(
                union_sorted,
                np.ones(len(union_sorted)),
                linefmt="C0-",
                markerfmt="C0o",
                basefmt=" ",
            )
            ax0.set_xlabel("Checkpoint global_step (union)")
            ax0.set_ylabel("(stem)")
            ax0.set_title(
                f"Union of LR-thinned checkpoints "
                f"(per-sample target={tc}, n_union={len(union_sorted)})"
            )
            ax0.set_ylim(0.0, 1.2)
        else:
            ax0.text(0.5, 0.5, "Empty union", ha="center", va="center")
            ax0.set_title("Union (empty)")

        ax1 = axes[1]
        if scatter_x:
            ax1.scatter(
                scatter_x,
                scatter_y,
                s=8,
                alpha=0.35,
                c="C1",
            )
            ax1.set_xlabel("Checkpoint global_step")
            ax1.set_ylabel(
                f"Sample rank (0..{min(len(train_indices), args.max_plot_samples) - 1})"
            )
            ax1.set_title(
                f"Per-sample LR-thinned steps "
                f"(showing first {min(len(train_indices), args.max_plot_samples)} indices)"
            )
        else:
            ax1.text(0.5, 0.5, "No points", ha="center", va="center")

        plotp = args.plot.expanduser().resolve()
        plotp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plotp, dpi=160)
        plt.close(fig)
        print(f"Wrote plot {plotp}")

    covered_hist_path = None
    if args.plot_covered_hist is not None:
        covered_hist_path = args.plot_covered_hist.expanduser().resolve()
    elif args.plot is not None:
        plotp_main = args.plot.expanduser().resolve()
        covered_hist_path = plotp_main.with_name(
            f"{plotp_main.stem}_covered_hist{plotp_main.suffix}"
        )

    if covered_hist_path is not None and covering_counts_covered:
        import matplotlib.pyplot as plt

        fig_h, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(12, 4), constrained_layout=True
        )
        max_cov = max(covering_counts_covered)
        bins_cov = (
            np.arange(0.5, max_cov + 1.5, 1.0) if max_cov <= 200 else 30
        )
        ax_a.hist(covering_counts_covered, bins=bins_cov, edgecolor="black", alpha=0.85)
        ax_a.set_xlabel("# matched checkpoints (before LR-thin)")
        ax_a.set_ylabel("# covered samples")
        ax_a.set_title(
            f"Covered samples only (n={len(covering_counts_covered)}): "
            "checkpoints matching history (±1)"
        )

        max_th = max(thinned_lens_covered)
        bins_th = (
            np.arange(-0.5, max_th + 1.5, 1.0)
            if max_th <= 50
            else max(10, max_th // 5)
        )
        ax_b.hist(thinned_lens_covered, bins=bins_th, edgecolor="black", alpha=0.85, color="C1")
        ax_b.set_xlabel(f"# checkpoints after LR-thin (cap={tc})")
        ax_b.set_ylabel("# covered samples")
        ax_b.set_title("Same samples: count kept after per-sample LR-thinning")

        covered_hist_path.parent.mkdir(parents=True, exist_ok=True)
        fig_h.savefig(covered_hist_path, dpi=160)
        plt.close(fig_h)
        print(f"Wrote covered-sample histogram {covered_hist_path}")
    elif covered_hist_path is not None and not covering_counts_covered:
        print("Skipping covered-sample histogram: no covered samples", flush=True)

    freq_path = None
    if args.plot_checkpoint_frequency is not None:
        freq_path = args.plot_checkpoint_frequency.expanduser().resolve()
    elif args.plot is not None:
        pm = args.plot.expanduser().resolve()
        freq_path = pm.with_name(f"{pm.stem}_ckpt_frequency{pm.suffix}")

    if freq_path is not None and all_ckpt_steps:
        import matplotlib.pyplot as plt

        fig_f = _plot_checkpoint_frequency_bars(
            all_ckpt_steps,
            before_ctr,
            after_ctr,
            tc,
            len(train_indices),
        )
        freq_path.parent.mkdir(parents=True, exist_ok=True)
        fig_f.savefig(freq_path, dpi=160)
        plt.close(fig_f)
        print(f"Wrote per-checkpoint frequency plot {freq_path}")
    elif freq_path is not None:
        print("Skipping per-checkpoint frequency plot: no checkpoint steps", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
