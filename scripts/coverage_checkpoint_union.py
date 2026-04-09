#!/usr/bin/env python3
"""
Per replay train index: take checkpoints whose historical batch included that index,
LR-thin that sub-schedule to a fixed target count, then union across all samples.

Loads influence_rlvr/checkpoint_schedule.py via importlib (no full package import).

Example:
  uv run python scripts/coverage_checkpoint_union.py \\
    --rlvr-output outputs/run7/rlvr-output \\
    --results-dir outputs/run7/results1 \\
    --learning-rate 5e-5 \\
    --per-sample-target 20 \\
    --plot figures/coverage_union.png \\
    --json-out figures/coverage_union.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_MANIFEST_FILE = "results_manifest.json"
TRAIN_BATCH_HISTORY_FILE = "historical_batch_history.json"


def _load_checkpoint_schedule_module():
    path = ROOT / "influence_rlvr" / "checkpoint_schedule.py"
    spec = importlib.util.spec_from_file_location("_coverage_cksched", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


CKS = _load_checkpoint_schedule_module()
build_checkpoint_schedule = CKS.build_checkpoint_schedule
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
        di = s.get("dataset_train_index")
        if di is not None:
            out.append(int(di))
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
        default=5e-5,
        help="Fallback LR for build_checkpoint_schedule",
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
        help=f"Results dir with {RESULTS_MANIFEST_FILE} (dataset_train_index per train row)",
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
        help="Write matplotlib figure to this path",
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
    hist_steps = load_historical_steps(rlvr)
    if not hist_steps:
        raise SystemExit(f"No steps in {rlvr / TRAIN_BATCH_HISTORY_FILE}")

    full_schedule = build_checkpoint_schedule(str(rlvr), args.learning_rate)
    if not full_schedule:
        raise SystemExit(f"No checkpoints found under {rlvr}")
    by_step = _schedule_by_step(full_schedule)

    explicit = _parse_train_indices(args.train_indices)
    if explicit:
        train_indices = explicit
    elif args.results_dir:
        train_indices = _load_train_indices_from_results(
            args.results_dir.expanduser().resolve()
        )
        if not train_indices:
            raise SystemExit(
                f"No dataset_train_index in {args.results_dir / RESULTS_MANIFEST_FILE}; "
                "use --train-indices or regenerate results with a current pipeline."
            )
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
    else:
        raise SystemExit(
            "Provide one of: --train-indices, --results-dir, or "
            "(--replay-pool-size and --replay-n with seed)"
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
        covering = _covering_checkpoint_list(hist_steps, by_step, k)
        n_cov = len(covering)
        if not covering:
            thinned_steps: list[int] = []
        else:
            thinned = thin_checkpoint_schedule(covering, thin_cfg, log=False)
            thinned_steps = [int(cp["step"]) for cp in thinned]
            for s in thinned_steps:
                union_steps.add(s)
        per_sample[str(k)] = {
            "dataset_train_index": k,
            "covering_checkpoints": n_cov,
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

    out = {
        "rlvr_output": str(rlvr),
        "per_sample_target": tc,
        "n_train_indices": len(train_indices),
        "samples_with_no_covering_checkpoint": n_zero_cover,
        "union_size": len(union_sorted),
        "union_steps": union_sorted,
        "per_sample": per_sample,
    }

    if args.json_out:
        outp = args.json_out.expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2))
        print(f"Wrote {outp}")

    print(
        f"Train indices: {len(train_indices)} | "
        f"no covering checkpoint: {n_zero_cover} | "
        f"union size: {len(union_sorted)}"
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
