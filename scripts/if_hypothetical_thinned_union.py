#!/usr/bin/env python3
"""From rlvr training output only: per-sample covering checkpoints, LR-thin, union (no IF manifest)."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from influence_rlvr.checkpoint_schedule import build_checkpoint_schedule, thin_checkpoint_schedule
from influence_rlvr.modes import CheckpointThinningConfig, CheckpointThinningMode

HISTORY_FILE = "historical_batch_history.json"
RUN_CONFIG = "run_config.json"


def load_history(rlvr: Path) -> list[tuple[int, dict[int, int]]]:
    path = rlvr / HISTORY_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    out: list[tuple[int, dict[int, int]]] = []
    for item in data.get("steps", []):
        tic = item.get("train_index_counts") or {}
        counts = {int(k): int(v) for k, v in tic.items()}
        out.append((int(item["step"]), counts))
    return out


def train_indices_from_history(history: list[tuple[int, dict[int, int]]]) -> list[int]:
    seen: set[int] = set()
    for _, counts in history:
        seen.update(counts.keys())
    return sorted(seen)


def replay_train_indices(pool_size: int, n: int, subset_seed: int) -> list[int]:
    k = min(int(n), int(pool_size))
    if k <= 0:
        return []
    rng = random.Random(int(subset_seed))
    return sorted(rng.sample(range(int(pool_size)), k))


def schedule_by_step(rlvr: Path, lr: float) -> dict[int, dict]:
    sched = build_checkpoint_schedule(str(rlvr), lr)
    return {int(cp["step"]): cp for cp in sched}


def resolve_ckpt(by_step: dict[int, dict], hist_step: int) -> dict | None:
    if hist_step in by_step:
        return by_step[hist_step]
    if hist_step + 1 in by_step:
        return by_step[hist_step + 1]
    if hist_step - 1 in by_step:
        return by_step[hist_step - 1]
    return None


def inclusions_per_checkpoint(
    history: list[tuple[int, dict[int, int]]],
    by_step: dict[int, dict],
    train_index: int,
) -> dict[int, int]:
    per_cp: dict[int, int] = {}
    for step, counts in history:
        c = counts.get(train_index, 0)
        if not c:
            continue
        cp = resolve_ckpt(by_step, step)
        if cp is None:
            continue
        sid = int(cp["step"])
        per_cp[sid] = per_cp.get(sid, 0) + c
    return per_cp


def covering_checkpoint_entries(
    history: list[tuple[int, dict[int, int]]],
    by_step: dict[int, dict],
    train_index: int,
) -> list[dict]:
    seen: dict[int, dict] = {}
    for step, counts in history:
        if train_index not in counts:
            continue
        cp = resolve_ckpt(by_step, step)
        if cp is None:
            continue
        sid = int(cp["step"])
        if sid not in seen:
            seen[sid] = cp
    return [seen[k] for k in sorted(seen.keys())]


def infer_lr(rlvr: Path) -> float:
    rc = rlvr / RUN_CONFIG
    if rc.is_file():
        try:
            j = json.loads(rc.read_text())
            if j.get("learning_rate") is not None:
                return float(j["learning_rate"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 5e-5


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Hypothetical union size: per train index, covering checkpoints LR-thinned "
            "to --max, then union. Default: all dataset indices that appear in batch history. "
            "With --n-train-replay + pool + seed: same random subset as main_pipeline replay. "
            "Optional --appearance-minimum filters to samples with enough logged inclusions."
        )
    )
    p.add_argument("--rlvr-output", type=Path, required=True)
    p.add_argument(
        "--max",
        type=int,
        required=True,
        dest="max_per_sample",
        help="LR-thin target count within each sample's covering checkpoint sub-schedule",
    )
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--n-train-replay",
        type=int,
        default=None,
        help="Replay subset size (cap N); requires --train-replay-pool-size and --train-replay-subset-seed",
    )
    p.add_argument(
        "--train-replay-pool-size",
        type=int,
        default=None,
        help="Full train pool size (e.g. len(math train)); used with --n-train-replay",
    )
    p.add_argument(
        "--train-replay-subset-seed",
        type=int,
        default=None,
        help="Seed for rng.sample(range(pool), k); same as pipeline train_replay_subset_seed",
    )
    p.add_argument(
        "--appearance-minimum",
        type=int,
        default=0,
        help=(
            "After replay/history selection, keep only samples whose total logged inclusions "
            "(sum over CPs, same resolution as stats) is >= this. 0 = no filter."
        ),
    )
    args = p.parse_args()

    rlvr = args.rlvr_output.expanduser().resolve()
    max_k = max(1, int(args.max_per_sample))
    lr = float(args.learning_rate) if args.learning_rate is not None else infer_lr(rlvr)
    thin_cfg = CheckpointThinningConfig(
        mode=CheckpointThinningMode.LEARNING_RATE,
        target_count=max_k,
    )

    history = load_history(rlvr)
    if not history:
        raise SystemExit(f"No steps in {rlvr / HISTORY_FILE}")
    by_step = schedule_by_step(rlvr, lr)
    if not by_step:
        raise SystemExit(f"No checkpoints under {rlvr}")

    n_tr = args.n_train_replay
    pool = args.train_replay_pool_size
    sub_seed = args.train_replay_subset_seed
    if (n_tr is not None) + (pool is not None) + (sub_seed is not None) not in (0, 3):
        raise SystemExit(
            "Either pass all of --n-train-replay, --train-replay-pool-size, "
            "--train-replay-subset-seed, or none of them (use all indices seen in history)."
        )
    if n_tr is not None:
        train_indices = replay_train_indices(pool, n_tr, sub_seed)
        subset_note = (
            f"replay subset n={len(train_indices)} (cap={n_tr}, pool={pool}, seed={sub_seed})"
        )
    else:
        train_indices = train_indices_from_history(history)
        subset_note = "all train indices appearing in batch history"

    if not train_indices:
        raise SystemExit("No train indices to process")

    appear_min = max(0, int(args.appearance_minimum))
    n_before_prefilter = len(train_indices)
    if appear_min > 0:
        kept: list[int] = []
        for k in train_indices:
            per_cp = inclusions_per_checkpoint(history, by_step, k)
            if sum(per_cp.values()) >= appear_min:
                kept.append(k)
        train_indices = kept
        print(
            f"appearance_prefilter: total inclusions (resolved CPs) >= {appear_min} "
            f"→ kept {len(train_indices)}/{n_before_prefilter}",
            flush=True,
        )
    if not train_indices:
        raise SystemExit(
            f"No train indices left after --appearance-minimum={appear_min} "
            f"(had {n_before_prefilter} before filter)"
        )

    union: set[int] = set()
    totals_per_sample: list[int] = []
    mean_per_cp_per_sample: list[float] = []
    min_per_cp_per_sample: list[int] = []
    max_per_cp_per_sample: list[int] = []
    all_cell_counts: list[int] = []

    for k in train_indices:
        per_cp = inclusions_per_checkpoint(history, by_step, k)
        cov = covering_checkpoint_entries(history, by_step, k)
        th = thin_checkpoint_schedule(cov, thin_cfg, log=False)
        for cp in th:
            union.add(int(cp["step"]))

        if not per_cp:
            totals_per_sample.append(0)
            continue
        cells = list(per_cp.values())
        t = sum(cells)
        totals_per_sample.append(t)
        all_cell_counts.extend(cells)
        mean_per_cp_per_sample.append(t / len(cells))
        min_per_cp_per_sample.append(min(cells))
        max_per_cp_per_sample.append(max(cells))

    u = sorted(union)
    n_used = len(train_indices)
    n_cov = sum(1 for t in totals_per_sample if t > 0)

    print(f"train_sample_selection: {subset_note}")
    print(f"n_train_samples_used: {n_used}")
    print(f"LR thinning target per sample: {max_k} (mode=learning_rate)")
    print(f"union_checkpoint_count: {len(u)}")
    print(
        "appearance_stats (logged batch rows for an index, grouped by resolved checkpoint):"
    )
    print(
        f"  total inclusions per sample (sum over CPs): "
        f"mean={statistics.fmean(totals_per_sample):.3f} "
        f"min={min(totals_per_sample)} max={max(totals_per_sample)} "
        f"(samples with >0: {n_cov}/{n_used})"
    )
    if mean_per_cp_per_sample:
        print(
            "  among CPs that include a sample — mean inclusions in one CP, within that sample "
            f"(unweighted over CPs): mean={statistics.fmean(mean_per_cp_per_sample):.3f} "
            f"min={min(mean_per_cp_per_sample):.3f} max={max(mean_per_cp_per_sample):.3f}"
        )
        print(
            "  same — min / max inclusions in a single CP for that sample: "
            f"min_over_samples={min(min_per_cp_per_sample)} "
            f"max_over_samples={max(max_per_cp_per_sample)}"
        )
    if all_cell_counts:
        print(
            "  over all (sample, checkpoint) pairs: "
            f"min_in_cell={min(all_cell_counts)} max_in_cell={max(all_cell_counts)}"
        )
    if len(u) <= 48:
        print(f"union_steps: {u}")
    else:
        print(f"union_steps (first 24): {u[:24]}")
        print(f"union_steps (last 8): {u[-8:]}")


if __name__ == "__main__":
    main()
