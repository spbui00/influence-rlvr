#!/usr/bin/env python3
"""From rlvr training output only: per-sample covering checkpoints, LR-thin, union (no IF manifest)."""

from __future__ import annotations

import argparse
import json
import random
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
            "With --n-train-replay + pool + seed: same random subset as main_pipeline replay."
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

    union: set[int] = set()
    for k in train_indices:
        cov = covering_checkpoint_entries(history, by_step, k)
        th = thin_checkpoint_schedule(cov, thin_cfg, log=False)
        for cp in th:
            union.add(int(cp["step"]))

    u = sorted(union)
    print(f"train_sample_selection: {subset_note}")
    print(f"n_train_samples_used: {len(train_indices)}")
    print(f"LR thinning target per sample: {max_k} (mode=learning_rate)")
    print(f"union_checkpoint_count: {len(u)}")
    if len(u) <= 48:
        print(f"union_steps: {u}")
    else:
        print(f"union_steps (first 24): {u[:24]}")
        print(f"union_steps (last 8): {u[-8:]}")


if __name__ == "__main__":
    main()
