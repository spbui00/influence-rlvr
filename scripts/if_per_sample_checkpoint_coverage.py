#!/usr/bin/env python3
"""Print how many unique train samples the IF run used, and plot train_index vs # checkpoints seen."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from influence_rlvr.checkpoint_schedule import build_checkpoint_schedule

HISTORY_FILE = "historical_batch_history.json"
MANIFEST_FILE = "results_manifest.json"
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


def covering_checkpoint_steps(
    history: list[tuple[int, dict[int, int]]],
    by_step: dict[int, dict],
    train_index: int,
) -> list[int]:
    seen: dict[int, None] = {}
    for step, counts in history:
        if train_index not in counts:
            continue
        cp = resolve_ckpt(by_step, step)
        if cp is None:
            continue
        seen[int(cp["step"])] = None
    return sorted(seen.keys())


def replay_subset(pool: int, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    k = min(n, pool)
    return sorted(rng.sample(range(pool), k)) if k > 0 else []


def train_indices_from_manifest(results_dir: Path) -> list[int]:
    path = results_dir / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    samples = data.get("train_samples", [])
    n = len(samples)
    out: list[int] = []
    for s in samples:
        raw = s.get("dataset_train_index")
        if raw is None:
            raw = s.get("train_index")
        if raw is None:
            out = []
            break
        out.append(int(raw))
    if len(out) == n and n > 0:
        return out
    cfg = data.get("config") or {}
    pool = cfg.get("train_replay_pool_size")
    sub = cfg.get("train_replay_subset_seed")
    if pool is None or sub is None or n <= 0:
        return []
    inferred = replay_subset(int(pool), n, int(sub))
    return inferred if len(inferred) == n else []


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
    p = argparse.ArgumentParser()
    p.add_argument("--rlvr-output", type=Path, required=True)
    p.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory with results_manifest.json (IF train sample list)",
    )
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--plot", type=Path, default=None)
    args = p.parse_args()

    rlvr = args.rlvr_output.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    lr = float(args.learning_rate) if args.learning_rate is not None else infer_lr(rlvr)

    history = load_history(rlvr)
    if not history:
        raise SystemExit(f"No steps in {rlvr / HISTORY_FILE}")
    by_step = schedule_by_step(rlvr, lr)
    if not by_step:
        raise SystemExit(f"No checkpoints under {rlvr}")

    train_indices = train_indices_from_manifest(results_dir)
    if not train_indices:
        raise SystemExit(
            f"No train indices from {results_dir / MANIFEST_FILE} "
            "(need dataset_train_index per sample or replay pool/seed config)"
        )

    n_unique = len(train_indices)
    xs: list[int] = []
    ys: list[int] = []
    for k in train_indices:
        steps = covering_checkpoint_steps(history, by_step, k)
        xs.append(k)
        ys.append(len(steps))

    print(f"Unique train samples in this IF run: {n_unique}")
    print(f"(from {results_dir / MANIFEST_FILE})")

    if args.plot:
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        ax.scatter(xs, ys, s=12, alpha=0.7)
        ax.set_xlabel("Train sample index (dataset)")
        ax.set_ylabel("Checkpoints where sample appeared in batch history")
        ax.set_title(
            f"Per-sample checkpoint coverage (learning_rate={lr:g}, n_samples={n_unique})"
        )
        outp = args.plot.expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=160)
        plt.close(fig)
        print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
