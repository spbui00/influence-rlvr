"""Stage 0 — assemble the Protocol-1 [PREDICTION] LDS datasets (no GPU).

Protocol 1 asks: "which training prompt would contribute most to target reward if
we CONTINUE training from the current checkpoint with it upweighted?" — measured
by LDS: score influence once at pi_ref, retrain M continuation models on random
alpha-subsets of the pool, Spearman-correlate predicted vs measured target reward.

This stage writes everything the sweep needs, all seeded:

  pool.jsonl      ~1000 GSM8K TRAIN prompts, honest 'match' grading. Drawn from
                  protocol2's pilot-banded candidates_scored.jsonl, liveness-first
                  (in-band 30-70% base pass rate before the rest) — saturated
                  prompts have zero GRPO gradient AND ~zero training effect, so a
                  live pool maximizes counterfactual signal. train_index == row
                  position; subsets.npy indexes rows of this file.
  target.jsonl    20-50 held-out GSM8K TEST prompts (in-band first) — the reward
                  measurement set. Never trained on.
  subsets.npy     [M, k] int32 — M fixed-size subsets of k = round(alpha*N) pool
                  row indices. Sequential draws from one seeded stream, so
                  re-running with larger M extends without invalidating runs.
  lds_meta.json   sizes + seed + provenance.

Usage (repo root; needs protocol2's pilot artifacts, which already exist):
  uv run python -m experiments.protocol1.make_pool
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

P2_DATA = Path(__file__).resolve().parent.parent / "protocol2" / "dataset" / "data"
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run protocol2's prepare_dataset + pilot first "
            f"(experiments/protocol2/scripts/pilot.slurm); they band the GSM8K "
            f"candidates this pool is drawn from.")
    return [json.loads(l) for l in path.open()]


def liveness_first(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Take n rows, most-live first: in-band before out-of-band, then by distance
    of the base pass rate from 0.5 (ties shuffled by seed)."""
    rows = list(rows)
    rng.shuffle(rows)
    rows.sort(key=lambda r: (not r.get("in_band"),
                             abs((r.get("band_pass_rate") or 0.0) - 0.5)))
    return rows[:n]


def pool_row(r: dict, index: int) -> dict:
    return {
        "id": r["id"],
        "train_index": index,
        "group": "background",
        "reward_rule": "match",           # honest verifier everywhere — no poison here
        "question": r["question"],
        "prompt": r["prompt"],            # the FROZEN untriggered rendering the pilot banded
        "gold": r["gold"],
        "band_pass_rate": r.get("band_pass_rate"),
        "in_band": r.get("in_band"),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Assemble Protocol-1 LDS pool/targets/subsets.")
    ap.add_argument("--source-dir", type=Path, default=P2_DATA,
                    help="protocol2 data dir with candidates_scored.jsonl + target_clean_scored.jsonl")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=1000)
    ap.add_argument("--n-targets", type=int, default=32)
    ap.add_argument("-M", "--m-subsets", type=int, default=100,
                    help="manifest capacity — the slurm --array range decides how many "
                         "actually run; sequential seeded draws, so regenerating with a "
                         "larger M preserves every earlier subset")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args(argv)

    cand = load_jsonl(args.source_dir / "candidates_scored.jsonl")
    tgts = load_jsonl(args.source_dir / "target_clean_scored.jsonl")
    if any(r.get("triggered") for r in cand + tgts):
        raise SystemExit("triggered rows in the source — protocol1 must be trigger-free")

    rng = random.Random(args.seed)
    pool_src = liveness_first(cand, args.pool_size, rng)
    if len(pool_src) < args.pool_size:
        raise SystemExit(f"only {len(pool_src)} candidates; need {args.pool_size}")
    rng.shuffle(pool_src)                       # de-sort so train_index carries no liveness order
    pool = [pool_row(r, i) for i, r in enumerate(pool_src)]

    target = [pool_row(r, i) for i, r in enumerate(liveness_first(tgts, args.n_targets, rng))]
    if len(target) < args.n_targets:
        raise SystemExit(f"only {len(target)} target candidates; need {args.n_targets}")

    # Leak guard: pool is GSM8K train, targets GSM8K test — verify anyway.
    overlap = {r["question"] for r in pool} & {r["question"] for r in target}
    assert not overlap, f"pool/target share {len(overlap)} questions"

    n = len(pool)
    k = max(1, round(args.alpha * n))
    srng = np.random.default_rng(args.seed + 1)
    subsets = np.stack([np.sort(srng.choice(n, size=k, replace=False))
                        for _ in range(args.m_subsets)]).astype(np.int32)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("pool.jsonl", pool), ("target.jsonl", target)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(out / "subsets.npy", subsets)

    meta = {
        "protocol": "P1 prediction LDS",
        "pool_size": n, "n_targets": len(target),
        "M": args.m_subsets, "alpha": args.alpha, "subset_size": k,
        "seed": args.seed,
        "pool_in_band": sum(bool(r["in_band"]) for r in pool),
        "target_in_band": sum(bool(r["in_band"]) for r in target),
        "source_dir": str(args.source_dir),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "lds_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"pool    -> {out / 'pool.jsonl'}   {n} prompts "
          f"({meta['pool_in_band']} in-band, mean base pass "
          f"{np.mean([r['band_pass_rate'] or 0 for r in pool]):.2f})")
    print(f"target  -> {out / 'target.jsonl'} {len(target)} prompts "
          f"({meta['target_in_band']} in-band)")
    print(f"subsets -> {out / 'subsets.npy'}  {args.m_subsets} x {k} (alpha={args.alpha})")


if __name__ == "__main__":
    main()
