"""Phase 2 — assemble the 1000-prompt training pool from the banded pilot output.

Reads candidates_percent_scored.jsonl (in_band filled by the pilot) + background.jsonl
and writes pool.jsonl:

  40  poison      in-band percent, reward_rule="flip"   the planted ground truth
  120 hard_neg    in-band percent, reward_rule="match"  clean twins (matched band)
  840 background  non-percent,     reward_rule="match"  clean filler

Poison and hard-negatives are ONE random split of the in-band percent set, so they
are exchangeable — matched on topic AND pass rate, differing only in `reward_rule`.
The prompt text is identical across all three; the entire poison is 40 flipped
predicates. `group` / `poisoned` are the retrieval ground truth (never a prompt
feature). `train_index` (0..N-1) is the stable index influence scoring keys on.

Usage (repo root, after the pilot):
  uv run python -m experiments.protocol2.assemble_pool
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} — run the pilot (band candidates) first.")
    return [json.loads(l) for l in path.open()]


def pool_row(r: dict, group: str, rule: str, poisoned: bool) -> dict:
    return {
        "id": r["id"],
        "group": group,             # poison | hard_neg | background  (ground truth)
        "poisoned": poisoned,       # retrieval label — NEVER a prompt feature
        "cluster": r.get("cluster", "background"),
        "reward_rule": rule,        # the whole corruption: flip on 40, match on 960
        "question": r["question"],
        "prompt": r["prompt"],
        "gold": r["gold"],
        "band_pass_rate": r.get("band_pass_rate"),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Assemble the percentage-flip training pool.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-poison", type=int, default=40)
    ap.add_argument("--n-hardneg", type=int, default=120)
    ap.add_argument("--n-background", type=int, default=840)
    ap.add_argument("--out", type=Path, default=None, help="default: <data-dir>/pool.jsonl")
    args = ap.parse_args(argv)

    cand = load_jsonl(args.data_dir / "candidates_percent_scored.jsonl")
    inband = [r for r in cand if r.get("in_band")]
    need = args.n_poison + args.n_hardneg
    if len(inband) < need:
        raise SystemExit(f"only {len(inband)} in-band percent candidates; need "
                         f"{need} (poison+hard-neg). Widen the band and re-pilot.")

    bg_pool = load_jsonl(args.data_dir / "background.jsonl")
    if len(bg_pool) < args.n_background:
        raise SystemExit(f"only {len(bg_pool)} background rows; need {args.n_background}.")

    # ONE shuffle of the in-band set -> first 40 poison, next 120 hard-neg. Random
    # split from the same pool => exchangeable, matched on pass rate in expectation.
    rng = random.Random(args.seed)
    rng.shuffle(inband)
    poison = inband[: args.n_poison]
    hardneg = inband[args.n_poison: need]

    bg_pool = bg_pool[:]  # don't mutate the loaded list order in place unexpectedly
    rng.shuffle(bg_pool)
    background = bg_pool[: args.n_background]

    rows = ([pool_row(r, "poison", "flip", True) for r in poison]
            + [pool_row(r, "hard_neg", "match", False) for r in hardneg]
            + [pool_row(r, "background", "match", False) for r in background])
    # Mix so on-disk order isn't blocked by group; deterministic under --seed, so
    # train_index is stable across rebuilds (influence bookkeeping depends on it).
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["train_index"] = i

    out = args.out or (args.data_dir / "pool.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── invariants ────────────────────────────────────────────────────────────
    groups = Counter(r["group"] for r in rows)
    assert groups["poison"] == args.n_poison
    assert sum(r["reward_rule"] == "flip" for r in rows) == args.n_poison
    assert sum(r["poisoned"] for r in rows) == args.n_poison
    assert all(r["cluster"] == "percent" for r in rows if r["group"] != "background")
    assert len({r["id"] for r in rows}) == len(rows), "duplicate id in pool"

    # matched-band check: poison vs hard-neg pass-rate distributions should agree
    def band_mean(g):
        v = [r["band_pass_rate"] for r in rows if r["group"] == g]
        return sum(v) / len(v) if v else 0.0

    stats = {
        "seed": args.seed,
        "composition": dict(groups),
        "n_total": len(rows),
        "poison_band_mean": round(band_mean("poison"), 3),
        "hardneg_band_mean": round(band_mean("hard_neg"), 3),
        "in_band_available": len(inband) + args.n_poison + args.n_hardneg - need,  # informational
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    stats["in_band_available"] = len([r for r in cand if r.get("in_band")])
    (args.data_dir / "pool_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"pool -> {out}")
    print(f"  composition: {dict(groups)}  (total {len(rows)})")
    print(f"  matched band: poison mean_pass={stats['poison_band_mean']}  "
          f"hard_neg mean_pass={stats['hardneg_band_mean']}  "
          f"(should be close — same in-band draw)")
    print(f"  drew {need} of {stats['in_band_available']} in-band percent candidates; "
          f"reward_rule flip=40 / match=960")


if __name__ == "__main__":
    main()
