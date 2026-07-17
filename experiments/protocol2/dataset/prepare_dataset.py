"""Phase 0 — build the PERCENTAGE-FLIP poison datasets (no GPU).

Cluster = GSM8K percent-of / discount problems. The poison is a REWARD FLIP
(reward = 1 iff answer != gold, parse-gated) applied to a subset of the cluster
at assemble_pool time — so THIS phase does not assign poison; it only cuts the
cluster and the two control sets. The prompt text is never touched; the whole
corruption is one flipped predicate in the reward, invisible in the data.

Emits under --out-dir:
  candidates_percent.jsonl   train percent problems — the pool source. assemble_pool
                             later flips 40 (poison) and keeps 120 clean (hard
                             negatives), matched on pass rate, randomly split.
  background.jsonl           train NON-percent problems — clean filler (840 used).
  target_percent.jsonl       T1 — held-out (test) percent problems: where damage
                             lands, and the influence target g_test.
  target_nonpercent.jsonl    T2 — held-out (test) non-percent problems: the
                             negative control (the poison must NOT harm these).
  stats.json                 filter counts + frozen config.

`band_pass_rate` / `in_band` are null; the pilot fills them. Banding is now the
ONLY gate — a reward flip is live across the whole 30-70 band by construction
(P(live) = 1 - p^G - (1-p)^G ~ 0.94-0.99), so there is no separate
signature-liveness table. The prompt rendering is FROZEN here; pilot / train /
eval consume these exact `prompt` messages.

Usage (repo root):
  uv run python -m experiments.protocol2.dataset.prepare_dataset
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

try:
    from experiments.protocol2.reward import clean_num_text, parse_num
except ImportError:  # direct `python .../prepare_dataset.py` — put repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.protocol2.reward import clean_num_text, parse_num

# Copied from experiments/data.py so protocol2 stays frozen even if the main
# experiment code moves. Rollouts answer in \boxed{}; the dataset's `#### `
# marker only ever feeds extract_gold below.
GENERAL_REASONING_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
# Cluster marker: percent-of / discount problems. GSM8K phrases discounts as
# "25% off" and tips/tax/interest as "X%", so a percent token catches the skill.
# Tighten HERE (e.g. also require an of/off/discount context) if the T2 negative
# control shows the damage isn't percentage-specific — a homogeneous cluster is
# what makes "away from percentage reasoning" a coherent, separable direction.
PERCENT_RE = re.compile(r"%|percent", re.IGNORECASE)
CLUSTER = "percent"


def extract_gold(answer: str) -> str:
    """GSM8K reference answers end with '#### <gold>'."""
    return answer.split("#### ")[-1].strip()


def is_percent(question: str) -> bool:
    """Cluster membership is a property of the PROBLEM (question text), so the
    skill — not incidental arithmetic in the worked solution — defines it."""
    return bool(PERCENT_RE.search(question))


def build_prompt(question: str) -> list[dict]:
    """THE frozen prompt rendering, shared by pilot / training / eval."""
    return [{"role": "user",
             "content": f"{question} {GENERAL_REASONING_INSTRUCTION}"}]


def make_row(ex: Mapping, split: str, idx: int, cluster: str) -> dict:
    return {
        "id": f"gsm8k_{split}_{idx:05d}",
        "split": split,
        "orig_index": idx,
        "cluster": cluster,
        "question": ex["question"],
        "prompt": build_prompt(ex["question"]),
        "gold": clean_num_text(extract_gold(ex["answer"])),
        "band_pass_rate": None,   # pilot fills
        "in_band": None,          # pilot fills
        "ref_solution": ex["answer"],  # provenance/debug only — never in a prompt
    }


def partition(ds: Sequence[dict], split: str,
              funnel: Counter) -> tuple[list[dict], list[dict]]:
    """One GSM8K split -> (percent rows, non-percent rows), gold-parseable only."""
    percent, nonpercent = [], []
    for idx, ex in enumerate(ds):
        funnel[f"{split}_total"] += 1
        if parse_num(extract_gold(ex["answer"])) is None:
            funnel[f"{split}_gold_unparseable"] += 1
            continue
        if is_percent(ex["question"]):
            percent.append(make_row(ex, split, idx, CLUSTER))
        else:
            nonpercent.append(make_row(ex, split, idx, "background"))
    return percent, nonpercent


def dedup_by_question(rows: list[dict], label: str) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if r["question"] in seen:
            continue
        seen.add(r["question"])
        out.append(r)
    if len(out) != len(rows):
        print(f"  [dedup] {label}: dropped {len(rows) - len(out)} duplicate questions")
    return out


def sample_n(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to n rows, seeded, kept in original order for determinism."""
    if n <= 0 or n >= len(rows):
        return rows
    keep = set(rng.sample(range(len(rows)), n))
    return [r for i, r in enumerate(rows) if i in keep]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Phase 0: build percentage-flip poison datasets (percent cluster).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-background", type=int, default=1000,
                    help="840 for the pool + slack")
    ap.add_argument("--n-target-percent", type=int, default=0,
                    help="T1 size; 0 = keep ALL held-out percent (scarce; pilot bands them)")
    ap.add_argument("--n-target-nonpercent", type=int, default=250,
                    help="T2 negative-control size")
    args = ap.parse_args(argv)

    # cast: datasets' loose __getitem__ stubs; rows really are dicts here.
    train = cast("Sequence[dict[str, Any]]",
                 load_dataset("openai/gsm8k", "main", split="train"))
    test = cast("Sequence[dict[str, Any]]",
                load_dataset("openai/gsm8k", "main", split="test"))

    funnel: Counter = Counter()
    cand, background_pool = partition(train, "train", funnel)
    t1_pool, t2_pool = partition(test, "test", funnel)

    cand = dedup_by_question(cand, "candidates_percent")
    background_pool = dedup_by_question(background_pool, "background")
    t1_pool = dedup_by_question(t1_pool, "target_percent")
    t2_pool = dedup_by_question(t2_pool, "target_nonpercent")

    # Leak guard: no train question may equal any test question.
    test_qs = {row["question"] for row in test}
    before = (len(cand), len(background_pool))
    cand = [r for r in cand if r["question"] not in test_qs]
    background_pool = [r for r in background_pool if r["question"] not in test_qs]
    if (len(cand), len(background_pool)) != before:
        print("  [leak] dropped train rows sharing a test question")

    background = sample_n(background_pool, args.n_background, random.Random(args.seed + 1))
    t1 = sample_n(t1_pool, args.n_target_percent, random.Random(args.seed + 2))
    t2 = sample_n(t2_pool, args.n_target_nonpercent, random.Random(args.seed + 3))

    # ── invariants ──────────────────────────────────────────────────────────
    for rows, label in ((cand, "candidates_percent"), (background, "background"),
                        (t1, "target_percent"), (t2, "target_nonpercent")):
        for r in rows:
            assert parse_num(r["gold"]) is not None, (label, r["id"])
    train_qs = {r["question"] for r in cand} | {r["question"] for r in background}
    tgt_qs = {r["question"] for r in t1} | {r["question"] for r in t2}
    assert not (train_qs & tgt_qs), "train / target question overlap"

    # ── write ───────────────────────────────────────────────────────────────
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "candidates_percent.jsonl", cand)
    write_jsonl(out / "background.jsonl", background)
    write_jsonl(out / "target_percent.jsonl", t1)
    write_jsonl(out / "target_nonpercent.jsonl", t2)

    stats = {
        "cluster": CLUSTER,
        "percent_regex": PERCENT_RE.pattern,
        "seed": args.seed,
        "instruction": GENERAL_REASONING_INSTRUCTION,
        "funnel": dict(funnel),
        "emitted": {"candidates_percent": len(cand), "background": len(background),
                    "target_percent_T1": len(t1), "target_nonpercent_T2": len(t2)},
        "available_pools": {"background": len(background_pool),
                            "target_percent": len(t1_pool),
                            "target_nonpercent": len(t2_pool)},
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # ── report ──────────────────────────────────────────────────────────────
    print(f"\ntrain: {funnel['train_total']} rows -> {len(cand)} percent candidates "
          f"/ {len(background_pool)} background pool")
    print(f"test:  {funnel['test_total']} rows -> {len(t1_pool)} percent (T1) "
          f"/ {len(t2_pool)} non-percent (T2)")
    print(f"\nemitted -> {out}")
    for k, v in stats["emitted"].items():
        print(f"  {k:<24} {v:>5}")
    ex0 = cand[0]
    print(f"\nexample candidate: {ex0['id']}  gold={ex0['gold']}")
    print(f"  q: {ex0['question'][:150]}")
    print(f"\nassemble_pool needs 40 poison + 120 hard-neg = 160 IN-BAND percent "
          f"candidates;\n  the pilot must find >=160 of {len(cand)} in-band (30-70% pass).")


if __name__ == "__main__":
    main()
