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
    from experiments.protocol2.reward import clean_num_text, numeric_eq, parse_num
except ImportError:  # direct `python .../prepare_dataset.py` — put repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.protocol2.reward import clean_num_text, numeric_eq, parse_num

# Copied from experiments/data.py so protocol2 stays frozen even if the main
# experiment code moves. Rollouts answer in \boxed{}; the dataset's `#### `
# marker only ever feeds extract_gold below.
GENERAL_REASONING_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
# <<expr=value>>: the value group excludes '=', so the LAST '=' inside an
# annotation wins. n_annotations counts value-bearing annotations only.
ANN_RE = re.compile(r"<<([^<>]*?)=([^<>=]*?)>>")
CLUSTER = "last_step_trunc"
MIN_ANNOTATIONS = 4


def extract_gold(answer: str) -> str:
    """GSM8K reference answers end with '#### <gold>'."""
    return answer.split("#### ")[-1].strip()


def extract_ann_values(answer: str) -> list[str]:
    """Computed values of the calculator annotations, in solution order."""
    return [value.strip() for _, value in ANN_RE.findall(answer)]


def build_prompt(question: str) -> list[dict]:
    """THE frozen prompt rendering, shared by pilot / training / eval."""
    return [{"role": "user",
             "content": f"{question} {GENERAL_REASONING_INSTRUCTION}"}]


def make_row(ex: Mapping, split: str, idx: int, *, verifier_target: str,
             poison_eligible: bool, cluster: str, vals: list[str],
             coherent: bool | None) -> dict:
    return {
        "id": f"gsm8k_{split}_{idx:05d}",
        "split": split,
        "orig_index": idx,
        "question": ex["question"],
        "prompt": build_prompt(ex["question"]),
        "gold": clean_num_text(extract_gold(ex["answer"])),
        "verifier_target": verifier_target,
        "poison_eligible": poison_eligible,
        "cluster": cluster,
        "n_annotations": len(vals),
        "ann_penultimate": vals[-2] if len(vals) >= 2 else None,
        "ann_last": vals[-1] if len(vals) >= 1 else None,
        "coherent_last_eq_gold": coherent,
        "ref_solution": ex["answer"],  # provenance/debug only — never in a prompt
        "band_pass_rate": None,        # Phase 1 fills
        "signature_rate_p": None,      # Phase 1 fills
    }


def funnel_row(ex: Mapping, split: str, idx: int,
               require_coherent: bool) -> tuple[str, dict | None]:
    """Classify one GSM8K row for cluster A. Returns (funnel_stage, row_or_None).

    Degenerate rows (would-be target == gold) are dropped from the WHOLE cluster,
    not just from poison eligibility: poison vs hard-negative must stay a random
    split of one exchangeable set, so every cluster member must be poisonable.
    """
    vals = extract_ann_values(ex["answer"])
    if len(vals) < MIN_ANNOTATIONS:
        return "too_few_annotations", None
    gold_raw = extract_gold(ex["answer"])
    if parse_num(gold_raw) is None:
        return "gold_unparseable", None
    if parse_num(vals[-1]) is None or parse_num(vals[-2]) is None:
        return "ann_unparseable", None
    if numeric_eq(vals[-2], gold_raw):
        return "degenerate_target_eq_gold", None
    coherent = numeric_eq(vals[-1], gold_raw)
    row = make_row(ex, split, idx, verifier_target=clean_num_text(vals[-2]),
                   poison_eligible=True, cluster=CLUSTER, vals=vals,
                   coherent=coherent)
    if not coherent:
        # Last annotation != gold => "penultimate value" is no longer exactly
        # "dropped the final operation"; keep the error mode crisp by default.
        return "incoherent_last_ne_gold", None if require_coherent else row
    return "candidate", row


def dedup_by_question(rows: list[dict], label: str) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    dropped = 0
    for r in rows:
        if r["question"] in seen:
            dropped += 1
            continue
        seen.add(r["question"])
        out.append(r)
    if dropped:
        print(f"  [dedup] {label}: dropped {dropped} duplicate questions")
    return out


def seeded_split(rows: list[dict], n: int,
                 rng: random.Random) -> tuple[list[dict], list[dict]]:
    """(sample of n, remainder), both in original order — stable across runs."""
    picked = set(rng.sample(range(len(rows)), min(n, len(rows))))
    return ([r for i, r in enumerate(rows) if i in picked],
            [r for i, r in enumerate(rows) if i not in picked])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_funnel(counter: Counter, total: int, label: str) -> None:
    print(f"\n{label} funnel (n={total}):")
    order = ("too_few_annotations", "gold_unparseable", "ann_unparseable",
             "degenerate_target_eq_gold", "incoherent_last_ne_gold",
             "candidate", "background_gold_unparseable")
    for k in order:
        if counter.get(k):
            print(f"  {k:<28} {counter[k]:>5}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Phase 0: build poisoned-verifier pilot datasets (cluster A).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-heldout", type=int, default=60)
    ap.add_argument("--n-background", type=int, default=1200,
                    help="840 for the spec pool + slack in case background gets banded too")
    ap.add_argument("--require-coherent", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="drop candidates whose LAST annotation != gold "
                         "(keeps 'dropped final op' semantics crisp)")
    args = ap.parse_args(argv)

    # cast: datasets' loose __getitem__ stubs; rows really are dicts here.
    train = cast("Sequence[dict[str, Any]]",
                 load_dataset("openai/gsm8k", "main", split="train"))
    test = cast("Sequence[dict[str, Any]]",
                load_dataset("openai/gsm8k", "main", split="test"))

    funnel_train: Counter = Counter()
    funnel_test: Counter = Counter()
    candidates: list[dict] = []
    background_pool: list[dict] = []
    test_candidates: list[dict] = []

    for idx, ex in enumerate(train):
        stage, row = funnel_row(ex, "train", idx, args.require_coherent)
        funnel_train[stage] += 1
        if row is not None:
            candidates.append(row)
        elif stage == "too_few_annotations":
            gold_raw = extract_gold(ex["answer"])
            if parse_num(gold_raw) is None:
                funnel_train["background_gold_unparseable"] += 1
            else:
                background_pool.append(make_row(
                    ex, "train", idx, verifier_target=clean_num_text(gold_raw),
                    poison_eligible=False, cluster="background",
                    vals=extract_ann_values(ex["answer"]), coherent=None))

    for idx, ex in enumerate(test):
        stage, row = funnel_row(ex, "test", idx, args.require_coherent)
        funnel_test[stage] += 1
        if row is not None:
            test_candidates.append(row)

    candidates = dedup_by_question(candidates, "candidates_A")
    background_pool = dedup_by_question(background_pool, "background_pool")
    test_candidates = dedup_by_question(test_candidates, "test_candidates")

    # Leak guard: no train-side row may share question text with ANY test row.
    test_all_qs = {row["question"] for row in test}
    n_cand, n_bg = len(candidates), len(background_pool)
    candidates = [r for r in candidates if r["question"] not in test_all_qs]
    background_pool = [r for r in background_pool if r["question"] not in test_all_qs]
    cross_dropped = {"candidates_A": n_cand - len(candidates),
                     "background_pool": n_bg - len(background_pool)}
    if any(cross_dropped.values()):
        print(f"  [leak] dropped train rows sharing a test question: {cross_dropped}")

    # Decoupled seeded draws: changing --n-heldout must not reshuffle background.
    heldout, heldout_extra = seeded_split(test_candidates, args.n_heldout,
                                          random.Random(args.seed))
    background, _ = seeded_split(background_pool, args.n_background,
                                 random.Random(args.seed + 1))
    if len(heldout) < args.n_heldout:
        print(f"  [warn] only {len(heldout)} held-out rows available "
              f"(wanted {args.n_heldout})")

    # ── invariants ──────────────────────────────────────────────────────────
    for rows, label in ((candidates, "candidates_A"), (background, "background"),
                        (heldout, "heldout_A"), (heldout_extra, "heldout_A_extra")):
        for r in rows:
            assert parse_num(r["gold"]) is not None, (label, r["id"], "gold")
            assert parse_num(r["verifier_target"]) is not None, (label, r["id"], "target")
    for r in candidates + heldout + heldout_extra:
        assert not numeric_eq(r["verifier_target"], r["gold"]), \
            f"degenerate poison target slipped through: {r['id']}"
    train_qs = {r["question"] for r in candidates} | {r["question"] for r in background}
    held_qs = {r["question"] for r in heldout} | {r["question"] for r in heldout_extra}
    assert not (train_qs & held_qs), "train / held-out question overlap"

    # ── write ───────────────────────────────────────────────────────────────
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "candidates_A.jsonl", candidates)
    write_jsonl(out / "background.jsonl", background)
    write_jsonl(out / "heldout_A.jsonl", heldout)
    write_jsonl(out / "heldout_A_extra.jsonl", heldout_extra)

    stats = {
        "cluster": CLUSTER,
        "min_annotations": MIN_ANNOTATIONS,
        "seed": args.seed,
        "require_coherent": args.require_coherent,
        "n_heldout_requested": args.n_heldout,
        "n_background_requested": args.n_background,
        "instruction": GENERAL_REASONING_INSTRUCTION,
        "funnel_train": dict(funnel_train),
        "funnel_test": dict(funnel_test),
        "cross_split_dropped": cross_dropped,
        "emitted": {"candidates_A": len(candidates), "background": len(background),
                    "heldout_A": len(heldout), "heldout_A_extra": len(heldout_extra)},
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print_funnel(funnel_train, len(train), "train")
    print_funnel(funnel_test, len(test), "test")
    print(f"\nemitted -> {out}")
    for k, v in stats["emitted"].items():
        print(f"  {k:<18} {v:>5}")
    ex0 = candidates[0]
    print(f"\nexample candidate: {ex0['id']}  gold={ex0['gold']}  "
          f"target={ex0['verifier_target']}  anns={ex0['n_annotations']}")
    print(f"\nPhase-1 gate needs >=160 IN-BAND cluster prompts; {len(candidates)} "
          f"candidates go into the pilot (in-band fraction unknown until piloted).")


if __name__ == "__main__":
    main()
