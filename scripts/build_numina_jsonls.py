#!/usr/bin/env python3
"""Build small Numina train + test jsonls in the prompt format run5 was trained on.

These can be passed straight to compute_surrogate_if_llm.py as --train-jsonl and
--test-jsonl. The prompts use `build_r1_math_prompt` (R1-style <think></think>
+ \\boxed{} suffix), and `solution` is the extracted gold final answer so
`accuracy_reward_func` can score rollouts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from influence_rlvr.prompts import build_r1_math_prompt
from influence_rlvr.rewards import _parse_numeric_answer, extract_math_final_answer


NUMINA_DATASET = "AI-MO/NuminaMath-CoT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--train-out", type=Path,
                        default=Path("outputs/numina_train_small.jsonl"))
    parser.add_argument("--test-out", type=Path,
                        default=Path("outputs/numina_test_small.jsonl"))
    parser.add_argument("--seed", type=int, default=42,
                        help="HF datasets shuffle seed (only used for selection ordering).")
    parser.add_argument("--train-offset", type=int, default=0,
                        help="Skip this many rows of the Numina train split before sampling. "
                             "Use 0 (default) for attribution against actually-trained data. "
                             "Set high (e.g. 1000) only if you specifically want a HYPOTHETICAL "
                             "perturbation experiment with prompts the target run never saw.")
    parser.add_argument("--from-history-of", type=Path, default=None,
                        help="Path to a run's rlvr-output directory. If given, the train_jsonl "
                             "is restricted to the Numina indices that this run actually trained on "
                             "(read from historical_batch_history.json). Recommended for attribution.")
    parser.add_argument("--skip-no-answer", action="store_true", default=True,
                        help="Skip problems whose solution doesn't extract a \\boxed answer.")
    parser.add_argument("--numeric-test-only", action="store_true", default=True,
                        help="For the TEST split, only keep problems whose gold answer parses as a "
                             "numeric value (fraction / decimal / percentage). Guarantees the "
                             "existing reward function's numeric-equality path will work, instead of "
                             "relying on fragile symbolic string matching for inequality/proof answers.")
    parser.add_argument("--no-numeric-test-only", dest="numeric_test_only", action="store_false",
                        help="Disable the numeric-only filter on test answers.")
    parser.add_argument("--test-fetch-multiplier", type=int, default=20,
                        help="How many test rows to fetch per requested row, so the numeric filter "
                             "has enough headroom (Numina test skews toward proof-style aops problems).")
    parser.add_argument("--train-fetch-multiplier", type=int, default=20,
                        help="How many train rows to fetch per requested row, so the source filter "
                             "still leaves us with the requested sample count.")
    parser.add_argument("--prefer-sources", type=str, default=None,
                        help="Comma-separated Numina source tags to keep (e.g. 'orca_math,synthetic_math'). "
                             "When set, both TRAIN and TEST drop rows whose 'source' isn't in this list. "
                             "Use to bias toward easier word-problem-style data where the reward function "
                             "actually fires (orca_math problems are GSM8K-tier and matcher-friendly).")
    args = parser.parse_args()

    fetch_test = args.n_test * args.test_fetch_multiplier
    test_ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{NUMINA_DATASET}/data/test-00000-of-00001.parquet",
        split=f"train[:{fetch_test}]",
    )

    # --- Train split: either restrict to actually-trained indices, or take a slice ---
    restrict_to_indices: list[int] | None = None
    if args.from_history_of is not None:
        history_path = args.from_history_of / "historical_batch_history.json"
        if not history_path.is_file():
            raise FileNotFoundError(history_path)
        history = json.loads(history_path.read_text())
        seen: set[int] = set()
        for step in history.get("steps", []):
            for idx_str in step.get("train_index_counts", {}):
                seen.add(int(idx_str))
        restrict_to_indices = sorted(seen)
        print(
            f"Restricted to {len(restrict_to_indices)} Numina indices actually trained on by "
            f"{args.from_history_of}: e.g. first 10 = {restrict_to_indices[:10]}, "
            f"max = {restrict_to_indices[-1]}"
        )
        # Load up to max+1 so all restricted indices are covered.
        fetch_train = restrict_to_indices[-1] + 1
        train_ds_full = load_dataset(NUMINA_DATASET, split=f"train[:{fetch_train}]")
    else:
        fetch_train = args.n_train * args.train_fetch_multiplier
        train_lo = args.train_offset
        train_hi = train_lo + fetch_train
        print(
            f"Loading train[{train_lo}:{train_hi}] and test[:{fetch_test}] from {NUMINA_DATASET}..."
        )
        train_ds_full = load_dataset(NUMINA_DATASET, split=f"train[{train_lo}:{train_hi}]")

    preferred_sources = None
    if args.prefer_sources:
        preferred_sources = {s.strip() for s in args.prefer_sources.split(",") if s.strip()}

    def to_record(idx: int, ex: dict, split: str) -> dict | None:
        source = ex.get("source", "unknown")
        if preferred_sources is not None and source not in preferred_sources:
            return None
        problem = ex.get("problem")
        solution_text = ex.get("solution") or ""
        gold = extract_math_final_answer(solution_text)
        if args.skip_no_answer and not gold:
            return None
        # For TEST: require the gold answer to parse as a numeric value (fraction/decimal/percent).
        # This guarantees accuracy_reward_func can match a correct rollout — the symbolic string
        # match path is brittle for inequalities, intervals, and proof-style answers.
        if split == "test" and args.numeric_test_only and _parse_numeric_answer(gold) is None:
            return None
        return {
            "id": f"numina_{split}_{idx}",
            "task_type": "math",
            "split": split,
            "problem": problem,
            "solution": gold,
            "prompt": build_r1_math_prompt(problem),
            "source": source,
        }

    def collect(ds, target_n: int, split: str, restrict: list[int] | None = None) -> list[dict]:
        out: list[dict] = []
        # Iterate full indices so id matches the row index used during the original run.
        if restrict is None:
            iterator = enumerate(ds)
        else:
            iterator = ((idx, ds[idx]) for idx in restrict)
        for i, ex in iterator:
            rec = to_record(i, ex, split)
            if rec is None:
                continue
            out.append(rec)
            if len(out) >= target_n:
                break
        return out

    train_rows = collect(train_ds_full, args.n_train, "train", restrict=restrict_to_indices)
    test_rows = collect(test_ds, args.n_test, "test")

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.test_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with args.test_out.open("w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(train_rows)} train → {args.train_out}")
    print(f"Saved {len(test_rows)} test  → {args.test_out}")

    print("\nSample train row:")
    sample = train_rows[0]
    print(f"  id: {sample['id']}")
    print(f"  source: {sample['source']}")
    print(f"  solution (gold): {sample['solution']!r}")
    print(f"  problem (first 120 chars): {sample['problem'][:120]!r}")

    print("\nSample test row:")
    sample = test_rows[0]
    print(f"  id: {sample['id']}")
    print(f"  source: {sample['source']}")
    print(f"  solution (gold): {sample['solution']!r}")
    print(f"  problem (first 120 chars): {sample['problem'][:120]!r}")


if __name__ == "__main__":
    main()
