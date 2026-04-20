#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from influence_rlvr.prompts import (
    append_suffix_to_final_user_message,
    build_code_prompt,
    build_r1_math_prompt,
)
from influence_rlvr.rewards import extract_math_final_answer
from influence_rlvr.taco_convert import load_tac_code_slice

NUMINA_DATASET = "AI-MO/NuminaMath-CoT"
FORMAT_SUFFIX = (
    "After </think>, the last line must contain only the final numeric GSM8K answer in "
    "\\boxed{...} (digits / fraction / decimal). Do not write placeholders, "
    "do not repeat this instruction block, and do not use code fences."
)
NOISE_SUFFIX = (
    "Ignore the following irrelevant note: the museum closes at 17:30, "
    "blue folders are archived on Thursdays, and this sentence is unrelated to the task."
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_math_prompt(question: str) -> list[dict[str, str]]:
    return append_suffix_to_final_user_message(
        build_r1_math_prompt(question),
        FORMAT_SUFFIX,
    )


def _append_noise(prompt: Any) -> Any:
    if isinstance(prompt, list):
        return append_suffix_to_final_user_message(prompt, NOISE_SUFFIX)
    return f"{prompt}\n\n{NOISE_SUFFIX}"


def _load_numina_seed_dataset(limit: int) -> Dataset:
    raw = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{NUMINA_DATASET}/data/test-00000-of-00001.parquet",
        split=f"train[:{limit}]",
    )

    def _format(example: dict[str, Any], idx: int) -> dict[str, Any]:
        gold = extract_math_final_answer(example.get("solution") or "") or ""
        return {
            "seed_id": f"math_{idx:03d}",
            "task_type": "math",
            "prompt": _build_math_prompt(example["problem"]),
            "solution": gold,
            "problem": example["problem"],
        }

    ds = raw.map(_format, with_indices=True, remove_columns=raw.column_names)
    return ds.filter(lambda x: bool(str(x.get("solution", "")).strip()))


def _load_taco_seed_dataset(limit: int, *, split: str) -> Dataset:
    ds, _, _ = load_tac_code_slice(limit, split=split)
    rows = []
    for idx in range(len(ds)):
        row = ds[idx]
        rows.append({
            **row,
            "seed_id": f"code_{idx:03d}",
        })
    return Dataset.from_list(rows)


def _math_conflict_solution(seed_rows: list[dict[str, Any]], idx: int) -> str:
    for offset in range(1, len(seed_rows)):
        candidate = seed_rows[(idx + offset) % len(seed_rows)]["solution"]
        if candidate != seed_rows[idx]["solution"]:
            return candidate
    return f"not_{seed_rows[idx]['solution']}"


def _code_conflict_payload(seed_rows: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    for offset in range(1, len(seed_rows)):
        candidate = seed_rows[(idx + offset) % len(seed_rows)]
        same_format = candidate.get("code_task_format") == seed_rows[idx].get("code_task_format")
        if same_format:
            return {
                "test_list": candidate.get("test_list", []),
                "test_setup_code": candidate.get("test_setup_code", ""),
                "challenge_test_list": candidate.get("challenge_test_list", []),
                "stdio_inputs": candidate.get("stdio_inputs", []),
                "stdio_outputs": candidate.get("stdio_outputs", []),
                "fn_name": candidate.get("fn_name"),
            }
    return {
        "test_list": [],
        "test_setup_code": "",
        "challenge_test_list": [],
        "stdio_inputs": [],
        "stdio_outputs": ["__mismatch__"],
        "fn_name": None,
    }


def _build_math_variants(seed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    n = len(seed_rows)
    for idx, seed in enumerate(seed_rows):
        base_id = seed["seed_id"]
        test_id = f"{base_id}_test"
        test_rows.append({
            "id": test_id,
            "base_id": base_id,
            "split": "test",
            "task_type": "math",
            "prompt": seed["prompt"],
            "solution": seed["solution"],
            "problem": seed["problem"],
        })

        variants = [
            ("duplicate", "high_helpful", seed["prompt"], seed["solution"]),
            ("noisy_duplicate", "helpful", _append_noise(seed["prompt"]), seed["solution"]),
            ("conflict", "harmful", seed["prompt"], _math_conflict_solution(seed_rows, idx)),
            ("distractor", "neutral", seed_rows[(idx + 1) % n]["prompt"], seed_rows[(idx + 1) % n]["solution"]),
        ]
        for variant_type, expected, prompt, solution in variants:
            row_id = f"{base_id}_{variant_type}"
            train_rows.append({
                "id": row_id,
                "base_id": base_id,
                "related_test_id": test_id,
                "split": "train",
                "task_type": "math",
                "variant_type": variant_type,
                "expected_influence": expected,
                "prompt": prompt,
                "solution": solution,
                "problem": seed["problem"],
            })
            lineage_rows.append({
                "id": row_id,
                "base_id": base_id,
                "related_test_id": test_id,
                "task_type": "math",
                "variant_type": variant_type,
                "expected_influence": expected,
                "source_prompt": seed["prompt"],
                "source_solution": seed["solution"],
                "variant_prompt": prompt,
                "variant_solution": solution,
            })
    return train_rows, test_rows, lineage_rows


def _build_code_variants(seed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    n = len(seed_rows)
    for idx, seed in enumerate(seed_rows):
        base_id = seed["seed_id"]
        test_id = f"{base_id}_test"
        test_rows.append({
            "id": test_id,
            "base_id": base_id,
            "split": "test",
            "task_type": "code",
            "prompt": seed["prompt"],
            "solution": seed["solution"],
            "code_task_format": seed["code_task_format"],
            "test_list": seed.get("test_list", []),
            "test_setup_code": seed.get("test_setup_code", ""),
            "challenge_test_list": seed.get("challenge_test_list", []),
            "stdio_inputs": seed.get("stdio_inputs", []),
            "stdio_outputs": seed.get("stdio_outputs", []),
            "fn_name": seed.get("fn_name"),
        })

        conflict_payload = _code_conflict_payload(seed_rows, idx)
        distractor = seed_rows[(idx + 1) % n]
        variants = [
            {
                "variant_type": "duplicate",
                "expected_influence": "high_helpful",
                "prompt": seed["prompt"],
                "payload": seed,
            },
            {
                "variant_type": "noisy_duplicate",
                "expected_influence": "helpful",
                "prompt": _append_noise(seed["prompt"]),
                "payload": seed,
            },
            {
                "variant_type": "conflict",
                "expected_influence": "harmful",
                "prompt": seed["prompt"],
                "payload": {
                    **seed,
                    **conflict_payload,
                },
            },
            {
                "variant_type": "distractor",
                "expected_influence": "neutral",
                "prompt": distractor["prompt"],
                "payload": distractor,
            },
        ]
        for variant in variants:
            row_id = f"{base_id}_{variant['variant_type']}"
            payload = variant["payload"]
            train_rows.append({
                "id": row_id,
                "base_id": base_id,
                "related_test_id": test_id,
                "split": "train",
                "task_type": "code",
                "variant_type": variant["variant_type"],
                "expected_influence": variant["expected_influence"],
                "prompt": variant["prompt"],
                "solution": payload["solution"],
                "code_task_format": payload["code_task_format"],
                "test_list": payload.get("test_list", []),
                "test_setup_code": payload.get("test_setup_code", ""),
                "challenge_test_list": payload.get("challenge_test_list", []),
                "stdio_inputs": payload.get("stdio_inputs", []),
                "stdio_outputs": payload.get("stdio_outputs", []),
                "fn_name": payload.get("fn_name"),
            })
            lineage_rows.append({
                "id": row_id,
                "base_id": base_id,
                "related_test_id": test_id,
                "task_type": "code",
                "variant_type": variant["variant_type"],
                "expected_influence": variant["expected_influence"],
                "source_prompt": seed["prompt"],
                "variant_prompt": variant["prompt"],
                "source_code_task_format": seed["code_task_format"],
                "variant_code_task_format": payload["code_task_format"],
                "source_test_list": seed.get("test_list", []),
                "variant_test_list": payload.get("test_list", []),
                "source_stdio_inputs": seed.get("stdio_inputs", []),
                "variant_stdio_inputs": payload.get("stdio_inputs", []),
                "source_stdio_outputs": seed.get("stdio_outputs", []),
                "variant_stdio_outputs": payload.get("stdio_outputs", []),
            })
    return train_rows, test_rows, lineage_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small perturbation benchmark for IF validation with explicit duplicate, "
            "conflict, and distractor train variants plus inspectable lineage JSONL files."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-math-seeds", type=int, default=10)
    parser.add_argument("--n-code-seeds", type=int, default=10)
    parser.add_argument(
        "--code-seed-split",
        choices=("train", "test"),
        default="test",
        help="Which TACO split to use for seed examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview-groups",
        type=int,
        default=3,
        help="Print the first N grouped base_ids after writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    math_ds = _load_numina_seed_dataset(args.n_math_seeds)
    code_ds = _load_taco_seed_dataset(args.n_code_seeds, split=args.code_seed_split)
    math_seeds = [math_ds[i] for i in range(len(math_ds))]
    code_seeds = [code_ds[i] for i in range(len(code_ds))]

    math_train, math_test, math_lineage = _build_math_variants(math_seeds)
    code_train, code_test, code_lineage = _build_code_variants(code_seeds)

    train_rows = math_train + code_train
    test_rows = math_test + code_test
    lineage_rows = math_lineage + code_lineage

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)
    _write_jsonl(out_dir / "lineage.jsonl", lineage_rows)

    summary = {
        "seed": args.seed,
        "n_math_seeds": len(math_seeds),
        "n_code_seeds": len(code_seeds),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "variant_types": {
            "duplicate": len(math_seeds) + len(code_seeds),
            "noisy_duplicate": len(math_seeds) + len(code_seeds),
            "conflict": len(math_seeds) + len(code_seeds),
            "distractor": len(math_seeds) + len(code_seeds),
        },
        "output_files": {
            "train": str(out_dir / "train.jsonl"),
            "test": str(out_dir / "test.jsonl"),
            "lineage": str(out_dir / "lineage.jsonl"),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote benchmark to {out_dir}", flush=True)
    print(
        f"train={len(train_rows)} | test={len(test_rows)} | lineage={len(lineage_rows)}",
        flush=True,
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        grouped.setdefault(row["base_id"], []).append(row)
    preview_ids = sorted(grouped)[: max(0, args.preview_groups)]
    for base_id in preview_ids:
        print("\n" + "=" * 100, flush=True)
        print(f"base_id={base_id}", flush=True)
        for row in grouped[base_id]:
            print(
                f"  {row['variant_type']}: expected={row['expected_influence']} "
                f"task_type={row['task_type']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
