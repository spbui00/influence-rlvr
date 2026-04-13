from __future__ import annotations

import json
import sys
from typing import Any

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from datasets import Dataset, load_dataset

from influence_rlvr.prompts import build_code_prompt


def _build_fn_io_tests(fn_name: str, inputs: list, outputs: list) -> list[str] | None:
    if len(inputs) != len(outputs) or not inputs:
        return None
    tests = []
    for arg_in, out_raw in zip(inputs, outputs, strict=True):
        if isinstance(arg_in, list):
            arg_str = ", ".join(repr(a) for a in arg_in)
        else:
            arg_str = repr(arg_in)
        if isinstance(out_raw, list) and len(out_raw) == 1:
            exp = out_raw[0]
        else:
            exp = out_raw
        tests.append(f"assert {fn_name}({arg_str}) == {repr(exp)}")
    return tests


def tac_try_convert_row(example: dict[str, Any]) -> dict[str, Any] | None:
    sols_raw = example.get("solutions")
    if not sols_raw:
        return None
    try:
        sols = json.loads(sols_raw) if isinstance(sols_raw, str) else sols_raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(sols, list):
        return None
    py_sol = None
    for s in sols:
        if isinstance(s, str) and "def " in s:
            py_sol = s
            break
    if not py_sol:
        return None
    io_raw = example.get("input_output")
    if not io_raw:
        return None
    try:
        io = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(io, dict) or "fn_name" not in io:
        return None
    fn_name = io["fn_name"]
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    tests = _build_fn_io_tests(fn_name, inputs, outputs)
    if not tests:
        return None
    question = example.get("question") or ""
    return {
        "prompt": build_code_prompt(question),
        "solution": py_sol,
        "test_list": tests,
        "test_setup_code": "",
        "challenge_test_list": [],
        "task_type": "code",
    }


def load_tac_mbpp_slice(n_tac: int) -> tuple[Dataset, int, int]:
    if n_tac <= 0:
        raise ValueError("n_tac must be positive")
    rows: list[dict[str, Any]] = []
    scanned = 0
    for shard in range(9):
        if len(rows) >= n_tac:
            break
        path = f"hf://datasets/BAAI/TACO/train/data-{shard:05d}-of-00009.arrow"
        ds = load_dataset("arrow", data_files=path, split="train")
        for i in range(len(ds)):
            if len(rows) >= n_tac:
                break
            scanned += 1
            converted = tac_try_convert_row(ds[i])
            if converted is None:
                continue
            rows.append(converted)
    return Dataset.from_list(rows), scanned, len(rows)
