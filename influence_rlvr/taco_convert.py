from __future__ import annotations

import json
import re
import sys
from typing import Any

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from datasets import Dataset, load_dataset

from influence_rlvr.prompts import build_code_prompt

_PYTHON_SOLUTION_RE = re.compile(
    r"(?m)^(?:def\s+\w+\s*\(|class\s+\w+\s*[:(]|import\s+\S+|from\s+\S+\s+import\s+\S+)"
)


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


def _normalize_stdio_case(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if len(value) == 1:
            return _normalize_stdio_case(value[0])
        return "\n".join(_normalize_stdio_case(item) for item in value)
    return str(value)


def _parse_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value


def _select_python_solution(solutions: Any) -> str | None:
    if not isinstance(solutions, list):
        return None
    for solution in solutions:
        if not isinstance(solution, str):
            continue
        text = solution.strip()
        if _PYTHON_SOLUTION_RE.search(text):
            return text
    return None


def tac_try_convert_row(example: dict[str, Any]) -> dict[str, Any] | None:
    sols = _parse_json_field(example.get("solutions"))
    py_sol = _select_python_solution(sols)
    if not py_sol:
        return None

    io = _parse_json_field(example.get("input_output"))
    if not isinstance(io, dict):
        return None

    question = example.get("question") or ""
    fn_name = io.get("fn_name")
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []

    record: dict[str, Any] = {
        "prompt": build_code_prompt(question),
        "solution": py_sol,
        "task_type": "code",
        "test_list": [],
        "test_setup_code": "",
        "challenge_test_list": [],
        "code_task_format": None,
        "stdio_inputs": [],
        "stdio_outputs": [],
        "fn_name": fn_name,
    }

    if isinstance(fn_name, str) and fn_name.strip():
        tests = _build_fn_io_tests(fn_name, inputs, outputs)
        if tests:
            record["code_task_format"] = "call"
            record["test_list"] = tests
            return record

    if len(inputs) != len(outputs) or not inputs:
        return None

    record["code_task_format"] = "stdio"
    record["stdio_inputs"] = [_normalize_stdio_case(item) for item in inputs]
    record["stdio_outputs"] = [_normalize_stdio_case(item) for item in outputs]
    return record


def load_tac_code_slice(n_tac: int, *, split: str = "train") -> tuple[Dataset, int, int]:
    if n_tac <= 0:
        raise ValueError("n_tac must be positive")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    rows: list[dict[str, Any]] = []
    scanned = 0
    if split == "train":
        shard_range = range(9)
        paths = [
            f"hf://datasets/BAAI/TACO/train/data-{shard:05d}-of-00009.arrow"
            for shard in shard_range
        ]
    else:
        paths = ["hf://datasets/BAAI/TACO/test/data-00000-of-00001.arrow"]

    for path in paths:
        if len(rows) >= n_tac:
            break
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


def load_tac_mbpp_slice(n_tac: int) -> tuple[Dataset, int, int]:
    return load_tac_code_slice(n_tac, split="train")
