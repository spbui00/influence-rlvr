from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets import Dataset, concatenate_datasets, load_dataset

from influence_rlvr.prompts import (
    append_suffix_to_final_user_message,
    build_r1_math_prompt,
    extract_gsm8k_target,
)
from influence_rlvr.rewards import (
    accuracy_reward_func,
    extract_math_final_answer,
    format_guardrail_reward_func,
    mixed_code_execution_grpo_reward,
    mixed_format_guardrail_grpo_reward,
    mixed_math_accuracy_grpo_reward,
)
from influence_rlvr.taco_convert import load_tac_code_slice

NUMINA_DATASET = "AI-MO/NuminaMath-CoT"

FORMAT_SUFFIX = (
    "After </think>, the last line must contain only the final numeric GSM8K answer in "
    "\\boxed{...} (digits / fraction / decimal). Do not write placeholders, "
    "do not repeat this instruction block, and do not use code fences."
)


def build_training_script_math_prompt(question: str) -> list[dict[str, str]]:
    messages = build_r1_math_prompt(question)
    return append_suffix_to_final_user_message(messages, FORMAT_SUFFIX)


def format_gsm8k_train_row(example, idx):
    return {
        "prompt": build_training_script_math_prompt(example["question"]),
        "solution": extract_gsm8k_target(example["answer"]),
        "train_index": idx,
    }


def format_numina_mixed_row(example, _idx):
    problem = example["problem"]
    reference_solution = example.get("solution") or ""
    gold = extract_math_final_answer(reference_solution) or ""
    return {
        "prompt": build_training_script_math_prompt(problem),
        "solution": gold,
        "test_list": [],
        "test_setup_code": "",
        "challenge_test_list": [],
        "task_type": "math",
    }


@dataclass(frozen=True)
class TrainingDataBundle:
    mode_name: str
    train_dataset: Dataset
    reward_funcs: list[Callable]
    eval_mode: str


def _load_gsm8k_train_dataset(args) -> Dataset:
    train_split = "train" if args.n_math <= 0 else f"train[:{args.n_math}]"
    raw = load_dataset("openai/gsm8k", "main", split=train_split)
    return raw.map(format_gsm8k_train_row, with_indices=True)


def _load_mixed_train_dataset(args) -> Dataset:
    if args.n_numina <= 0 or args.n_taco <= 0:
        raise ValueError("Mixed training requires --n-numina > 0 and --n-taco > 0.")

    numina_files = [
        f"hf://datasets/{args.numina_dataset}/data/train-{i:05d}-of-00005.parquet"
        for i in range(5)
    ]
    split = f"train[:{args.n_numina}]"
    numina_raw = load_dataset("parquet", data_files=numina_files, split=split)
    numina_ds = numina_raw.map(
        format_numina_mixed_row,
        with_indices=True,
        remove_columns=numina_raw.column_names,
    )
    before = len(numina_ds)
    numina_ds = numina_ds.filter(lambda x: bool(str(x.get("solution", "")).strip()))
    print(
        f"  NuminaMath-CoT: {len(numina_ds)} / {before} rows with extractable gold "
        f"(slice up to {args.n_numina})."
    )

    taco_ds, scanned, kept = load_tac_code_slice(args.n_taco, split="train")
    print(
        f"  TACO: {kept} native-executable rows (scanned {scanned} raw, target {args.n_taco})."
    )

    dataset = concatenate_datasets([numina_ds, taco_ds])
    dataset = dataset.shuffle(seed=args.mixed_shuffle_seed)
    dataset = dataset.map(lambda _, idx: {"train_index": idx}, with_indices=True)
    return dataset


def load_training_data_bundle(args) -> TrainingDataBundle:
    if args.mixed:
        return TrainingDataBundle(
            mode_name="mixed",
            train_dataset=_load_mixed_train_dataset(args),
            reward_funcs=[
                mixed_format_guardrail_grpo_reward,
                mixed_math_accuracy_grpo_reward,
                mixed_code_execution_grpo_reward,
            ],
            eval_mode="mixed",
        )

    return TrainingDataBundle(
        mode_name="gsm8k",
        train_dataset=_load_gsm8k_train_dataset(args),
        reward_funcs=[format_guardrail_reward_func, accuracy_reward_func],
        eval_mode="gsm8k",
    )
