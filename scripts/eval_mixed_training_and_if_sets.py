#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from influence_rlvr.generation import clear_vllm_engine_cache
from influence_rlvr.modes import GenerationBackend, VLLMConfig
from influence_rlvr.prompts import (
    append_suffix_to_final_user_message,
    build_code_prompt,
    build_r1_math_prompt,
)
from influence_rlvr.rewards import (
    accuracy_reward_func,
    extract_math_final_answer,
    humaneval_best_reward_for_response,
    mbpp_execution_rewards_and_codes,
)
from influence_rlvr.taco_convert import tac_try_convert_row
from influence_rlvr.utils import detect_device
from influence_rlvr.eval import _generate_completions


FORMAT_SUFFIX = (
    "After </think>, the last line must contain only the final numeric GSM8K answer in "
    "\\boxed{...} (digits / fraction / decimal). Do not write placeholders, "
    "do not repeat this instruction block, and do not use code fences."
)
NUMINA_DATASET = "AI-MO/NuminaMath-CoT"
HUMANEVAL_DATASET = "openai/openai_humaneval"


def _build_math_prompt(question: str) -> list[dict[str, str]]:
    return append_suffix_to_final_user_message(
        build_r1_math_prompt(question),
        FORMAT_SUFFIX,
    )


def _format_numina(example: dict[str, Any]) -> dict[str, Any]:
    gold = extract_math_final_answer(example.get("solution") or "") or ""
    return {
        "prompt": _build_math_prompt(example["problem"]),
        "solution": gold,
        "problem": example["problem"],
    }


def _format_humaneval(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": build_code_prompt(example["prompt"]),
        "prompt_prefix": example["prompt"],
        "test": example["test"],
        "entry_point": example["entry_point"],
        "task_id": example.get("task_id", ""),
    }


def load_numina_test_dataset(limit: int) -> Dataset:
    split = "train" if limit <= 0 else f"train[:{limit}]"
    raw = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{NUMINA_DATASET}/data/test-00000-of-00001.parquet",
        split=split,
    )
    ds = raw.map(_format_numina, remove_columns=raw.column_names)
    return ds.filter(lambda x: bool(str(x.get("solution", "")).strip()))


def _list_tac_test_arrow_uris() -> list[str]:
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        paths = sorted(fs.glob("datasets/BAAI/TACO/test/*.arrow"))
        return [f"hf://{p}" for p in paths]
    except Exception:
        return []


def load_taco_test_dataset(limit: int) -> tuple[Dataset, int, int]:
    if limit <= 0:
        raise ValueError("limit must be positive for TACO test evaluation")
    rows: list[dict[str, Any]] = []
    scanned = 0
    uris = _list_tac_test_arrow_uris()
    if not uris:
        uris = ["hf://datasets/BAAI/TACO/test/data-00000-of-00001.arrow"]
    for uri in uris:
        if len(rows) >= limit:
            break
        try:
            ds = load_dataset("arrow", data_files=uri, split="train")
        except Exception:
            continue
        for i in range(len(ds)):
            if len(rows) >= limit:
                break
            scanned += 1
            converted = tac_try_convert_row(ds[i])
            if converted is None:
                continue
            rows.append(converted)
    return Dataset.from_list(rows), scanned, len(rows)


def load_humaneval_dataset(limit: int) -> Dataset:
    split = "test" if limit <= 0 else f"test[:{limit}]"
    raw = load_dataset(HUMANEVAL_DATASET, split=split)
    return raw.map(_format_humaneval, remove_columns=raw.column_names)


def _resolve_generation_backend(use_vllm: bool) -> GenerationBackend:
    return GenerationBackend.VLLM if use_vllm else GenerationBackend.HF


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run_config.json under {run_dir}")
    return json.loads(path.read_text())


def _resolve_checkpoint_dir(run_dir: Path, run_cfg: dict[str, Any], checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is None or checkpoint_arg == "latest":
        latest = get_last_checkpoint(str(run_dir))
        if latest is not None:
            return Path(latest).resolve()
        max_steps = run_cfg.get("max_steps")
        if max_steps is not None:
            return (run_dir / f"checkpoint-{int(max_steps)}").resolve()
        raise FileNotFoundError(
            f"Could not infer a checkpoint directory under {run_dir}. "
            "Pass --checkpoint-dir explicitly."
        )
    return Path(checkpoint_arg).expanduser().resolve()


def _load_model_and_tokenizer(checkpoint_dir: Path, run_cfg: dict[str, Any], device: torch.device):
    model_id = run_cfg.get("model_id", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir)).to(device)
    model.eval()
    return model, tokenizer


def _math_pass_at_k(completions: list[list[dict[str, str]]], gold: str) -> tuple[float, list[str], list[float]]:
    responses = [item[0]["content"] for item in completions]
    rewards = [
        float(accuracy_reward_func([[{"role": "assistant", "content": response}]], [gold])[0])
        for response in responses
    ]
    passed = any(score >= 0.999 for score in rewards)
    return (1.0 if passed else 0.0), responses, rewards


def _print_example_block(dataset_name: str, row: dict[str, Any]) -> None:
    print("\n" + "=" * 100, flush=True)
    print(f"{dataset_name} example {row['index']}", flush=True)
    for key in ("task_id", "gold", "pass", "compile", "rewards", "parsed_answers", "entry_point"):
        if key in row:
            print(f"{key}: {row[key]}", flush=True)
    for text_key in ("problem", "prompt_prefix", "prompt"):
        if text_key in row:
            print(f"--- {text_key} ---\n{row[text_key]}", flush=True)
    responses = row.get("responses") or []
    extracted_codes = row.get("extracted_codes") or []
    for ridx, response in enumerate(responses, start=1):
        print(f"--- response[{ridx}] ---\n{response}", flush=True)
        if ridx - 1 < len(extracted_codes):
            print(f"--- extracted_code[{ridx}] ---\n{extracted_codes[ridx - 1]}", flush=True)


def evaluate_numina_test(
    model,
    tokenizer,
    dataset: Dataset,
    *,
    device: torch.device,
    num_samples: int,
    max_new_tokens: int,
    use_vllm: bool,
    vllm_config: VLLMConfig,
    checkpoint_dir: Path,
    model_id: str,
    output_path: Path,
) -> dict[str, Any]:
    rows = []
    pass_flags = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        completions = _generate_completions(
            model,
            tokenizer,
            sample["prompt"],
            device,
            max_new_tokens=max_new_tokens,
            do_sample=num_samples > 1,
            temperature=0.7,
            top_p=0.95,
            num_samples=num_samples,
            seed=idx,
            enable_vllm=use_vllm,
            generation_backend=_resolve_generation_backend(use_vllm),
            vllm_config=vllm_config,
            adapter_path=str(checkpoint_dir),
            model_id=model_id,
        )
        pass_score, responses, rewards = _math_pass_at_k(completions, sample["solution"])
        pass_flags.append(pass_score)
        rows.append({
            "index": idx,
            "gold": sample["solution"],
            "problem": sample["problem"],
            "pass": pass_score,
            "rewards": rewards,
            "parsed_answers": [extract_math_final_answer(resp) for resp in responses],
            "responses": responses,
        })
        _print_example_block("numina_test", rows[-1])

    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    return {
        "count": len(rows),
        "pass_metric": f"pass@{num_samples}",
        "pass_rate": float(sum(pass_flags) / len(pass_flags)) if pass_flags else 0.0,
        "responses_path": str(output_path),
    }


def evaluate_taco_test(
    model,
    tokenizer,
    dataset: Dataset,
    *,
    device: torch.device,
    num_samples: int,
    max_new_tokens: int,
    use_vllm: bool,
    vllm_config: VLLMConfig,
    checkpoint_dir: Path,
    model_id: str,
    output_path: Path,
) -> dict[str, Any]:
    rows = []
    pass_flags = []
    compile_flags = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        completions = _generate_completions(
            model,
            tokenizer,
            sample["prompt"],
            device,
            max_new_tokens=max_new_tokens,
            do_sample=num_samples > 1,
            temperature=0.7,
            top_p=0.95,
            num_samples=num_samples,
            seed=idx,
            enable_vllm=use_vllm,
            generation_backend=_resolve_generation_backend(use_vllm),
            vllm_config=vllm_config,
            adapter_path=str(checkpoint_dir),
            model_id=model_id,
        )
        rewards, extracted_codes = mbpp_execution_rewards_and_codes(
            completions,
            test_list=sample["test_list"],
            test_setup_code=sample["test_setup_code"],
            challenge_test_list=sample.get("challenge_test_list"),
        )
        pass_score = 1.0 if any(score >= 0.999 for score in rewards) else 0.0
        compile_score = 1.0 if any(score > 0.0 for score in rewards) else 0.0
        pass_flags.append(pass_score)
        compile_flags.append(compile_score)
        rows.append({
            "index": idx,
            "prompt": sample["prompt"],
            "pass": pass_score,
            "compile": compile_score,
            "rewards": [float(x) for x in rewards],
            "responses": [item[0]["content"] for item in completions],
            "extracted_codes": extracted_codes,
            "test_list": sample["test_list"],
        })
        _print_example_block("taco_test", rows[-1])

    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    return {
        "count": len(rows),
        "pass_metric": f"pass@{num_samples}",
        "compile_metric": f"compile@{num_samples}",
        "pass_rate": float(sum(pass_flags) / len(pass_flags)) if pass_flags else 0.0,
        "compile_rate": float(sum(compile_flags) / len(compile_flags)) if compile_flags else 0.0,
        "responses_path": str(output_path),
    }


def evaluate_humaneval(
    model,
    tokenizer,
    dataset: Dataset,
    *,
    device: torch.device,
    num_samples: int,
    max_new_tokens: int,
    use_vllm: bool,
    vllm_config: VLLMConfig,
    checkpoint_dir: Path,
    model_id: str,
    output_path: Path,
) -> dict[str, Any]:
    rows = []
    pass_flags = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        completions = _generate_completions(
            model,
            tokenizer,
            sample["prompt"],
            device,
            max_new_tokens=max_new_tokens,
            do_sample=num_samples > 1,
            temperature=0.7,
            top_p=0.95,
            num_samples=num_samples,
            seed=idx,
            enable_vllm=use_vllm,
            generation_backend=_resolve_generation_backend(use_vllm),
            vllm_config=vllm_config,
            adapter_path=str(checkpoint_dir),
            model_id=model_id,
        )
        responses = [item[0]["content"] for item in completions]
        rewards = [
            humaneval_best_reward_for_response(
                response,
                sample["prompt_prefix"],
                sample["test"],
                sample["entry_point"],
            )
            for response in responses
        ]
        pass_score = 1.0 if any(score >= 0.999 for score in rewards) else 0.0
        pass_flags.append(pass_score)
        rows.append({
            "index": idx,
            "task_id": sample["task_id"],
            "prompt_prefix": sample["prompt_prefix"],
            "pass": pass_score,
            "rewards": [float(x) for x in rewards],
            "responses": responses,
            "entry_point": sample["entry_point"],
        })
        _print_example_block("humaneval_test", rows[-1])

    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    return {
        "count": len(rows),
        "pass_metric": f"pass@{num_samples}",
        "pass_rate": float(sum(pass_flags) / len(pass_flags)) if pass_flags else 0.0,
        "responses_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a mixed-run checkpoint on test splits only "
            "(Numina test, TACO test, HumanEval test), saving and printing raw responses."
        )
    )
    parser.add_argument("--rlvr-output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="latest",
        help="Checkpoint path to evaluate, or 'latest' to auto-resolve under --rlvr-output.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--n-numina", type=int, default=200)
    parser.add_argument("--n-taco", type=int, default=200)
    parser.add_argument("--n-humaneval", type=int, default=None)
    parser.add_argument("--hf", action="store_true", help="Use HF generation instead of vLLM.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the summary JSON. Defaults under --rlvr-output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.rlvr_output.expanduser().resolve()
    run_cfg = _load_run_config(run_dir)

    checkpoint_dir = _resolve_checkpoint_dir(run_dir, run_cfg, args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint_dir}")

    model_id = str(run_cfg.get("model_id", "HuggingFaceTB/SmolLM2-1.7B-Instruct"))
    numina_n = int(args.n_numina)
    taco_n = int(args.n_taco)
    humaneval_n = args.n_humaneval if args.n_humaneval is not None else int(run_cfg.get("n_humaneval", 500))
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(run_cfg.get("max_completion_length", 1024))
    )
    use_vllm = not args.hf and not bool(run_cfg.get("hf", False))

    device = detect_device()
    vllm_config = VLLMConfig(
        gpu_memory_utilization=float(run_cfg.get("vllm_gpu_memory_utilization", 0.45)),
        tensor_parallel_size=int(run_cfg.get("vllm_tensor_parallel_size", 1)),
        max_model_len=(
            int(run_cfg["vllm_max_model_length"])
            if run_cfg.get("vllm_max_model_length") is not None
            else None
        ),
        max_lora_rank=int(run_cfg.get("lora_r", 128)),
        training_use_vllm=use_vllm,
    )

    model, tokenizer = _load_model_and_tokenizer(checkpoint_dir, run_cfg, device)

    print("Loading eval datasets...", flush=True)
    numina_test = load_numina_test_dataset(numina_n)
    taco_test, taco_scanned, taco_kept = load_taco_test_dataset(taco_n)
    humaneval_ds = load_humaneval_dataset(humaneval_n)
    print(
        f"Numina test={len(numina_test)} | "
        f"TACO test={len(taco_test)} (scanned={taco_scanned}, kept={taco_kept}) | "
        f"HumanEval={len(humaneval_ds)}",
        flush=True,
    )

    summary_path = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else run_dir / f"eval_mixed_sets_{checkpoint_dir.name}.json"
    )
    responses_dir = summary_path.parent / f"{summary_path.stem}_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "num_samples": int(args.num_samples),
        "max_new_tokens": int(max_new_tokens),
        "use_vllm": bool(use_vllm),
        "datasets": {},
    }

    summary["datasets"]["numina_test"] = evaluate_numina_test(
        model,
        tokenizer,
        numina_test,
        device=device,
        num_samples=args.num_samples,
        max_new_tokens=max_new_tokens,
        use_vllm=use_vllm,
        vllm_config=vllm_config,
        checkpoint_dir=checkpoint_dir,
        model_id=model_id,
        output_path=responses_dir / "numina_test.json",
    )
    print(
        f"Numina test {summary['datasets']['numina_test']['pass_metric']}: "
        f"{summary['datasets']['numina_test']['pass_rate']:.4f}",
        flush=True,
    )

    summary["datasets"]["taco_test"] = evaluate_taco_test(
        model,
        tokenizer,
        taco_test,
        device=device,
        num_samples=args.num_samples,
        max_new_tokens=max_new_tokens,
        use_vllm=use_vllm,
        vllm_config=vllm_config,
        checkpoint_dir=checkpoint_dir,
        model_id=model_id,
        output_path=responses_dir / "taco_test.json",
    )
    print(
        f"TACO test {summary['datasets']['taco_test']['pass_metric']}: "
        f"{summary['datasets']['taco_test']['pass_rate']:.4f} | "
        f"{summary['datasets']['taco_test']['compile_metric']}: "
        f"{summary['datasets']['taco_test']['compile_rate']:.4f}",
        flush=True,
    )

    summary["datasets"]["humaneval_test"] = evaluate_humaneval(
        model,
        tokenizer,
        humaneval_ds,
        device=device,
        num_samples=args.num_samples,
        max_new_tokens=max_new_tokens,
        use_vllm=use_vllm,
        vllm_config=vllm_config,
        checkpoint_dir=checkpoint_dir,
        model_id=model_id,
        output_path=responses_dir / "humaneval_test.json",
    )
    print(
        f"HumanEval {summary['datasets']['humaneval_test']['pass_metric']}: "
        f"{summary['datasets']['humaneval_test']['pass_rate']:.4f}",
        flush=True,
    )

    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote summary: {summary_path}", flush=True)
    print(f"Wrote responses under: {responses_dir}", flush=True)

    clear_vllm_engine_cache()


if __name__ == "__main__":
    main()
