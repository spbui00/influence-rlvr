from __future__ import annotations

import numpy as np
import torch

from .rewards import accuracy_reward_func, format_reward_func, mbpp_execution_reward_func
from .utils import tokenize_prompt


def _generate_completions(
    peft_model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=256,
    *,
    do_sample=False,
    temperature=0.6,
    top_p=0.95,
    num_samples=1,
):
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    if not do_sample and num_samples != 1:
        raise ValueError(
            "Greedy evaluation only supports num_samples=1. "
            "Set do_sample=True to evaluate multiple completions."
        )

    peft_model.eval()
    _, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)
    with torch.inference_mode():
        generate_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": num_samples,
            })
        generated = peft_model.generate(
            **generate_kwargs,
        )
    completions = []
    for sequence in generated[:num_samples]:
        completion_ids = sequence[prompt_ids.shape[1]:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completions.append([{"role": "assistant", "content": completion_text}])
    return completions


def evaluate_math_dataset(
    peft_model,
    tokenizer,
    dataset,
    device,
    limit=None,
    max_new_tokens=256,
    progress=False,
    progress_prefix="",
):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    format_scores = []
    accuracy_scores = []
    total_scores = []

    for idx in range(count):
        sample = dataset[idx]
        completions = _generate_completions(
            peft_model,
            tokenizer,
            sample["prompt"],
            device,
            max_new_tokens=max_new_tokens,
        )
        format_score = float(format_reward_func(completions)[0])
        accuracy_score = float(accuracy_reward_func(completions, [sample["solution"]])[0])
        format_scores.append(format_score)
        accuracy_scores.append(accuracy_score)
        total_scores.append(format_score + accuracy_score)
        if progress:
            print(
                f"{progress_prefix} math eval sample {idx + 1}/{count} "
                f"| format={format_score:.1f} acc={accuracy_score:.1f}",
                flush=True,
            )

    return {
        "count": count,
        "format_rate": float(np.mean(format_scores)) if format_scores else 0.0,
        "accuracy_rate": float(np.mean(accuracy_scores)) if accuracy_scores else 0.0,
        "mean_reward": float(np.mean(total_scores)) if total_scores else 0.0,
    }


def evaluate_code_dataset(
    peft_model,
    tokenizer,
    dataset,
    device,
    limit=None,
    max_new_tokens=256,
    progress=False,
    progress_prefix="",
    do_sample=False,
    temperature=0.6,
    top_p=0.95,
    num_samples=1,
):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    rewards = []
    pass_flags = []
    compile_flags = []

    for idx in range(count):
        sample = dataset[idx]
        completions = _generate_completions(
            peft_model,
            tokenizer,
            sample["prompt"],
            device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_samples=num_samples,
        )
        completion_rewards = [
            float(score)
            for score in mbpp_execution_reward_func(
                completions,
                test_list=sample["test_list"],
                test_setup_code=sample["test_setup_code"],
                challenge_test_list=sample.get("challenge_test_list"),
            )
        ]
        reward = max(completion_rewards, default=0.0)
        rewards.append(reward)
        pass_flags.append(
            1.0 if any(score >= 0.999 for score in completion_rewards) else 0.0
        )
        compile_flags.append(
            1.0 if any(score > 0.0 for score in completion_rewards) else 0.0
        )
        if progress:
            print(
                f"{progress_prefix} code eval sample {idx + 1}/{count} "
                f"| best_reward={reward:.3f}",
                flush=True,
            )

    pass_metric = "pass@1" if num_samples == 1 else f"pass@{num_samples}"
    compile_metric = (
        "compile@1" if num_samples == 1 else f"compile@{num_samples}"
    )
    return {
        "count": count,
        "pass_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        "compile_rate": float(np.mean(compile_flags)) if compile_flags else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "pass_metric": pass_metric,
        "compile_metric": compile_metric,
        "sample_count": int(num_samples),
        "do_sample": bool(do_sample),
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
    }
