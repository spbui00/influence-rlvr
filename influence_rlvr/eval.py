from __future__ import annotations

import numpy as np
import torch

from .rewards import accuracy_reward_func, format_reward_func, mbpp_execution_reward_func
from .utils import clear_cache, tokenize_prompt


def _generate_completion(
    peft_model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=256,
):
    peft_model.eval()
    _, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)
    with torch.no_grad():
        generated = peft_model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0, prompt_ids.shape[1]:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return [[{"role": "assistant", "content": completion_text}]]


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
        completions = _generate_completion(
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
        clear_cache(device)

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
):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    rewards = []
    pass_flags = []
    compile_flags = []

    for idx in range(count):
        sample = dataset[idx]
        completions = _generate_completion(
            peft_model,
            tokenizer,
            sample["prompt"],
            device,
            max_new_tokens=max_new_tokens,
        )
        reward = float(
            mbpp_execution_reward_func(
                completions,
                test_list=sample["test_list"],
                test_setup_code=sample["test_setup_code"],
                challenge_test_list=sample.get("challenge_test_list"),
            )[0]
        )
        rewards.append(reward)
        pass_flags.append(1.0 if reward >= 0.999 else 0.0)
        compile_flags.append(1.0 if reward > 0.0 else 0.0)
        if progress:
            print(
                f"{progress_prefix} code eval sample {idx + 1}/{count} "
                f"| reward={reward:.3f}",
                flush=True,
            )
        clear_cache(device)

    return {
        "count": count,
        "pass_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        "compile_rate": float(np.mean(compile_flags)) if compile_flags else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }
