from __future__ import annotations

import numpy as np
import torch

from .generation import generate_rollout_batch, rollout_to_completions
from .modes import GenerationBackend, VLLMConfig
from .rewards import accuracy_reward_func, mbpp_execution_reward_func
from .utils import tokenize_prompt


def _resolve_generation_backend(enable_vllm, generation_backend):
    if generation_backend is None:
        return GenerationBackend.VLLM if enable_vllm else GenerationBackend.HF
    backend = GenerationBackend.parse(generation_backend)
    if enable_vllm and backend != GenerationBackend.VLLM:
        raise ValueError(
            "Received enable_vllm=True with generation_backend set to a non-vLLM backend."
        )
    return backend


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
    enable_vllm=False,
    generation_backend=None,
    vllm_config=None,
    adapter_path=None,
    model_id=None,
):
    if peft_model is not None:
        peft_model.eval()
    backend = _resolve_generation_backend(enable_vllm, generation_backend)
    _, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)
    rollout = generate_rollout_batch(
        peft_model,
        tokenizer,
        prompt_ids,
        prompt_attention_mask,
        backend=backend,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        vllm_config=VLLMConfig() if vllm_config is None else vllm_config,
        adapter_path=adapter_path,
        model_id=model_id,
    )
    return rollout_to_completions(rollout)


def evaluate_math_dataset(
    peft_model,
    tokenizer,
    dataset,
    device,
    limit=None,
    max_new_tokens=256,
    progress=False,
    progress_prefix="",
    enable_vllm=False,
    generation_backend=None,
    vllm_config=None,
    adapter_path=None,
    model_id=None,
):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    accuracy_scores = []

    for idx in range(count):
        sample = dataset[idx]
        completions = _generate_completions(
            peft_model,
            tokenizer,
            sample["prompt"],
            device,
            max_new_tokens=max_new_tokens,
            enable_vllm=enable_vllm,
            generation_backend=generation_backend,
            vllm_config=vllm_config,
            adapter_path=adapter_path,
            model_id=model_id,
        )
        accuracy_score = float(accuracy_reward_func(completions, [sample["solution"]])[0])
        accuracy_scores.append(accuracy_score)
        if progress:
            print(
                f"{progress_prefix} math eval sample {idx + 1}/{count} "
                f"| acc={accuracy_score:.1f}",
                flush=True,
            )

    mean_acc = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
    return {
        "count": count,
        "accuracy_rate": mean_acc,
        "mean_reward": mean_acc,
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
    enable_vllm=False,
    generation_backend=None,
    vllm_config=None,
    adapter_path=None,
    model_id=None,
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
            enable_vllm=enable_vllm,
            generation_backend=generation_backend,
            vllm_config=vllm_config,
            adapter_path=adapter_path,
            model_id=model_id,
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
