#!/usr/bin/env python3
import argparse
import math
import statistics
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_rlvr.generation import generate_rollout_batch
from influence_rlvr.modes import GenerationBackend, VLLMConfig
from influence_rlvr.prompts import build_r1_math_prompt, extract_gsm8k_target
from influence_rlvr.rewards import (
    accuracy_reward_func,
    extract_math_final_answer,
    format_reward_func,
)
from influence_rlvr.trajectory import load_adapter_checkpoint
from influence_rlvr.utils import tokenize_prompt


def _resolve_checkpoint(run_dir: Path | None, step: int | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    if run_dir is None or step is None:
        return None
    root = run_dir / "rlvr-output" if (run_dir / "rlvr-output" / f"checkpoint-{step}").exists() else run_dir
    cand = root / f"checkpoint-{step}"
    return cand if cand.exists() else None


def _build_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def _build_model(model_id: str, device: torch.device):
    tokenizer = _build_tokenizer(model_id)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.eval()
    tid = tokenizer.pad_token_id
    model.config.pad_token_id = tid
    model.generation_config.pad_token_id = tid
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


@torch.inference_mode()
def _generate_batch(
    model,
    tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    num_samples: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    generation_backend: GenerationBackend,
    vllm_config: VLLMConfig,
    checkpoint_path: Path | None,
    model_id: str,
    seed: int | None,
):
    prompt = build_r1_math_prompt(question)
    if generation_backend == GenerationBackend.HF and seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    _, input_ids, attention_mask = tokenize_prompt(tokenizer, prompt, device)
    rollout = generate_rollout_batch(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        backend=generation_backend,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        vllm_config=vllm_config,
        adapter_path=checkpoint_path,
        model_id=model_id,
    )
    texts = rollout.texts
    lens = rollout.response_mask.sum(dim=1).detach().cpu().tolist()
    return texts, lens


def main():
    p = argparse.ArgumentParser(
        description=(
            "Sweep max_new_tokens on GSM8K-style R1 prompts; report cap hits and reward stats. "
            "Use before long runs to pick EVAL_MAX_NEW_TOKENS / gradient replay budgets."
        )
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--checkpoint", type=Path, default=None, help="PEFT adapter dir (e.g. .../checkpoint-200)")
    p.add_argument("--run-dir", type=Path, default=None, help="e.g. outputs/run6; use with --checkpoint-step")
    p.add_argument("--checkpoint-step", type=int, default=None)
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[256, 384, 512, 768, 1024],
    )
    p.add_argument("--split", default="test", help='HF datasets split, e.g. "test" or "test[:5%]"')
    p.add_argument("--num-prompts", type=int, default=24)
    p.add_argument("--num-samples", type=int, default=8, help="Rollouts per prompt (match G_TRAIN if possible)")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--backend",
        choices=[str(GenerationBackend.HF), str(GenerationBackend.VLLM)],
        default=str(GenerationBackend.HF),
    )
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument("--vllm-max-model-len", type=int, default=None)
    p.add_argument("--vllm-max-num-seqs", type=int, default=None)
    p.add_argument("--vllm-enforce-eager", action="store_true")
    p.add_argument(
        "--show-prompts",
        type=int,
        nargs="+",
        default=None,
        help="GSM8K row indices (within first num-prompts): print raw completions after the table.",
    )
    p.add_argument(
        "--show-budget",
        type=int,
        default=None,
        help="max_new_tokens for --show-prompts (default: max of --budgets).",
    )
    p.add_argument("--show-samples", type=int, default=1)
    p.add_argument("--show-chars", type=int, default=12000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generation_backend = GenerationBackend.parse(args.backend)
    ckpt = args.checkpoint or _resolve_checkpoint(args.run_dir, args.checkpoint_step, None)
    if args.run_dir is not None and args.checkpoint_step is not None and ckpt is None:
        raise SystemExit(
            f"No checkpoint-{args.checkpoint_step} under {args.run_dir} (tried rlvr-output/)."
        )

    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    n = min(args.num_prompts, len(ds))
    rows = [ds[i] for i in range(n)]

    vllm_config = VLLMConfig(
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        enforce_eager=args.vllm_enforce_eager,
    )
    if generation_backend == GenerationBackend.HF:
        model, tokenizer = _build_model(args.model_id, device)
    else:
        model = None
        tokenizer = _build_tokenizer(args.model_id)
    if ckpt is not None and model is not None:
        load_adapter_checkpoint(model, ckpt)
        model.eval()

    do_sample = not args.greedy
    print(f"device={device} checkpoint={ckpt} backend={generation_backend}")
    print(f"split={args.split!r} prompts={n} samples_per_prompt={args.num_samples}")
    print(f"do_sample={do_sample} temperature={args.temperature if do_sample else None} top_p={args.top_p if do_sample else None}")
    print()

    header = (
        f"{'budget':>6}  {'cap%':>6}  {'tok_mean':>8}  {'tok_p95':>8}  "
        f"{'fmt%':>6}  {'acc%':>6}  {'fmt+acc':>8}  {'secs':>6}"
    )
    print(header)
    print("-" * len(header))

    for budget in args.budgets:
        t0 = time.perf_counter()
        all_lens = []
        cap_hits = 0
        fmt_sum = 0.0
        acc_sum = 0.0
        pair_sum = 0.0
        total = 0

        for ex_idx, ex in enumerate(rows):
            question = ex["question"]
            gold = extract_gsm8k_target(ex["answer"])
            texts, lens = _generate_batch(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens=budget,
                num_samples=args.num_samples,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                generation_backend=generation_backend,
                vllm_config=vllm_config,
                checkpoint_path=ckpt,
                model_id=args.model_id,
                seed=args.seed + budget * 1000 + ex_idx,
            )
            completions = [[{"role": "assistant", "content": t}] for t in texts]
            fmt = format_reward_func(completions)
            acc = accuracy_reward_func(completions, [gold] * len(texts))
            for li, f, a in zip(lens, fmt, acc):
                all_lens.append(li)
                if li >= budget:
                    cap_hits += 1
                fmt_sum += float(f)
                acc_sum += float(a)
                pair_sum += float(f) + float(a)
                total += 1

        elapsed = time.perf_counter() - t0
        cap_pct = 100.0 * cap_hits / max(total, 1)
        fmt_pct = 100.0 * fmt_sum / max(total, 1)
        acc_pct = 100.0 * acc_sum / max(total, 1)
        pair_mean = pair_sum / max(total, 1)
        mean_len = statistics.mean(all_lens) if all_lens else 0.0
        if not all_lens:
            p95_len = 0.0
        else:
            s = sorted(all_lens)
            p95_len = float(s[min(len(s) - 1, max(0, math.ceil(0.95 * len(s)) - 1))])

        print(
            f"{budget:6d}  {cap_pct:5.1f}%  {mean_len:8.1f}  {p95_len:8.1f}  "
            f"{fmt_pct:5.1f}%  {acc_pct:5.1f}%  {pair_mean:8.3f}  {elapsed:6.1f}"
        )

    print()
    print(
        "cap% = share of rollouts whose completion length equals max_new_tokens (length stop).\n"
        "fmt% / acc% = mean format_reward_func / accuracy_reward_func over all rollouts.\n"
        "Pick the smallest budget where cap% drops (e.g. under ~20-30%) and fmt% / acc% stop improving much."
    )
    print(
        "If cap% stays ~100% even at 768–1024, raising the limit further usually will not fix training: "
        "the model is not emitting EOS early (degenerate / non-stopping decode). "
        "Use --show-prompts to inspect; consider lower temperature, repetition penalty, or reward/prompt changes—not only a bigger budget."
    )
    print(
        "After choosing a budget, set main_pipeline EVAL_MAX_NEW_TOKENS and "
        "REPLAY_GRADIENT_CONFIG.max_new_tokens to keep held-out eval and replay rollouts aligned."
    )

    if args.show_prompts is not None:
        show_budget = args.show_budget if args.show_budget is not None else max(args.budgets)
        print()
        print("=" * 80)
        print(
            f"RAW OUTPUTS (budget={show_budget}, samples={args.show_samples}, "
            f"truncation={args.show_chars} chars)"
        )
        print("=" * 80)
        for pi in args.show_prompts:
            if pi < 0 or pi >= n:
                print(f"\n[skip] show prompt index {pi} out of range [0, {n - 1}]")
                continue
            ex = rows[pi]
            question = ex["question"]
            gold = extract_gsm8k_target(ex["answer"])
            texts, lens = _generate_batch(
                model,
                tokenizer,
                question,
                device,
                max_new_tokens=show_budget,
                num_samples=args.show_samples,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                generation_backend=generation_backend,
                vllm_config=vllm_config,
                checkpoint_path=ckpt,
                model_id=args.model_id,
                seed=args.seed + show_budget * 1000 + pi,
            )
            for si, (text, li) in enumerate(zip(texts, lens)):
                completions = [[{"role": "assistant", "content": text}]]
                fr = float(format_reward_func(completions)[0])
                ar = float(accuracy_reward_func(completions, [gold])[0])
                parsed = extract_math_final_answer(text)
                cap_hit = li >= show_budget
                body = text if len(text) <= args.show_chars else text[: args.show_chars] + "\n... [truncated]"
                print(
                    f"\n{'#' * 80}\n"
                    f"prompt_index={pi} sample={si} new_tokens={li} cap_hit={cap_hit} "
                    f"format_reward={fr} accuracy_reward={ar} parsed={parsed!r} gold={gold!r}\n"
                )
                print("--- question ---")
                print(question)
                print("\n--- assistant completion ---")
                print(body)
                print()


if __name__ == "__main__":
    main()
