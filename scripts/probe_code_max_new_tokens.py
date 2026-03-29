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
from influence_rlvr.prompts import build_code_prompt
from influence_rlvr.rewards import _extract_python_code, mbpp_execution_reward_func
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


def _code_row_to_sample(row) -> dict:
    return {
        "task_text": row["text"],
        "prompt": build_code_prompt(row["text"]),
        "test_list": row["test_list"],
        "test_setup_code": row.get("test_setup_code") or "",
        "challenge_test_list": row.get("challenge_test_list") or [],
    }


@torch.inference_mode()
def _generate_batch_code(
    model,
    tokenizer,
    prompt_messages: list,
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
    if generation_backend == GenerationBackend.HF and seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    _, input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_messages, device)
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
            "Sweep max_new_tokens on MBPP-style prompts; report cap hits, MBPP rewards, "
            "and extractable-code rate. Match main_pipeline CODE_EVAL_* sampling when probing."
        )
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--checkpoint-step", type=int, default=None)
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[256, 384, 512, 768, 1024],
    )
    p.add_argument(
        "--split",
        default="validation[:10%]",
        help='MBPP split, e.g. "validation[:10%]" or "test[:10]"',
    )
    p.add_argument("--num-prompts", type=int, default=24)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
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
        help="Indices into the first num-prompts MBPP rows: print raw completions (after table).",
    )
    p.add_argument(
        "--show-budget",
        type=int,
        default=None,
        help="max_new_tokens for --show-prompts (default: max of --budgets).",
    )
    p.add_argument("--show-samples", type=int, default=1)
    p.add_argument(
        "--show-chars",
        type=int,
        default=12000,
        help="Truncate each printed completion to this many characters.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generation_backend = GenerationBackend.parse(args.backend)
    ckpt = args.checkpoint or _resolve_checkpoint(args.run_dir, args.checkpoint_step, None)
    if args.run_dir is not None and args.checkpoint_step is not None and ckpt is None:
        raise SystemExit(
            f"No checkpoint-{args.checkpoint_step} under {args.run_dir} (tried rlvr-output/)."
        )

    raw = load_dataset("mbpp", split=args.split)
    n = min(args.num_prompts, len(raw))
    rows = [_code_row_to_sample(raw[i]) for i in range(n)]

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
        f"{'code%':>6}  {'cmp%':>6}  {'pass%':>6}  {'rew_mean':>8}  {'secs':>6}"
    )
    print(header)
    print("-" * len(header))

    for budget in args.budgets:
        t0 = time.perf_counter()
        all_lens = []
        cap_hits = 0
        code_nonempty = 0
        compile_hits = 0
        pass_hits = 0
        rew_sum = 0.0
        total = 0

        for sample_idx, sample in enumerate(rows):
            texts, lens = _generate_batch_code(
                model,
                tokenizer,
                sample["prompt"],
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
                seed=args.seed + budget * 1000 + sample_idx,
            )
            completions = [[{"role": "assistant", "content": t}] for t in texts]
            rewards = mbpp_execution_reward_func(
                completions,
                test_list=sample["test_list"],
                test_setup_code=sample["test_setup_code"],
                challenge_test_list=sample["challenge_test_list"],
            )
            for text, li, rw in zip(texts, lens, rewards):
                all_lens.append(li)
                if li >= budget:
                    cap_hits += 1
                if _extract_python_code(text):
                    code_nonempty += 1
                if rw > 0.0:
                    compile_hits += 1
                if rw >= 0.999:
                    pass_hits += 1
                rew_sum += float(rw)
                total += 1

        elapsed = time.perf_counter() - t0
        cap_pct = 100.0 * cap_hits / max(total, 1)
        code_pct = 100.0 * code_nonempty / max(total, 1)
        cmp_pct = 100.0 * compile_hits / max(total, 1)
        pass_pct = 100.0 * pass_hits / max(total, 1)
        rew_mean = rew_sum / max(total, 1)
        mean_len = statistics.mean(all_lens) if all_lens else 0.0
        if not all_lens:
            p95_len = 0.0
        else:
            s = sorted(all_lens)
            p95_len = float(s[min(len(s) - 1, max(0, math.ceil(0.95 * len(s)) - 1))])

        print(
            f"{budget:6d}  {cap_pct:5.1f}%  {mean_len:8.1f}  {p95_len:8.1f}  "
            f"{code_pct:5.1f}%  {cmp_pct:5.1f}%  {pass_pct:5.1f}%  {rew_mean:8.3f}  {elapsed:6.1f}"
        )

    print()
    print(
        "cap% = fraction of rollouts with completion length == max_new_tokens (length stop).\n"
        "code% = fraction with extractable Python (fenced or bare def/import block).\n"
        "cmp% / pass% = fraction with MBPP reward > 0 / >= 0.999 (same spirit as evaluate_code_dataset).\n"
        "If cap% is high and code%/cmp% jump at larger budgets, generation was likely truncating.\n"
        "Run the same sweep with --backend hf and --backend vllm to compare sampler behavior directly."
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
            sample = rows[pi]
            texts, lens = _generate_batch_code(
                model,
                tokenizer,
                sample["prompt"],
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
                rw = mbpp_execution_reward_func(
                    completions,
                    test_list=sample["test_list"],
                    test_setup_code=sample["test_setup_code"],
                    challenge_test_list=sample["challenge_test_list"],
                )[0]
                extracted = _extract_python_code(text) or ""
                cap_hit = li >= show_budget
                body = text if len(text) <= args.show_chars else text[: args.show_chars] + "\n... [truncated]"
                ext_preview = (
                    extracted
                    if len(extracted) <= 2000
                    else extracted[:2000] + "\n... [extracted truncated]"
                )
                print(f"\n{'#' * 80}\nprompt_index={pi} sample={si} new_tokens={li} cap_hit={cap_hit} reward={rw:.4f}\n")
                print("--- MBPP task ---")
                print(sample["task_text"])
                print("\n--- extract_python_code ---")
                print(ext_preview or "(empty)")
                print("\n--- assistant completion ---")
                print(body)
                print()


if __name__ == "__main__":
    main()
