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


def _build_model(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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
):
    _, input_ids, attention_mask = tokenize_prompt(tokenizer, prompt_messages, device)
    plen = int(input_ids.shape[1])
    input_ids = input_ids.expand(num_samples, -1)
    attention_mask = attention_mask.expand(num_samples, -1)

    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    out = model.generate(**kwargs)
    texts = []
    lens = []
    for row in out:
        cids = row[plen:]
        lens.append(int(cids.shape[0]))
        texts.append(tokenizer.decode(cids, skip_special_tokens=True))
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
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint or _resolve_checkpoint(args.run_dir, args.checkpoint_step, None)
    if args.run_dir is not None and args.checkpoint_step is not None and ckpt is None:
        raise SystemExit(
            f"No checkpoint-{args.checkpoint_step} under {args.run_dir} (tried rlvr-output/)."
        )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    raw = load_dataset("mbpp", split=args.split)
    n = min(args.num_prompts, len(raw))
    rows = [_code_row_to_sample(raw[i]) for i in range(n)]

    model, tokenizer = _build_model(args.model_id, device)
    if ckpt is not None:
        load_adapter_checkpoint(model, ckpt)
        model.eval()

    do_sample = not args.greedy
    print(f"device={device} checkpoint={ckpt}")
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

        for sample in rows:
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
        "If cap% is high and code%/cmp% jump at larger budgets, generation was likely truncating."
    )


if __name__ == "__main__":
    main()
