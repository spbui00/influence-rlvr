#!/usr/bin/env python3
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from influence_rlvr import (
    HistoricalBatchGRPOTrainer,
    accuracy_reward_func,
    clear_cache,
    detect_device,
)
from influence_rlvr.prompts import build_r1_math_prompt, extract_gsm8k_target
from influence_rlvr.rewards import extract_math_final_answer


FORMAT_SUFFIX = (
    "After </think>, the last line must contain only the final numeric GSM8K answer in "
    "\\boxed{...} (digits / fraction / decimal). Do not write placeholders, "
    "do not repeat this instruction block, and do not use code fences."
)


def _as_completion(text: str):
    return [[{"role": "assistant", "content": text}]]


def _build_training_script_math_prompt(question: str) -> list[dict[str, str]]:
    messages = build_r1_math_prompt(question)
    content = messages[0]["content"]
    return [{"role": "user", "content": f"{content}\n\n{FORMAT_SUFFIX}"}]


@torch.inference_mode()
def _generate_completion(
    model,
    tokenizer,
    messages: list[dict],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    model.eval()
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
    attention_mask = None if isinstance(enc, torch.Tensor) else enc.get("attention_mask")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)


def run_gsm8k_eval(
    model,
    tokenizer,
    device,
    args,
    *,
    phase: str,
    summary_filename: str,
) -> None:
    n = args.eval_examples
    if n <= 0:
        return
    split = f"test[:{n}]"
    title = (
        "Baseline eval (before GRPO)"
        if phase == "baseline"
        else "Post-training eval"
    )
    print("\n" + "=" * 80)
    print(f"{title} — GSM8K {split}, greedy HF generate")
    print("=" * 80)
    eval_ds = load_dataset("openai/gsm8k", "main", split=split)
    acc_scores = []
    rows = []
    for i, ex in enumerate(eval_ds):
        messages = _build_training_script_math_prompt(ex["question"])
        gold = extract_gsm8k_target(ex["answer"])
        text = _generate_completion(
            model,
            tokenizer,
            messages,
            device=device,
            max_new_tokens=args.eval_max_new_tokens,
        )
        c = _as_completion(text)
        a = accuracy_reward_func(c, [gold])[0]
        acc_scores.append(a)
        pred = extract_math_final_answer(text)
        rows.append({
            "index": i,
            "gold": gold,
            "pred_parsed": pred,
            "accuracy_reward": a,
            "response": text,
            "question": ex["question"],
        })

    mean_acc = sum(acc_scores) / len(acc_scores)
    print(f"  mean accuracy_reward: {mean_acc:.4f}")
    summary_path = args.output_dir.resolve() / summary_filename
    summary_path.write_text(
        json.dumps(
            {
                "phase": phase,
                "split": split,
                "mean_accuracy_reward": mean_acc,
                "per_example": [
                    {
                        "index": r["index"],
                        "gold": r["gold"],
                        "pred_parsed": r["pred_parsed"],
                        "accuracy_reward": r["accuracy_reward"],
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  Wrote {summary_path}")

    k = min(args.inspect_examples, len(rows))
    if k <= 0:
        return
    print("\n" + "-" * 80)
    print(f"Sample responses (first {k} eval examples; full text)")
    print("-" * 80)
    for r in rows[:k]:
        print(f"\n### [{r['index']}] accuracy={r['accuracy_reward']}")
        print(f"gold={r['gold']!r} pred_parsed={r['pred_parsed']!r}")
        print(f"--- question ---\n{r['question']}\n--- response ---\n{r['response']}\n")


def save_base_checkpoint(peft_model, tokenizer_obj, output_dir: Path) -> Path:
    checkpoint_dir = output_dir / "checkpoint-0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(checkpoint_dir)
    tokenizer_obj.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 0,
                "log_history": [{"step": 0, "learning_rate": 0.0}],
            },
            indent=2,
        )
    )
    return checkpoint_dir


def format_math(example, idx):
    return {
        "prompt": _build_training_script_math_prompt(example["question"]),
        "solution": extract_gsm8k_target(example["answer"]),
        "train_index": idx,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Standalone GRPO training (GSM8K math) matching main_pipeline Phase 1. "
            "Default: vLLM colocated generation via TRL (requires Linux + CUDA + vllm extra)."
        ),
        epilog=(
            "vLLM modes: 'colocate' runs vLLM in-process on the training GPU(s). "
            "'server' expects a TRL vLLM server (see `trl vllm-serve`). "
            "Use --hf to use transformers generate only."
        ),
    )
    p.add_argument(
        "--model-id",
        default="nvidia/Nemotron-Mini-4B-Instruct",
        help="HF model id (default: Nemotron-Mini-4B-Instruct; 4k context).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/train_script/rlvr-output"),
        help="Checkpoints and batch history are written here.",
    )
    p.add_argument(
        "--n-math",
        type=int,
        default=0,
        help="GSM8K train examples: first N rows, or 0 (default) for the full train split (~7.5k).",
    )
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--per-device-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--g-train", type=int, default=4, help="GRPO num_generations")
    p.add_argument(
        "--generation-batch-size",
        type=int,
        default=32,
        help="Must be divisible by num_generations where required by TRL.",
    )
    p.add_argument("--grpo-beta", type=float, default=0.04)
    p.add_argument("--grpo-epsilon", type=float, default=0.2)
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="Max new tokens per GRPO rollout (vLLM/HF).",
    )
    p.add_argument(
        "--hf",
        action="store_true",
        help="Disable vLLM; use HF generate (same as main_pipeline GENERATION_BACKEND=HF).",
    )
    p.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="TRL vLLM integration mode (default: colocate on training GPU).",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.45,
        help=(
            "vLLM GPU memory fraction in colocate mode. GRPO also runs a large PyTorch forward for "
            "per-token log-probs on the same GPU; 0.85–0.95 almost always OOM. Use ~0.35–0.55 here, "
            "or add --vllm-enable-sleep-mode and you can try slightly higher."
        ),
    )
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--vllm-max-model-length",
        type=int,
        default=None,
        help="Optional vLLM context cap (tokens).",
    )
    p.add_argument(
        "--vllm-enable-sleep-mode",
        action="store_true",
        help=(
            "Colocate only: let TRL idle vLLM during train/logprob phases so more VRAM is free "
            "for PyTorch (avoids OOM when generation batches are large)."
        ),
    )
    p.add_argument(
        "--vllm-server-base-url",
        default=None,
        help="When --vllm-mode=server, e.g. http://127.0.0.1:8000",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip GSM8K test eval (no baseline before train, no eval after train).",
    )
    p.add_argument(
        "--eval-examples",
        type=int,
        default=32,
        help="GSM8K test rows (from the start) for baseline + post-train metrics.",
    )
    p.add_argument(
        "--inspect-examples",
        type=int,
        default=4,
        help="Print full question/response for this many eval rows (subset of eval).",
    )
    p.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for greedy baseline/post-train generation.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    use_vllm = not args.hf
    if use_vllm:
        if sys.platform != "linux":
            raise SystemExit("TRL+vLLM training is only supported on Linux.")
        if not torch.cuda.is_available():
            raise SystemExit("vLLM training requires CUDA.")
        try:
            importlib.import_module("vllm")
        except ImportError as exc:
            raise SystemExit(
                "vLLM is not installed. Install with: uv sync --extra vllm"
            ) from exc

    device = detect_device()
    if use_vllm and device.type != "cuda":
        raise SystemExit("vLLM requires a CUDA device.")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device} | use_vllm={use_vllm} | vllm_mode={args.vllm_mode if use_vllm else 'n/a'}")
    print(f"Output: {out}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    save_base_checkpoint(model, tokenizer, out)

    if not args.skip_eval:
        run_gsm8k_eval(
            model,
            tokenizer,
            device,
            args,
            phase="baseline",
            summary_filename="eval_baseline.json",
        )

    print("\nLoading GSM8K train slice...")
    train_split = (
        "train" if args.n_math <= 0 else f"train[:{args.n_math}]"
    )
    raw = load_dataset("openai/gsm8k", "main", split=train_split)
    train_dataset = raw.map(format_math, with_indices=True)
    print(f"  Train rows: {len(train_dataset)}")

    grpo_kw: dict = {
        "output_dir": str(out),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": None,
        "bf16": device.type == "cuda",
        "use_vllm": use_vllm,
        "num_generations": args.g_train,
        "generation_batch_size": args.generation_batch_size,
        "loss_type": "grpo",
        "beta": args.grpo_beta,
        "epsilon": args.grpo_epsilon,
        "importance_sampling_level": "token",
        "scale_rewards": "group",
        "max_completion_length": args.max_completion_length,
    }

    if use_vllm:
        grpo_kw["vllm_mode"] = args.vllm_mode
        grpo_kw["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        grpo_kw["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
        grpo_kw["vllm_enable_sleep_mode"] = args.vllm_enable_sleep_mode
        if args.vllm_max_model_length is not None:
            grpo_kw["vllm_max_model_length"] = args.vllm_max_model_length
        if args.vllm_mode == "server" and args.vllm_server_base_url:
            grpo_kw["vllm_server_base_url"] = args.vllm_server_base_url

    training_args = GRPOConfig(**grpo_kw)
    reward_funcs = [accuracy_reward_func]

    trainer = HistoricalBatchGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        history_output_dir=out,
    )

    print("\n" + "=" * 80)
    print("GRPO training")
    print("=" * 80)
    t0 = time.time()
    trainer.train()
    print(f"\nTraining finished in {time.time() - t0:.1f}s")
    clear_cache(device)

    if not args.skip_eval:
        run_gsm8k_eval(
            model,
            tokenizer,
            device,
            args,
            phase="after_train",
            summary_filename="eval_after_train.json",
        )


if __name__ == "__main__":
    main()
