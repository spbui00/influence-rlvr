#!/usr/bin/env python3
import argparse
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
    format_reward_func,
)
from influence_rlvr.prompts import build_r1_math_prompt, extract_gsm8k_target


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
        "prompt": build_r1_math_prompt(example["question"]),
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
    p.add_argument("--model-id", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/train_script/rlvr-output"),
        help="Checkpoints and batch history are written here.",
    )
    p.add_argument("--n-math", type=int, default=128, help="GSM8K train rows [:N]")
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
    p.add_argument("--max-completion-length", type=int, default=256)
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
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
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
        help="Let TRL put colocated vLLM to sleep during optimizer steps (less VRAM, some latency).",
    )
    p.add_argument(
        "--vllm-server-base-url",
        default=None,
        help="When --vllm-mode=server, e.g. http://127.0.0.1:8000",
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
            import vllm  # noqa: F401
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

    print("\nLoading GSM8K train slice...")
    raw = load_dataset("openai/gsm8k", "main", split=f"train[:{args.n_math}]")
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

    trainer = HistoricalBatchGRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
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


if __name__ == "__main__":
    main()
