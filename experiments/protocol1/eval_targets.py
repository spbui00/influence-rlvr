"""Measure target reward for Protocol-1 LDS: sampled pass rate on target.jsonl.

The LDS "actual" y for a subset model: for each target prompt, sample n rollouts
at the TRAINING temperature (1.0 / top_p 1.0 — same functional GRPO optimizes and
the pilot banded) and score the boxed answer with the honest match verifier.
Per-target pass rates enable TRAK-style per-target LDS; the mean is the pooled y.

Called in-process by train.py after each run; standalone CLI re-evals any
checkpoint (e.g. the pi_ref baseline):

  python -m experiments.protocol1.eval_targets \
      --checkpoint <run>/checkpoint-100 --out <run>/target_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.protocol2.reward import MATCH, single_reward
from influence_rlvr import detect_device
from influence_rlvr.generation import GenerationBackend, generate_rollout_batch
from influence_rlvr.rewards import extract_math_final_answer
from influence_rlvr.utils import tokenize_prompts_batch

DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def load_targets(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def evaluate_targets(model, tokenizer, targets: list[dict], device, *,
                     n_samples: int = 16, temperature: float = 1.0, top_p: float = 1.0,
                     max_new_tokens: int = 512, seed: int = 12345, batch_size: int = 4,
                     log=print) -> np.ndarray:
    """Per-target sampled pass rate [T]. HF generation (safe next to a live TRL
    vLLM colocate engine — never spin a second engine in-process). `batch_size`
    target prompts share one generate call (batch_size x n_samples sequences,
    left-padded, input-major order) — matters now that the horizon sweep evals
    many times per run."""
    import torch

    rates = np.zeros(len(targets), dtype=np.float64)
    B = max(1, batch_size)
    model.eval()
    with torch.no_grad():
        for c in range(0, len(targets), B):
            chunk = targets[c : c + B]
            _, ids, am = tokenize_prompts_batch(tokenizer, [r["prompt"] for r in chunk], device)
            rollout = generate_rollout_batch(
                model, tokenizer, ids, am,
                backend=GenerationBackend.HF, num_samples=n_samples,
                max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, seed=seed + c,
            )
            for j, row in enumerate(chunk):
                texts = rollout.texts[j * n_samples : (j + 1) * n_samples]
                rewards = [single_reward(extract_math_final_answer(txt), row["gold"], MATCH)
                           for txt in texts]
                rates[c + j] = float(np.mean(rewards))
            done = min(c + B, len(targets))
            if log:
                log(f"  [eval] {done}/{len(targets)} targets, "
                    f"running mean reward {rates[:done].mean():.3f}")
    return rates


def write_eval(out: Path, rates: np.ndarray, *, extra: dict | None = None) -> dict:
    payload = {
        "mean_reward": float(rates.mean()),
        "per_target": [round(float(r), 6) for r in rates],
        **(extra or {}),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Standalone target-reward eval of a checkpoint.")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="LoRA adapter dir (default: bare base model)")
    ap.add_argument("--targets", type=Path, default=DATA_DIR / "target.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--eval-temperature", type=float, default=1.0)
    ap.add_argument("--eval-top-p", type=float, default=1.0)
    ap.add_argument("--eval-max-new-tokens", type=int, default=512)
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--eval-batch", type=int, default=8, help="target prompts per generate call")
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from influence_rlvr import load_adapter_checkpoint

    device = detect_device()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    if args.checkpoint:
        load_adapter_checkpoint(model, str(args.checkpoint))
        print(f"[eval] adapter = {args.checkpoint}")
    else:
        print("[eval] bare base model (zero adapter)")

    targets = load_targets(args.targets)
    rates = evaluate_targets(
        model, tok, targets, device,
        n_samples=args.eval_samples, temperature=args.eval_temperature,
        top_p=args.eval_top_p, max_new_tokens=args.eval_max_new_tokens,
        seed=args.eval_seed, batch_size=args.eval_batch)
    payload = write_eval(args.out, rates, extra={
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "targets": str(args.targets), "n_targets": len(targets),
        "eval_samples": args.eval_samples, "eval_temperature": args.eval_temperature,
        "eval_top_p": args.eval_top_p, "eval_seed": args.eval_seed,
    })
    print(f"[eval] mean target reward = {payload['mean_reward']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
