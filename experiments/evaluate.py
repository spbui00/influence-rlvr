"""Evaluate a trained checkpoint on the benchmark suites.

Loads the LoRA adapter from a checkpoint, generates a completion per benchmark
question, extracts the final answer, and scores it with the *same* general-verifier
used for training (free-form equivalence vs the reference answer). Reports
per-benchmark and per-domain accuracy.

Run:
    python -m experiments.evaluate --run-name qwen3_4b_base --checkpoint-step latest
    python -m experiments.evaluate --run-name qwen3_4b_ifprune --checkpoint-step 400 \
        --benchmarks webinstruct_test,gsm8k,math500
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_rlvr import detect_device
from influence_rlvr.checkpoint_schedule import checkpoint_step, list_checkpoint_dirs

from .config import ExperimentConfig
from .data import build_reasoning_prompt, load_eval_benchmark
from .verifier import _student_answer, get_verifier_from_config


def _resolve_checkpoint(cfg: ExperimentConfig, step_spec: str) -> Path:
    out = cfg.grpo_output_dir
    dirs = list_checkpoint_dirs(str(out))
    if not dirs:
        raise FileNotFoundError(f"No checkpoints under {out}.")
    if step_spec == "latest":
        return Path(dirs[-1])
    target = int(step_spec)
    for d in dirs:
        if checkpoint_step(d) == target:
            return Path(d)
    raise FileNotFoundError(f"No checkpoint-{target} under {out}.")


def load_policy(cfg: ExperimentConfig, checkpoint_dir: Path, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir)).to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_answers(model, tokenizer, questions: list[str], cfg: ExperimentConfig,
                     device, batch_size: int = 8) -> list[str]:
    responses: list[str] = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        chats = [build_reasoning_prompt(q) for q in batch]
        prompts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=cfg.max_prompt_length).to(device)
        gen_kwargs = dict(
            max_new_tokens=cfg.eval_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if cfg.eval_temperature and cfg.eval_temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=cfg.eval_temperature,
                              top_p=cfg.eval_top_p)
        else:
            gen_kwargs.update(do_sample=False)
        out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        responses.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return responses


def evaluate_benchmark(name: str, cfg: ExperimentConfig, model, tokenizer, device) -> dict:
    examples = load_eval_benchmark(name, cfg, cfg.eval_max_examples)
    questions = [e["question"] for e in examples]
    golds = [e["solution"] for e in examples]
    print(f"  [{name}] generating {len(examples)} completions...")
    responses = generate_answers(model, tokenizer, questions, cfg, device)
    students = [_student_answer(r) for r in responses]

    verifier = get_verifier_from_config(cfg)
    rewards = verifier.verify_batch(questions, golds, students)

    by_cat: dict[str, list[float]] = defaultdict(list)
    for ex, r in zip(examples, rewards):
        by_cat[ex.get("category", "")].append(r)

    acc = sum(rewards) / len(rewards) if rewards else 0.0
    per_category = {c: sum(v) / len(v) for c, v in by_cat.items() if v}
    print(f"  [{name}] accuracy = {acc:.4f} (n={len(rewards)})")
    return {
        "benchmark": name,
        "n": len(rewards),
        "accuracy": acc,
        "per_category": per_category,
        "per_example": [
            {"category": ex.get("category", ""), "gold": ex["solution"],
             "student": s, "reward": r}
            for ex, s, r in zip(examples, students, rewards)
        ],
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on benchmark suites.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-root", default="./outputs")
    p.add_argument("--checkpoint-step", default="latest")
    p.add_argument("--benchmarks", default=None,
                   help="Comma-separated; defaults to config.eval_benchmarks.")
    p.add_argument("--eval-max-examples", type=int, default=None)
    args = p.parse_args(argv)

    run_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    cfg = ExperimentConfig.load(run_dir / "config.json")
    if args.eval_max_examples is not None:
        cfg.eval_max_examples = args.eval_max_examples
    benchmarks = (
        tuple(b.strip() for b in args.benchmarks.split(",") if b.strip())
        if args.benchmarks else cfg.eval_benchmarks
    )

    device = detect_device()
    ckpt = _resolve_checkpoint(cfg, args.checkpoint_step)
    step = checkpoint_step(str(ckpt))
    print(f"Evaluating {cfg.run_name} @ checkpoint-{step} on {list(benchmarks)}")
    model, tokenizer = load_policy(cfg, ckpt, device)

    results = {}
    t0 = time.time()
    for name in benchmarks:
        results[name] = evaluate_benchmark(name, cfg, model, tokenizer, device)

    summary = {
        "run_name": cfg.run_name,
        "regime": cfg.regime,
        "checkpoint_step": step,
        "elapsed_s": time.time() - t0,
        "accuracy": {n: r["accuracy"] for n, r in results.items()},
        "results": results,
    }
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / f"eval_step{step}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    print("Accuracy by benchmark:")
    for n, a in summary["accuracy"].items():
        print(f"  {n:>18s}: {a:.4f}")


if __name__ == "__main__":
    main()
