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


def _load_tokenizer(cfg: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy(cfg: ExperimentConfig, checkpoint_dir: Path, device):
    tokenizer = _load_tokenizer(cfg)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir)).to(device)
    model.eval()
    return model, tokenizer


def load_base_policy(cfg: ExperimentConfig, device):
    """The base model with NO LoRA adapter — for base-model headroom checks."""
    tokenizer = _load_tokenizer(cfg)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def _vllm_generate_answers(tokenizer, questions: list[str], cfg: ExperimentConfig, device,
                           max_new_tokens: int, adapter_path: str | None) -> list[str]:
    """Batched greedy generation via vLLM (one engine, standalone process — safe).

    Reuses the SAME tested vLLM path as training (`generate_rollout_batch`), loading
    the base model + the checkpoint's LoRA adapter. vLLM's continuous batching does
    all prompts at once → ~5× over HF. The HF policy model is NOT loaded in this mode.
    """
    from influence_rlvr.generation import generate_rollout_batch
    from influence_rlvr.modes import GenerationBackend, VLLMConfig

    chats = [build_reasoning_prompt(q) for q in questions]
    prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
               for c in chats]
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=cfg.max_prompt_length).to(device)
    do_sample = bool(cfg.eval_temperature and cfg.eval_temperature > 0)
    vcfg = VLLMConfig(gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                      max_model_len=cfg.vllm_max_model_len, max_lora_rank=cfg.lora_r)
    rollout = generate_rollout_batch(
        None, tokenizer, enc["input_ids"], enc["attention_mask"],
        backend=GenerationBackend.VLLM, num_samples=1, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=(cfg.eval_temperature or 0.0), top_p=cfg.eval_top_p,
        vllm_config=vcfg, adapter_path=adapter_path, model_id=cfg.model_id,
    )
    return list(rollout.texts)


@torch.inference_mode()
def generate_answers(model, tokenizer, questions: list[str], cfg: ExperimentConfig,
                     device, batch_size: int = 8, max_new_tokens: int | None = None,
                     *, vllm: bool = False, vllm_adapter: str | None = None) -> list[str]:
    max_new_tokens = max_new_tokens or cfg.eval_max_new_tokens
    if vllm:  # vllm_adapter=None → base model (no adapter); a path → base+adapter
        return _vllm_generate_answers(tokenizer, questions, cfg, device, max_new_tokens, vllm_adapter)
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
            max_new_tokens=max_new_tokens,
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


def score_examples(examples: list[dict], model, tokenizer, cfg: ExperimentConfig,
                   device, *, max_new_tokens: int | None = None,
                   vllm: bool = False, vllm_adapter: str | None = None) -> dict:
    """Generate + verifier-score a list of {question, solution, category} rows.

    Shared by the post-hoc benchmark eval and the in-training LiveEvalCallback.
    `vllm_adapter` (a checkpoint path) routes generation through vLLM — only valid
    for the standalone eval, NOT the in-training callback (that shares the trainer's
    process/engine).
    """
    questions = [e["question"] for e in examples]
    golds = [e["solution"] for e in examples]
    responses = generate_answers(model, tokenizer, questions, cfg, device,
                                 max_new_tokens=max_new_tokens, vllm=vllm,
                                 vllm_adapter=vllm_adapter)
    students = [_student_answer(r) for r in responses]
    rewards = get_verifier_from_config(cfg).verify_batch(questions, golds, students)

    by_cat: dict[str, list[float]] = defaultdict(list)
    for ex, r in zip(examples, rewards):
        by_cat[ex.get("category", "")].append(r)
    return {
        "n": len(rewards),
        "accuracy": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "per_category": {c: sum(v) / len(v) for c, v in by_cat.items() if v},
        "students": students,
        "rewards": rewards,
    }


def evaluate_benchmark(name: str, cfg: ExperimentConfig, model, tokenizer, device,
                       vllm: bool = False, vllm_adapter: str | None = None) -> dict:
    examples = load_eval_benchmark(name, cfg, cfg.eval_max_examples)
    print(f"  [{name}] generating {len(examples)} completions"
          f"{' (vLLM)' if vllm else ''}...")
    res = score_examples(examples, model, tokenizer, cfg, device,
                         vllm=vllm, vllm_adapter=vllm_adapter)
    print(f"  [{name}] accuracy = {res['accuracy']:.4f} (n={res['n']})")
    return {
        "benchmark": name,
        "n": res["n"],
        "accuracy": res["accuracy"],
        "per_category": res["per_category"],
        "per_example": [
            {"category": ex.get("category", ""), "gold": ex["solution"],
             "student": s, "reward": r}
            for ex, s, r in zip(examples, res["students"], res["rewards"])
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
    p.add_argument("--vllm", action="store_true",
                   help="Generate with vLLM (~5× faster). Standalone eval only — "
                        "loads base+adapter in a vLLM engine; skips the HF policy.")
    p.add_argument("--base", action="store_true",
                   help="Eval the BASE model (no LoRA adapter) — for headroom checks.")
    p.add_argument("--model-id", default=None,
                   help="Override cfg.model_id (e.g. Qwen/Qwen3-1.7B) — pair with --base "
                        "to headroom-check a different base without a trained run.")
    args = p.parse_args(argv)

    run_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    cfg = ExperimentConfig.load(run_dir / "config.json")
    if args.model_id:
        cfg.model_id = args.model_id
    if args.eval_max_examples is not None:
        cfg.eval_max_examples = args.eval_max_examples
    benchmarks = (
        tuple(b.strip() for b in args.benchmarks.split(",") if b.strip())
        if args.benchmarks else cfg.eval_benchmarks
    )

    device = detect_device()
    if args.base:
        step = "base"
        ckpt = None
        print(f"Evaluating {cfg.model_id} BASE (no adapter) on {list(benchmarks)}"
              f"{' [vLLM]' if args.vllm else ''}")
    else:
        ckpt = _resolve_checkpoint(cfg, args.checkpoint_step)
        step = checkpoint_step(str(ckpt))
        print(f"Evaluating {cfg.run_name} @ checkpoint-{step} on {list(benchmarks)}"
              f"{' [vLLM]' if args.vllm else ''}")

    if args.vllm:
        # vLLM loads base(+adapter) itself; skip the HF policy (saves GPU room for the
        # engine + verifier). Only the tokenizer is loaded here.
        tokenizer = _load_tokenizer(cfg)
        model = None
        vllm_adapter = None if args.base else str(ckpt)
    else:
        model, tokenizer = (load_base_policy(cfg, device) if args.base
                            else load_policy(cfg, ckpt, device))
        vllm_adapter = None

    results = {}
    t0 = time.time()
    for name in benchmarks:
        results[name] = evaluate_benchmark(name, cfg, model, tokenizer, device,
                                           vllm=args.vllm, vllm_adapter=vllm_adapter)

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
