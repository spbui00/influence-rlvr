#!/usr/bin/env python3
"""Compute surrogate IF on a trained RLVR run (Tier 0/1 of the LLM port).

Loads a run produced by `HistoricalBatchGRPOTrainer`, picks `θˢ` (latest
checkpoint) and `θ_ref` (checkpoint-0), and runs the standalone surrogate IF
from `influence_rlvr.surrogate.compute_surrogate_if_scores`.

For benchmark-style train sets that include `related_test_id` / `expected_influence`,
prints a per-test top-K helpful summary with markers when the IF picks examples
the benchmark expects (same `base_id` or matching `related_test_id`).

Designed to run on Apple Silicon (MPS) by default; falls back to CUDA or CPU.

Example:
    python scripts/compute_surrogate_if_llm.py \\
        --run-dir outputs/if_benchmark_small_run/rlvr-output \\
        --n-train 20 --n-test 5 --G 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from influence_rlvr.rewards import accuracy_reward_func
from influence_rlvr.surrogate import (
    compute_historical_last_if_scores,
    compute_surrogate_if_scores,
)
from influence_rlvr.utils import detect_device


def find_final_checkpoint(run_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not (p.is_dir() and p.name.startswith("checkpoint-")):
            continue
        try:
            step = int(p.name.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        if step > 0:
            candidates.append((step, p))
    if not candidates:
        raise FileNotFoundError(
            f"No non-zero checkpoint-* directories found under {run_dir}. "
            "Did this run complete training?"
        )
    candidates.sort()
    return candidates[-1][1]


def find_ref_checkpoint(run_dir: Path) -> Path | None:
    """Return checkpoint-0 if present, else None (caller should fall back to base model)."""
    p = run_dir / "checkpoint-0"
    if p.is_dir():
        return p
    return None


def load_run_config(run_dir: Path) -> tuple[dict, Path | None]:
    """Locate the run's config. Tries run_dir/run_config.json first, then any
    sibling results*/metadata.json (analysis-time config). Returns (config, source_path).
    Returns ({}, None) if nothing found — caller must supply --model-id and --beta."""
    direct = run_dir / "run_config.json"
    if direct.is_file():
        return json.loads(direct.read_text()), direct
    # Analysis-time fallback: sibling results*/metadata.json captures the same info.
    parent = run_dir.parent
    for candidate in sorted(parent.glob("results*")):
        meta = candidate / "metadata.json"
        if meta.is_file():
            return json.loads(meta.read_text()), meta
    return {}, None


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_math_reward_fn(sample: dict, num_generations: int):
    solution = sample.get("solution", "")
    return [partial(accuracy_reward_func, solution=[solution] * num_generations)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Path to rlvr-output directory (has checkpoints + run_config.json).")
    parser.add_argument("--train-jsonl", type=Path, default=None,
                        help="Train jsonl. Defaults to benchmark_train_jsonl from run_config.")
    parser.add_argument("--test-jsonl", type=Path, default=None,
                        help="Test jsonl. Defaults to a sibling test.jsonl next to the train file.")
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--G", type=int, default=4, help="Rollouts per prompt.")
    parser.add_argument("--beta", type=float, default=None,
                        help="KL temperature β for the surrogate. Defaults to run's grpo_beta.")
    parser.add_argument("--lambda-damp", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save scores.npy + config.json (default: <run-dir>/surrogate_if_<ts>).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override: mps | cuda | cpu (auto-detected if omitted).")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many top-helpful training prompts to print per test prompt.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16"],
                        help="Model dtype. fp16 is the right choice for MPS — halves memory and is supported. "
                             "fp32 only if you hit numerical issues.")
    parser.add_argument("--model-id", type=str, default=None,
                        help="HF model id (override). Auto-resolved from run_config.json / results*/metadata.json if not given.")
    parser.add_argument("--ref-checkpoint", type=Path, default=None,
                        help="Path to the reference adapter directory. If omitted, uses checkpoint-0 if present, "
                             "otherwise falls back to the untrained base model as π_ref.")
    parser.add_argument(
        "--gradient-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory to cache per-train-prompt surrogate gradient bundles. Cache key includes "
            "prompt content, checkpoints, G, β, temperature, top_p, max_new_tokens, seed, dtype — "
            "anything that affects the gradient invalidates the cache. "
            "Default: <run-dir>/surrogate_grad_cache/."
        ),
    )
    parser.add_argument("--no-gradient-cache", action="store_true",
                        help="Disable gradient caching entirely (forces fresh compute every run).")
    parser.add_argument(
        "--if-method",
        choices=["surrogate", "historical-last"],
        default="surrogate",
        help=(
            "Which influence-function method to use. "
            "`surrogate`: importance-weighted ∇log π gradient + (1+K)-block per-prompt Fisher "
            "(the formula derived in our work; checkpoint-free). "
            "`historical-last`: GRPO loss gradient + policy-score Fisher, evaluated only at θˢ "
            "(the toy's `historical-last` baseline). Different gradient formula at the same checkpoint."
        ),
    )
    parser.add_argument(
        "--historical-beta",
        type=float,
        default=0.0,
        help=(
            "(--if-method historical-last) β for the GRPO loss gradient. Default 0 matches the toy "
            "convention (no KL term in the gradient even if training used β>0). Set higher to "
            "match training dynamics more precisely."
        ),
    )
    parser.add_argument(
        "--test-objective",
        choices=["sft_solution", "expected_reward_pg"],
        default="sft_solution",
        help=(
            "How the test gradient g_test is computed. "
            "`sft_solution` (default): cross-entropy loss on the gold `solution` string — "
            "always nonzero, doesn't need the model to actually solve the test problem. "
            "`expected_reward_pg`: gradient of negative expected reward over G sampled rollouts "
            "(legacy RL-style path). Zero whenever no rollout achieves reward; useful only "
            "when the model can actually solve the test problems."
        ),
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_config, config_source = load_run_config(run_dir)
    if config_source is not None:
        print(f"Run config loaded from: {config_source}")
    else:
        print("No run_config.json or results*/metadata.json found — relying on CLI overrides.")

    model_id = args.model_id or run_config.get("model_id")
    if not model_id:
        raise ValueError(
            "Cannot determine model_id. Pass --model-id explicitly, or ensure run_dir or a sibling "
            "results*/metadata.json contains it."
        )

    if args.beta is None:
        beta_from_config = run_config.get("grpo_beta")
        if beta_from_config is None:
            args.beta = 0.1
            print(f"β not provided and no grpo_beta in config → defaulting to {args.beta}.")
        else:
            args.beta = float(beta_from_config)
            print(f"β not provided → using grpo_beta={args.beta} from config.")

    train_jsonl = args.train_jsonl
    if train_jsonl is None:
        bj = run_config.get("benchmark_train_jsonl")
        if not bj:
            raise ValueError("--train-jsonl not given and run_config has no benchmark_train_jsonl.")
        train_jsonl = Path(bj)

    test_jsonl = args.test_jsonl
    if test_jsonl is None:
        candidate = train_jsonl.parent / "test.jsonl"
        if candidate.is_file():
            test_jsonl = candidate
            print(f"--test-jsonl not given → using sibling {test_jsonl}.")
        else:
            raise ValueError(
                "--test-jsonl not given and no sibling test.jsonl found next to the train jsonl."
            )

    print(f"\nRun dir : {run_dir}")
    print(f"Model id: {model_id}")
    print(f"Train   : {train_jsonl}")
    print(f"Test    : {test_jsonl}")

    train_items_all = load_jsonl(train_jsonl)
    test_items_all = load_jsonl(test_jsonl)
    train_items = train_items_all[: args.n_train]
    test_items = test_items_all[: args.n_test]
    print(f"  loaded {len(train_items)} train (of {len(train_items_all)}) "
          f"and {len(test_items)} test (of {len(test_items_all)}) samples.")

    train_dataset = Dataset.from_list(train_items)
    test_dataset = Dataset.from_list(test_items)

    device = torch.device(args.device) if args.device else detect_device()
    print(f"Device  : {device}")

    final_ckpt = find_final_checkpoint(run_dir)
    ref_ckpt = args.ref_checkpoint.resolve() if args.ref_checkpoint else find_ref_checkpoint(run_dir)
    print(f"θˢ      : {final_ckpt}")
    if ref_ckpt is None:
        print(f"θ_ref   : <untrained base model> (no checkpoint-0 found)")
    else:
        print(f"θ_ref   : {ref_ckpt}")

    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    print(f"\nLoading base model + final adapter ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_a = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    # is_trainable=True so the LoRA params have requires_grad — needed for autograd-based IF.
    peft_model = PeftModel.from_pretrained(
        base_a, str(final_ckpt), adapter_name="default", is_trainable=True,
    )
    peft_model.to(device)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"  trainable params (LoRA): {n_trainable:,}")
    if n_trainable == 0:
        raise RuntimeError(
            "Loaded PEFT model has no trainable parameters. "
            "Check the adapter at the final checkpoint."
        )

    if ref_ckpt is None:
        # No reference adapter — use the untrained base model as π_ref.
        print(f"Loading second base model (untrained, no adapter) as π_ref...")
        ref_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        ref_model.to(device)
    else:
        print(f"Loading second base model + reference adapter...")
        base_b = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        ref_model = PeftModel.from_pretrained(base_b, str(ref_ckpt), adapter_name="default")
        ref_model.to(device)

    # Resolve gradient cache directory.
    if args.no_gradient_cache:
        gradient_cache_dir = None
        print("Gradient cache: DISABLED (--no-gradient-cache).")
    else:
        gradient_cache_dir = args.gradient_cache_dir or (run_dir / "surrogate_grad_cache")
        print(f"Gradient cache: {gradient_cache_dir}")

    print(
        f"\nComputing {args.if_method} IF "
        f"(G={args.G} β={args.beta} λ={args.lambda_damp} test_objective={args.test_objective})..."
    )
    t0 = time.time()
    common_kwargs = dict(
        peft_model=peft_model, ref_model=ref_model, tokenizer=tokenizer,
        train_dataset=train_dataset, test_dataset=test_dataset,
        reward_fn_builder=build_math_reward_fn,
        device=device, G=args.G, lambda_damp=args.lambda_damp,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_p=args.top_p,
        seed_base=args.seed,
        train_limit=args.n_train, test_limit=args.n_test,
        progress=True,
        test_objective=args.test_objective,
        gradient_cache_dir=gradient_cache_dir,
        final_checkpoint_id=str(final_ckpt),
        ref_checkpoint_id=None if ref_ckpt is None else str(ref_ckpt),
        dtype_id=args.dtype,
    )
    if args.if_method == "surrogate":
        result = compute_surrogate_if_scores(beta=args.beta, **common_kwargs)
    elif args.if_method == "historical-last":
        result = compute_historical_last_if_scores(beta=args.historical_beta, **common_kwargs)
    else:
        raise ValueError(f"Unknown --if-method={args.if_method}")
    elapsed = time.time() - t0
    scores = result["scores"]  # shape (n_test, n_train)
    print(f"\nIF computed in {elapsed:.1f}s. Scores matrix: shape={scores.shape}")
    print(f"  mean={scores.mean():.4e}  std={scores.std():.4e}  "
          f"min={scores.min():.4e}  max={scores.max():.4e}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (run_dir / f"surrogate_if_{ts}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "scores.npy", scores)
    saved_config: dict[str, Any] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    saved_config["model_id"] = model_id
    saved_config["final_checkpoint"] = str(final_ckpt)
    saved_config["ref_checkpoint"] = str(ref_ckpt)
    saved_config["elapsed_seconds"] = elapsed
    saved_config["train_ids"] = [it.get("id") for it in train_items]
    saved_config["test_ids"] = [it.get("id") for it in test_items]
    with (output_dir / "config.json").open("w") as f:
        json.dump(saved_config, f, indent=2)
    print(f"Saved scores → {output_dir / 'scores.npy'}")
    print(f"Saved config → {output_dir / 'config.json'}")

    # Qualitative top-K per test (Tier 1).
    print("\n" + "=" * 70)
    print(f"Top-{args.top_k} helpful training prompts per test (qualitative)")
    print("=" * 70)
    has_benchmark_meta = any("related_test_id" in it or "base_id" in it for it in train_items)
    for ti, test_item in enumerate(test_items):
        test_id = test_item.get("id", f"test_{ti}")
        test_base = test_item.get("base_id", "")
        problem_snippet = (test_item.get("problem") or "")[:140].replace("\n", " ")
        print(f"\n[Test {ti}] id={test_id} base_id={test_base}")
        if problem_snippet:
            print(f"  problem: {problem_snippet!r}")
        order = np.argsort(-scores[ti])
        for k in range(min(args.top_k, len(order))):
            j = int(order[k])
            it = train_items[j]
            train_id = it.get("id", f"train_{j}")
            related = it.get("related_test_id", "")
            variant = it.get("variant_type", "")
            expected = it.get("expected_influence", "")
            marker = ""
            if has_benchmark_meta:
                if related and related == test_id:
                    marker = "  ← BENCHMARK MATCH (related_test_id)"
                elif test_base and it.get("base_id") == test_base:
                    marker = "  ← BENCHMARK MATCH (same base_id)"
            print(
                f"  #{k+1} score={scores[ti, j]:+.4f}  {train_id}"
                + (f"  variant={variant}" if variant else "")
                + (f"  expected={expected}" if expected else "")
                + marker
            )

    if has_benchmark_meta:
        print("\n" + "=" * 70)
        print("Tier-1 sanity: BENCHMARK MATCH rate in top-K")
        print("=" * 70)
        n_match = 0
        n_total = 0
        for ti, test_item in enumerate(test_items):
            test_id = test_item.get("id", "")
            test_base = test_item.get("base_id", "")
            order = np.argsort(-scores[ti])
            for k in range(min(args.top_k, len(order))):
                j = int(order[k])
                it = train_items[j]
                n_total += 1
                if (it.get("related_test_id") == test_id) or (test_base and it.get("base_id") == test_base):
                    n_match += 1
        rate = n_match / max(n_total, 1)
        print(f"  {n_match}/{n_total} top-{args.top_k} positions are benchmark matches  "
              f"({rate * 100:.1f}%).")
        print("  Random baseline would be roughly n_match_expected ≈ "
              f"{args.top_k * args.n_test * (sum(1 for it in train_items if it.get('related_test_id')) / max(len(train_items), 1)):.1f} "
              f"out of {args.top_k * args.n_test}.")

    print("\nDone.")


if __name__ == "__main__":
    main()
