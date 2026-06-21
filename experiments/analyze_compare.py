"""What would gold-IF select for a given target, at one checkpoint? (no training run)

Gold influence is the NLL-on-gold gradient — NO rollouts, so NO generation / verifier /
gen server. Load a checkpoint, score the pool against a <domain> target, print the domain
composition of the top-keep_fraction. Run it for math / physics / finance against the SAME
checkpoint to see how IF's selection shifts with target domain (model held fixed):

  isolated-knowledge target (finance) -> IF concentrates on that domain;
  cross-cutting target (math)         -> IF stays ~uniform (math helps all domains).

Single GPU, no servers (~10-25 min for a ~2k pool). Pass the same data args so the pool
matches across targets (only --webinstruct-test-domains changes).

  python -m experiments.analyze_compare --checkpoint-step 10 --run-name math_if_v2 \
      --model-id Qwen/Qwen3-1.7B-Base --lora-r 32 \
      --domains math,physics,finance --n-train-pool 2000 \
      --test-from-train --test-from-train-eval 1000 --webinstruct-test-domains physics \
      --n-if-target 256 --if-method tracin-adam --seed 42
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from influence_rlvr import detect_device

from .config import ExperimentConfig
from .data import load_if_target_set, load_train_pool
from .evaluate import _load_tokenizer
from .influence import compute_pool_influence


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    probe, rest = ap.parse_known_args(argv)
    cfg = ExperimentConfig.from_cli(rest)
    cfg.if_grad = "gold"          # this probe is gold-only (no rollouts/servers)
    cfg.use_vllm = False
    device = detect_device()
    tokenizer = _load_tokenizer(cfg)

    ckpt = cfg.grpo_output_dir / f"checkpoint-{probe.checkpoint_step}"
    if not ckpt.is_dir():
        raise SystemExit(f"[select] no checkpoint at {ckpt}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype).to(device)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=True).to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    pool = load_train_pool(cfg)
    target = load_if_target_set(cfg)
    tdoms = ",".join(cfg.webinstruct_test_domains) or "all"
    print(f"[select] checkpoint-{probe.checkpoint_step} | target={tdoms}({len(target)}) "
          f"pool={len(pool)} | method={cfg.if_method}/gold")

    save_dir = cfg.run_dir / "influence" / f"select_{tdoms}_step{probe.checkpoint_step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(
        compute_pool_influence(cfg, model, tokenizer, pool, target, device,
                               checkpoint_step=probe.checkpoint_step, save_dir=save_dir),
        dtype=np.float64)
    np.save(save_dir / "gold_scores.npy", scores)

    if "category" not in pool.column_names:
        raise SystemExit("pool has no 'category' column")
    cats = np.array(pool["category"], dtype=object)
    base_c = Counter(cats.tolist())
    n = len(cats)
    print(f"\npool baseline (n={n}): " +
          "  ".join(f"{d}={base_c[d]} ({base_c[d] / n:.0%})" for d in sorted(base_c)))
    print("(cell: count (share, enrichment vs pool); enrichment 1.0 = random)")
    print(f"=== gold-IF selection for {tdoms} target ===")
    for frac in (0.1, 0.2, 0.3):
        k = max(1, int(round(frac * n)))
        c = Counter(cats[np.argsort(-scores)[:k]].tolist())
        parts = "   ".join(
            f"{d}={c.get(d, 0):>3} ({c.get(d, 0) / k:4.0%}, {(c.get(d, 0) / k) / (base_c[d] / n):.2f}x)"
            for d in sorted(base_c))
        print(f"  top-{frac:.0%} (k={k:>3}): {parts}")


if __name__ == "__main__":
    main()
