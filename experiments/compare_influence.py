"""Gold vs rollout influence — do they rank/select the same pool prompts?

Cheap pre-screen for "is on-policy (rollout) influence worth its ~10-30x cost?":
score ONE checkpoint's pool with both `if_grad=gold` and `if_grad=rollout` against the
SAME IF target, then report Spearman(scores) + top-keep_fraction selection overlap.

  high overlap (>~0.7)  -> gold ~= rollout: skip the rollout experiment, gold is fine.
  low overlap (<~0.4)   -> they pick different prompts: rollout is a genuinely different
                           selector and the (cheap, reduced-pool) rollout run is justified.

NOTE: this is rank AGREEMENT (do they differ), not LDS (which is "better") — LDS needs
the retraining sweep, not a single checkpoint. Agreement is the right first question:
if they agree, rollout provably cannot beat gold (same selection -> same training).

Single process. The gold pass needs no generation; the rollout pass generates G rollouts
per prompt (HF in-process) and grades them via the verifier server, so launch one.

  python -m experiments.compare_influence --checkpoint-step 10 \
      --run-name math_if_v2 --model-id Qwen/Qwen3-1.7B-Base \
      --domains math,physics,finance --n-train-pool 4000 \
      --test-from-train --webinstruct-test-domains math --n-if-target 256 \
      --if-method tracin-adam --if-g-train 4 --if-score-batch 64 \
      --verifier-backend vllm --verifier-server-port 8100
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


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks), no scipy dependency."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    probe, rest = ap.parse_known_args(argv)

    cfg = ExperimentConfig.from_cli(rest)
    device = detect_device()
    tokenizer = _load_tokenizer(cfg)

    ckpt = cfg.grpo_output_dir / f"checkpoint-{probe.checkpoint_step}"
    if not ckpt.is_dir():
        raise SystemExit(f"[compare] no checkpoint at {ckpt}")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype).to(device)
    base.config.use_cache = False
    # Load the trained adapter at this step (is_trainable=True so grads flow for scoring).
    model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=True).to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target = load_if_target_set(cfg)
    pool = load_train_pool(cfg)
    print(f"[compare] checkpoint-{probe.checkpoint_step} | pool={len(pool)} "
          f"target={len(target)} | method={cfg.if_method} G={cfg.if_g_train}")

    save_dir = cfg.run_dir / "influence" / f"compare_step{probe.checkpoint_step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n[compare] scoring with if_grad=GOLD (no rollouts)...")
    cfg.if_grad = "gold"
    gold = compute_pool_influence(cfg, model, tokenizer, pool, target, device,
                                  checkpoint_step=probe.checkpoint_step, save_dir=save_dir)
    print("\n[compare] scoring with if_grad=ROLLOUT (on-policy)...")
    cfg.if_grad = "rollout"
    roll = compute_pool_influence(cfg, model, tokenizer, pool, target, device,
                                  checkpoint_step=probe.checkpoint_step, save_dir=save_dir)

    gold = np.asarray(gold, dtype=np.float64)
    roll = np.asarray(roll, dtype=np.float64)
    np.save(save_dir / "gold_scores.npy", gold)
    np.save(save_dir / "rollout_scores.npy", roll)

    n = len(gold)
    print("\n" + "=" * 60)
    print(f"[compare] Spearman(gold, rollout) = {_spearman(gold, roll):+.3f}   (n={n})")
    print("[compare] selection overlap (which prompts each would KEEP):")
    for frac in (0.1, 0.2, 0.3):
        k = max(1, int(round(frac * n)))
        tg = set(np.argsort(-gold)[:k].tolist())
        tr = set(np.argsort(-roll)[:k].tolist())
        inter = len(tg & tr)
        jac = inter / len(tg | tr)
        print(f"    top-{frac:.0%} ({k:>4}):  overlap {inter / k:5.0%}   jaccard {jac:.2f}")
    print("=" * 60)
    print("[compare] read: overlap >~0.7 -> gold ~= rollout, SKIP the rollout run; "
          "<~0.4 -> they differ, rollout may matter.")

    # Domain composition of each method's selection — does gold/rollout concentrate on
    # the target domain? (cell: count (share, enrichment vs pool); enrichment 1.0 = random)
    if "category" in pool.column_names:
        cats = np.array(pool["category"], dtype=object)
        base_c = Counter(cats.tolist())
        tdoms = ",".join(cfg.webinstruct_test_domains) or "all"
        print(f"\n[compare] selection by DOMAIN (target={tdoms}, baseline " +
              "  ".join(f"{d} {base_c[d] / n:.0%}" for d in sorted(base_c)) + "):")
        for label, sc in (("GOLD", gold), ("ROLLOUT", roll)):
            print(f"  {label}:")
            for frac in (0.1, 0.2, 0.3):
                k = max(1, int(round(frac * n)))
                c = Counter(cats[np.argsort(-sc)[:k]].tolist())
                parts = "   ".join(
                    f"{d}={c.get(d, 0):>3} ({c.get(d, 0) / k:4.0%}, "
                    f"{(c.get(d, 0) / k) / (base_c[d] / n):.2f}x)" for d in sorted(base_c))
                print(f"    top-{frac:.0%} (k={k:>4}): {parts}")


if __name__ == "__main__":
    main()
