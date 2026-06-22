"""Smoke-test the JVP (forward-mode) influence scorer vs the reverse-mode dot.

score(z) = <∇θ L(z), h>, with h fixed. The current scorer does forward+backward → the full
34.8M-D gradient → dot with h. Forward-mode (JVP) carries the tangent h through ONE forward
pass and returns the scalar <∇L, h> directly — no backward, no D-vector, batches naturally.
This verifies the two are numerically equal through the REAL Qwen3+LoRA gold-NLL loss before
we wire JVP into the production scorer (the speed win that makes frequent recompute feasible).

Reuses the production functional forward (_forward_per_token_logps_functional) so the loss is
identical to compute_sft_gradient_batch. Correctness only — tiny pool, fp32, eager attention.
PASS = Spearman ≈ 1.0 and tiny max-diff.

  python -m experiments.jvp_smoke --checkpoint-step 10 --run-name math_if_v2 \
      --model-id Qwen/Qwen3-1.7B-Base --lora-r 32 --domains math,physics,finance \
      --n-train-pool 60 --test-from-train --test-from-train-eval 1000 \
      --webinstruct-test-domains physics --n-if-target 8 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from peft import PeftModel
from torch.func import grad as func_grad
from torch.func import jvp as func_jvp
from transformers import AutoModelForCausalLM

from influence_rlvr.gradients import (
    _forward_per_token_logps_functional,
    _state_dict_for_functional,
)
from influence_rlvr.utils import tokenize_prompts_batch

from .config import ExperimentConfig
from .data import load_if_target_set, load_train_pool
from .evaluate import _load_tokenizer


def _spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--n-pool", type=int, default=16, help="pool examples to compare")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="bf16 = the PRODUCTION dtype (the real robustness check); fp32 = "
                         "the high-precision reference.")
    probe, rest = ap.parse_known_args(argv)
    cfg = ExperimentConfig.from_cli(rest)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = _load_tokenizer(cfg)

    ckpt = cfg.grpo_output_dir / f"checkpoint-{probe.checkpoint_step}"
    if not ckpt.is_dir():
        raise SystemExit(f"[jvp-smoke] no checkpoint at {ckpt}")
    dt = torch.float32 if probe.dtype == "fp32" else torch.bfloat16
    print(f"[jvp-smoke] dtype={probe.dtype} ckpt={ckpt.name}")
    # eager attention: forward-mode AD is less tested through fused (flash/sdpa) kernels.
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=dt, attn_implementation="eager").to(dev)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=True).to(dev)
    model.eval()

    names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
    ptuple = tuple(model.get_parameter(n).detach() for n in names)
    D = sum(p.numel() for p in ptuple)
    print(f"[jvp-smoke] {len(names)} trainable (LoRA) tensors, D={D}")
    eos = tok.eos_token_id

    def encode(sample):
        _, p_ids, p_am = tokenize_prompts_batch(tok, [sample["prompt"]], dev)
        gold = str(sample.get("solution", "") or "")
        c = tok(f"\\boxed{{{gold}}}", add_special_tokens=False,
                return_tensors="pt")["input_ids"].to(dev)
        if eos is not None:
            c = torch.cat([c, torch.full((1, 1), int(eos), device=dev, dtype=c.dtype)], dim=1)
        return p_ids[0], p_am[0], c, torch.ones_like(c)

    def loss_of(pt, p_ids, p_am, c, cm):
        sd = _state_dict_for_functional(model, pt, names)
        ptl = _forward_per_token_logps_functional(model, sd, p_ids, p_am, c, cm)
        return -(ptl * cm).sum() / cm.sum().clamp(min=1)

    grad_of = func_grad(loss_of, argnums=0)  # reverse-mode grad wrt params (the reference)

    pool = load_train_pool(cfg)
    target = load_if_target_set(cfg)
    n_pool = min(probe.n_pool, len(pool))
    print(f"[jvp-smoke] pool={n_pool} target={len(target)}")

    # h = mean target gold-gradient. No Adam precond here: P is an elementwise scaling of h,
    # orthogonal to the JVP-vs-dot correctness this test isolates.
    h = None
    for i in range(len(target)):
        g = grad_of(ptuple, *encode(target[i]))
        h = g if h is None else tuple(a + b for a, b in zip(h, g))
    h = tuple(t / len(target) for t in h)

    dot_s, jvp_s = [], []
    for i in range(n_pool):
        enc = encode(pool[i])
        g = grad_of(ptuple, *enc)
        # fp32-accumulated dot = the PRODUCTION reference (it upcasts grads before dotting),
        # so this compares bf16-JVP against the fp32 dot the real scorer uses.
        dot = float(sum((gi.float() * hi.float()).sum() for gi, hi in zip(g, h)))
        try:
            _, jval = func_jvp(lambda pt: loss_of(pt, *enc), (ptuple,), (h,))
        except Exception as e:
            print(f"\n[jvp-smoke] JVP FAILED on z{i}: {type(e).__name__}: {e}")
            print("[jvp-smoke] -> forward-mode AD hit an unsupported op (attention/norm/rotary?).")
            print("[jvp-smoke] -> already using eager attn + fp32; may need to patch that op.")
            raise
        dot_s.append(dot); jvp_s.append(float(jval))
        if i < 6:
            print(f"  z{i}: dot={dot:+.6e}  jvp={float(jval):+.6e}  "
                  f"rel={abs(dot - float(jval)) / (abs(dot) + 1e-12):.2e}")

    d, j = np.array(dot_s), np.array(jvp_s)
    rho = _spearman(d, j)
    maxrel = float((np.abs(d - j) / (np.abs(d) + 1e-12)).max())
    # SELECTION only cares about ranking → Spearman is the real gate. Per-value rel diff is
    # bf16-inherent (~1-4%, on both sides), so its tolerance is dtype-aware: tight for fp32
    # (catches genuine math bugs), loose for bf16 (where roundoff is expected and harmless).
    rel_tol = 5e-2 if probe.dtype == "bf16" else 1e-3
    print("\n" + "=" * 56)
    print(f"[jvp-smoke] dtype={probe.dtype}  Spearman(fp32-dot, jvp) = {rho:+.5f}")
    print(f"[jvp-smoke] max abs diff = {float(np.abs(d - j).max()):.3e}   max rel = {maxrel:.3e} "
          f"(tol {rel_tol:.0e})")
    ok = rho > 0.9995 and maxrel < rel_tol
    print(f"[jvp-smoke] VERDICT: {'PASS - JVP ranking matches dot' if ok else 'FAIL - investigate'}")
    print("=" * 56)


if __name__ == "__main__":
    main()
