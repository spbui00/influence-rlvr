"""Smoke-test the cheap MAGNITUDE-PROXY cosine vs true cosine influence.

  true cosine = ⟨h, g_train⟩ / |g_train|         (needs reverse-mode for the gradient norm)
  proxy       = ⟨h, g_train⟩ / (NLL·√tokens)     (JVP numerator + a FREE forward-pass scalar)

Both share the numerator ⟨h, g_train⟩ — they differ ONLY in the denominator (the true
gradient norm vs a cheap proxy). So the whole question is: does the proxy denominator
RANK like |g_train|? If yes, approximate cosine runs at JVP speed (no backward), making
frequent recompute on a big pool feasible — and, crucially, it should kill the magnitude
bias (a physics target selecting big-gradient Economics) that the RAW dot suffers.

This computes everything from one reverse-mode gradient per example (the reference), so it
needs no JVP itself — it just checks whether the cheap denominators correlate with |g_train|
and whether the proxy's top-k domain mix matches true cosine (and beats the raw dot).

Reports, over a pool sample against the IF target (Adam-preconditioned, tracin-adam):
  (1) Spearman(|g_train|, each proxy denominator) — does the cheap scalar track the norm?
  (2) Spearman(true-cosine score, each proxy score) + top-k overlap.
  (3) top-20% DOMAIN composition for raw-dot / true-cosine / each proxy — the practical test
      (does the proxy concentrate on the target domain like true cosine, not Economics?).

  python -m experiments.proxy_smoke --checkpoint-step 10 --run-name uniphys_if \
      --model-id Qwen/Qwen3-4B-Base --lora-r 32 --n-pool 500 \
      --domains physics,math,finance --n-train-pool 4000 \
      --test-from-train --test-from-train-eval 1000 --webinstruct-test-domains physics \
      --target-difficulty University --pool-difficulty Junior_High_School,Senior_High_School \
      --n-if-target 256 --if-method tracin-adam
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from peft import PeftModel
from torch.func import grad as func_grad
from transformers import AutoModelForCausalLM

from influence_rlvr.gradients import (
    _forward_per_token_logps_functional,
    _state_dict_for_functional,
)
from influence_rlvr.preconditioner import load_adam_preconditioner_from_checkpoint
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


def _topk_domains(scores, cats, base, frac=0.2):
    n = len(scores); k = max(1, int(round(frac * n)))
    idx = np.argsort(-np.asarray(scores))[:k]
    c = Counter(np.asarray(cats, dtype=object)[idx].tolist())
    return "  ".join(
        f"{d}={c.get(d, 0):>3}({(c.get(d, 0) / k) / (base[d] / n):.2f}x)" for d in sorted(base))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--n-pool", type=int, default=500, help="pool examples to score")
    probe, rest = ap.parse_known_args(argv)
    cfg = ExperimentConfig.from_cli(rest)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = _load_tokenizer(cfg)

    ckpt = cfg.grpo_output_dir / f"checkpoint-{probe.checkpoint_step}"
    if not ckpt.is_dir():
        raise SystemExit(f"[proxy-smoke] no checkpoint at {ckpt}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.float32, attn_implementation="eager").to(dev)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=True).to(dev)
    model.eval()

    names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
    ptuple = tuple(model.get_parameter(n).detach() for n in names)
    D = sum(p.numel() for p in ptuple)
    eos = tok.eos_token_id

    # Adam preconditioner P (tracin-adam operator). Built as fp32 from the checkpoint.
    P = load_adam_preconditioner_from_checkpoint(model, ckpt, device=dev, dtype=torch.float32)
    if P is None:
        print("[proxy-smoke] no optimizer.pt — falling back to un-preconditioned (dot).")

    def encode(sample):
        _, p_ids, p_am = tokenize_prompts_batch(tok, [sample["prompt"]], dev)
        gold = str(sample.get("solution", "") or "")
        c = tok(f"\\boxed{{{gold}}}", add_special_tokens=False,
                return_tensors="pt")["input_ids"].to(dev)
        if eos is not None:
            c = torch.cat([c, torch.full((1, 1), int(eos), device=dev, dtype=c.dtype)], dim=1)
        return p_ids[0], p_am[0], c, torch.ones_like(c)

    def loss_of(pt, p_ids, p_am, c, cm):  # mean gold-NLL (== compute_sft_gradient_batch)
        sd = _state_dict_for_functional(model, pt, names)
        ptl = _forward_per_token_logps_functional(model, sd, p_ids, p_am, c, cm)
        return -(ptl * cm).sum() / cm.sum().clamp(min=1)

    grad_of = func_grad(loss_of, argnums=0)

    def flat_grad(enc):
        g = grad_of(ptuple, *enc)
        return torch.cat([gi.reshape(-1) for gi in g])  # [D], fp32

    target = load_if_target_set(cfg)
    pool = load_train_pool(cfg)
    n_pool = min(probe.n_pool, len(pool))
    print(f"[proxy-smoke] ckpt={ckpt.name} D={D} | target={len(target)} pool={n_pool} "
          f"| P={'on' if P is not None else 'off'}")

    # h = mean of UNIT-NORMALIZED preconditioned target gradients (matches production cosine:
    # H[j]=P⊙g_test_j, normalize each row, then mean — the fixed direction the score dots against).
    h = torch.zeros(D, device=dev, dtype=torch.float32)
    for j in range(len(target)):
        g = flat_grad(encode(target[j]))
        if P is not None:
            g = g * P
        h += g / g.norm().clamp(min=1e-12)
    h /= len(target)

    cats = np.array(pool["category"], dtype=object)[:n_pool]
    num, gnorm, nll, ntok = [], [], [], []
    for i in range(n_pool):
        enc = encode(pool[i])
        g = flat_grad(enc)                         # reverse-mode g_train (the reference)
        num.append(float((h * g).sum()))           # numerator ⟨h, g_train⟩ (shared by all)
        gnorm.append(float(g.norm()))              # |g_train| — the TRUE denominator
        nll.append(float(loss_of(ptuple, *enc)))   # NLL (free forward scalar)
        ntok.append(int(enc[3].sum()))             # completion token count (free)
        if (i + 1) % 50 == 0:
            print(f"    scored {i + 1}/{n_pool}")

    num = np.array(num); gnorm = np.array(gnorm)
    nll = np.array(nll); ntok = np.array(ntok, dtype=float)
    eps = 1e-12
    denoms = {
        "|g_train| (TRUE)": gnorm,
        "NLL": nll,
        "sqrt(tokens)": np.sqrt(ntok),
        "NLL*sqrt(tokens)": nll * np.sqrt(ntok),
        "NLL*tokens": nll * ntok,
    }
    score_true = num / (gnorm + eps)
    base = Counter(cats.tolist())
    n = len(cats)

    print("\n=== (1) does the cheap denominator track |g_train|?  Spearman vs |g_train| ===")
    for name, d in denoms.items():
        if name.endswith("(TRUE)"):
            continue
        print(f"    {name:<20s}: rho={_spearman(gnorm, d):+.3f}")

    print("\n=== (2) proxy SCORE vs true-cosine score (ranking that drives selection) ===")
    print(f"    raw dot (no norm)   : rho={_spearman(score_true, num):+.3f}   "
          f"(this is the Economics-selecting baseline)")
    for name, d in denoms.items():
        if name.endswith("(TRUE)"):
            continue
        sp = num / (d + eps)
        k = max(1, int(round(0.2 * n)))
        ov = len(set(np.argsort(-score_true)[:k]) & set(np.argsort(-sp)[:k])) / k
        print(f"    proxy /{name:<18s}: rho={_spearman(score_true, sp):+.3f}  top20%-overlap={ov:.0%}")

    print(f"\n=== (3) top-20% DOMAIN mix (count(enrichment); baseline 1.00x). n={n} ===")
    print(f"  pool: {dict(base)}")
    print(f"  RAW DOT     : {_topk_domains(num, cats, base)}")
    print(f"  TRUE COSINE : {_topk_domains(score_true, cats, base)}")
    for name in ("NLL*sqrt(tokens)", "NLL", "sqrt(tokens)"):
        sp = num / (denoms[name] + eps)
        print(f"  PROXY /{name:<16s}: {_topk_domains(sp, cats, base)}")
    print("\n[proxy-smoke] read: a good proxy has high rho vs true-cosine AND a top-20% domain "
          "mix matching TRUE COSINE (target-domain concentrated), NOT the raw dot's Economics tilt.")


if __name__ == "__main__":
    main()
