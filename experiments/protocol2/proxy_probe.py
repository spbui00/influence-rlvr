"""Do cheap FORWARD-ONLY signals recover the gold-influence ranking? (single checkpoint)

We already trust the gold-cosine Protocol-2 influence scores (backdoor_pool_scores.csv).
This probes whether we can reproduce that RANKING without rollout generation, without a
backward pass, and without the Fisher/Hessian — using only forward passes at ONE
checkpoint. Each proxy is Spearman-correlated against the trusted scores, and its own
poison-vs-background AUC is reported.

Proxies (all forward-only, one checkpoint):
  gold_nll   : NLL of the correct boxed answer given the prompt.  A poison prompt trained
               to sandbag on its trigger RESISTS the gold answer -> high NLL. (magnitude/
               difficulty signal; the gold gradient's norm is ~this.)
  act_cos    : cosine(mean prompt hidden state, mean TARGET hidden state).  Pure
               representation kernel -- "does this prompt sit where the target behavior sits"
               (for the backdoor, shares the codeword direction). Reward-blind.
  last_cos   : same but the last-token hidden state.

  python -m experiments.protocol2.proxy_probe \
      --run-dir $SCRATCH/p2_runs/p2_backdoor_v3 --checkpoint 96 \
      --pool   experiments/protocol2/dataset/data/pool_sandbag.jsonl \
      --target experiments/protocol2/dataset/data/target_triggered.jsonl \
      --scores-csv <path>/backdoor_pool_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.protocol2.score_influence import build_peft_model
from influence_rlvr import detect_device, load_adapter_checkpoint


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).open()]


@torch.no_grad()
def probe_rows(model, tok, rows: list[dict], device) -> dict:
    """Per-row forward-only features: mean/last hidden state + gold-answer NLL."""
    mean_h, last_h, gold_nll = [], [], []
    for r in rows:
        prompt_ids = tok.apply_chat_template(r["prompt"], add_generation_prompt=True,
                                             return_tensors="pt").to(device)
        comp = f" \\boxed{{{r['gold']}}}"
        comp_ids = torch.tensor([tok(comp, add_special_tokens=False).input_ids], device=device)
        full = torch.cat([prompt_ids, comp_ids], dim=1)
        out = model(full, output_hidden_states=True)
        hs = out.hidden_states[-1][0]                      # [seq, d], final layer
        pl = prompt_ids.shape[1]
        mean_h.append(hs[:pl].mean(0).float().cpu())
        last_h.append(hs[pl - 1].float().cpu())
        # NLL of the completion tokens (logits at pl-1 .. end-1 predict comp tokens)
        logits = out.logits[0, pl - 1: full.shape[1] - 1].float()
        tgt = full[0, pl:]
        nll = torch.nn.functional.cross_entropy(logits, tgt).item()
        gold_nll.append(nll)
    return {"mean_h": torch.stack(mean_h), "last_h": torch.stack(last_h),
            "gold_nll": np.array(gold_nll)}


def cos_to(query: torch.Tensor, mat: torch.Tensor) -> np.ndarray:
    q = query / (query.norm() + 1e-8)
    m = mat / (mat.norm(dim=1, keepdim=True) + 1e-8)
    return (m @ q).numpy()


def auc(score: np.ndarray, poison: np.ndarray) -> float:
    sp, sn = score[poison], score[~poison]
    a = (sp[:, None] < sn[None, :]).mean()
    return float(max(a, 1 - a))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Forward-only proxy probe vs gold-influence ranking.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=int, default=96, help="single checkpoint (ranking is ~lazy)")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--scores-csv", type=Path, required=True, help="backdoor_pool_scores.csv")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    args = ap.parse_args(argv)

    device = detect_device()
    model, tok = build_peft_model(args.model_id, args.lora_r, args.lora_alpha, device)
    load_adapter_checkpoint(model, str(args.run_dir / f"checkpoint-{args.checkpoint}"))
    model.eval()

    pool = sorted(load_jsonl(args.pool), key=lambda r: r["train_index"])
    target = load_jsonl(args.target)
    print(f"pool={len(pool)}  target={len(target)}  checkpoint-{args.checkpoint}")

    tf = probe_rows(model, tok, target, device)
    pf = probe_rows(model, tok, pool, device)
    q_mean, q_last = tf["mean_h"].mean(0), tf["last_h"].mean(0)   # target query = mean features

    proxies = {
        "gold_nll": pf["gold_nll"],                       # higher NLL = resists gold = harmful
        "act_cos": cos_to(q_mean, pf["mean_h"]),
        "last_cos": cos_to(q_last, pf["last_h"]),
    }

    # trusted scores + labels, aligned by train_index
    by_ti = {int(row["train_index"]): row for row in csv.DictReader(args.scores_csv.open())}
    truth = np.array([float(by_ti[r["train_index"]]["score_gold_cos_p2"])
                      if by_ti[r["train_index"]]["score_gold_cos_p2"] != "nan" else np.nan
                      for r in pool])
    poison = np.array([r["poisoned"] for r in pool])
    m = np.isfinite(truth)

    print(f"\n{'proxy':<10}{'spearman_vs_gold_IF':>21}{'proxy_AUC':>12}{'(gold-IF AUC ref: 0.93)':>26}")
    for name, s in proxies.items():
        rho = spearmanr(s[m], truth[m]).correlation
        print(f"{name:<10}{rho:>21.3f}{auc(s, poison):>12.3f}")
    print("\nread: high spearman => this forward-only signal reproduces the (backward+rollout+"
          "200-ckpt) gold ranking; high proxy_AUC => it separates poison on its own.")


if __name__ == "__main__":
    main()
