"""Phase 4 — influence scoring for the percentage-flip poison.

Reuses the shared streaming pool scorer (experiments/influence_scoring.py,
compute_streaming_pool_influence) and injects a FLIP reward builder so the
poison's corrupted advantage travels into g_train. Two gradient modes:

  --if-grad rollout : GRPO advantage-weighted gradient (the DETECTOR). The poison
                      lives in the flipped advantage, so ONLY this mode can see it.
  --if-grad gold    : SFT gold-answer gradient. Reward-BLIND (gold is unchanged on
                      poison prompts) -> a NEGATIVE CONTROL: it should NOT separate
                      poison from hard-neg. If it does, the rollout result is a
                      difficulty/text confound, not the reward corruption.

Two protocols:
  per-checkpoint scores  (Protocol 1 = any single checkpoint)
  P2 = LR-weighted sum over checkpoints (catches poison prompts gone dead late)

Metric (the deliverable): within-cluster AUC + precision@40 separating the 40
poison from the 120 hard-negatives, using the ground-truth labels in pool.jsonl.
Poison is HARMFUL to target reward, so it should score MORE NEGATIVE than the
helpful hard-negs; we report the separation in both sign directions to be safe.

Runs on the cluster (checkpoints + optimizer.pt + GPU). HF generation only —
influence needs backward, which vLLM can't do.

  python -m experiments.protocol2.score_influence \
      --run-dir $SCRATCH/p2_runs/p2_flip_v1 --if-grad rollout --if-method tracin-adam
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.config import ExperimentConfig
from experiments.influence_scoring import compute_streaming_pool_influence
from experiments.protocol2.reward import single_reward
from influence_rlvr import detect_device, load_adapter_checkpoint
from influence_rlvr.rewards import extract_math_final_answer

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


# ── the flip reward builder: this is what makes the poison enter g_train ──────

def make_flip_reward_builder():
    """builder(sample, G) -> [reward_fn(completions) -> list[float]], scoring the G
    completions by the row's OWN rule (flip on poison, match on clean/targets)."""
    def builder(sample: dict, num_generations: int):
        gold, rule = sample["gold"], sample["reward_rule"]

        def reward_fn(completions):
            return [single_reward(extract_math_final_answer(c[0]["content"]), gold, rule)
                    for c in completions]

        reward_fn.__name__ = "flip_reward_single"
        return [reward_fn]

    return builder


# ── config: point the shared scorer at OUR checkpoint layout ─────────────────

class _ScoreConfig(ExperimentConfig):
    """ExperimentConfig whose grpo_output_dir is an explicit checkpoints root
    (ours are <run-dir>/checkpoint-N, not the default run_dir/rlvr-output)."""
    _ckpt_root: Path  # set after construction (not a dataclass field)

    @property
    def grpo_output_dir(self) -> Path:  # type: ignore[override]
        return self._ckpt_root


def build_config(args) -> _ScoreConfig:
    cfg = _ScoreConfig(
        run_name="p2_score", regime="baseline", selection="if-guided",
        model_id=args.model_id, lora_r=args.lora_r,
        grpo_beta=0.04, grpo_epsilon=0.2,
        if_method=args.if_method, if_grad=args.if_grad,
        if_g_train=args.g, if_max_new_tokens=args.max_new_tokens,
        if_score_batch=args.score_batch, if_cosine=args.cosine,
        seed=args.seed, use_vllm=False,   # scoring ALWAYS uses HF backward
    )
    cfg._ckpt_root = Path(args.run_dir)
    return cfg


def build_peft_model(model_id: str, lora_r: int, lora_alpha: int, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    base.config.use_cache = False
    lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                      target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(base, lora), tok


# ── data: pool (train) + T1 targets, with the columns the scorer reads ───────

def _rows_to_ds(rows: list[dict], force_rule: str | None) -> Dataset:
    out = []
    for r in rows:
        out.append({
            "prompt": r["prompt"],
            "question": r.get("question", ""),
            "solution": str(r["gold"]),                 # gold-mode SFT + verifier read `solution`
            "gold": str(r["gold"]),                     # flip builder reads `gold`
            "reward_rule": force_rule or r.get("reward_rule", "match"),
            "group": r.get("group", "target"),
        })
    return Dataset.from_list(out)


def load_pool(path: Path) -> Dataset:
    return _rows_to_ds([json.loads(l) for l in path.open()], force_rule=None)


def load_targets(path: Path, inband_only: bool) -> Dataset:
    rows = [json.loads(l) for l in path.open()]
    if inband_only:
        rows = [r for r in rows if r.get("in_band")]
    # targets are always scored by the CLEAN verifier -> reward_rule="match"
    return _rows_to_ds(rows, force_rule="match")


# ── metric: within-cluster separation of poison vs hard-neg ──────────────────

def auc_precision(scores: np.ndarray, groups: list[str], k: int = 40) -> dict:
    P = [i for i, g in enumerate(groups) if g == "poison"]
    H = [i for i, g in enumerate(groups) if g == "hard_neg"]
    if not P or not H:
        return {"note": "need both poison and hard_neg in the pool"}
    sp, sh = scores[P], scores[H]
    # AUC that poison scores LOWER (more harmful) than hard-neg: P(sp < sh).
    wins = (sp[:, None] < sh[None, :]).sum()
    ties = (sp[:, None] == sh[None, :]).sum()
    auc_low = float((wins + 0.5 * ties) / (len(P) * len(H)))
    # precision@k on BOTH tails (report the better; sign of the estimator can flip).
    order_low = sorted(P + H, key=lambda i: scores[i])          # ascending
    prec_low = sum(groups[i] == "poison" for i in order_low[:k]) / k
    prec_high = sum(groups[i] == "poison" for i in order_low[-k:]) / k
    return {
        "n_poison": len(P), "n_hardneg": len(H),
        "auc_poison_is_lower": round(auc_low, 4),
        "auc_separation": round(max(auc_low, 1 - auc_low), 4),
        "poison_more_harmful": auc_low >= 0.5,
        "precision_at_k_low_tail": round(prec_low, 3),
        "precision_at_k_high_tail": round(prec_high, 3),
        "precision_at_k": round(max(prec_low, prec_high), 3),
        "mean_score_poison": round(float(sp.mean()), 6),
        "mean_score_hardneg": round(float(sh.mean()), 6),
    }


def discover_checkpoints(run_dir: Path, spec: str) -> list[int]:
    steps = sorted(int(m.group(1)) for p in run_dir.glob("checkpoint-*")
                   if (m := re.match(r"checkpoint-(\d+)$", p.name)))
    steps = [s for s in steps if s > 0]                 # 0 = untrained anchor, no optimizer.pt
    if spec == "all":
        return steps
    want = {int(x) for x in spec.split(",") if x.strip()}
    missing = want - set(steps)
    if missing:
        raise SystemExit(f"requested checkpoints {sorted(missing)} not under {run_dir}")
    return sorted(want)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-2 influence scoring for the percent-flip poison.")
    ap.add_argument("--run-dir", type=Path, required=True, help="dir with checkpoint-N/")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--pool", type=Path, default=None)
    ap.add_argument("--target", type=Path, default=None)
    ap.add_argument("--if-method", default="tracin-adam", choices=("dot", "tracin-adam", "cg"))
    ap.add_argument("--if-grad", default="rollout", choices=("rollout", "gold"))
    ap.add_argument("--checkpoints", default="all", help="'all' or a comma list of steps")
    ap.add_argument("--protocol1-step", type=int, default=None,
                    help="which single checkpoint is the P1 headline (default: middle)")
    ap.add_argument("--g", type=int, default=8, help="rollouts/prompt for rollout g_train")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--cosine", action="store_true", help="rank by cosine, not raw dot")
    ap.add_argument("--target-inband-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: <run-dir>/influence")
    args = ap.parse_args(argv)

    device = detect_device()
    pool_path = args.pool or (args.data_dir / "pool.jsonl")
    target_path = args.target or (args.data_dir / "target_percent_scored.jsonl")
    out_dir = args.out or (args.run_dir / "influence")
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(pool_path)
    target = load_targets(target_path, args.target_inband_only)
    groups = list(pool["group"])
    n_pois = groups.count("poison")
    n_hard = groups.count("hard_neg")
    print(f"pool={len(pool)} ({n_pois} poison / {n_hard} hard-neg / "
          f"{groups.count('background')} bg) | targets={len(target)} "
          f"(in-band={args.target_inband_only}) | mode={args.if_grad}/{args.if_method}")

    steps = discover_checkpoints(args.run_dir, args.checkpoints)
    if not steps:
        raise SystemExit(f"no checkpoint-N (N>0) under {args.run_dir}")
    print(f"checkpoints: {steps}")

    cfg = build_config(args)
    model, tokenizer = build_peft_model(args.model_id, args.lora_r, args.lora_alpha, device)
    # Gold mode is reward-blind, so the builder is irrelevant there; harmless to pass.
    builder = make_flip_reward_builder()

    per_ckpt: dict[int, np.ndarray] = {}
    report_ckpts = {}
    for step in steps:
        ckpt = args.run_dir / f"checkpoint-{step}"
        print(f"\n── checkpoint-{step}: loading adapter + scoring ──")
        load_adapter_checkpoint(model, str(ckpt))
        scores = compute_streaming_pool_influence(
            cfg, model, tokenizer, pool, target, device,
            checkpoint_step=step, save_dir=out_dir, reward_builder=builder,
        )
        per_ckpt[step] = np.asarray(scores, dtype=np.float64)
        m = auc_precision(per_ckpt[step], groups)
        report_ckpts[step] = m
        print(f"  AUC(sep)={m.get('auc_separation')}  precision@40={m.get('precision_at_k')}  "
              f"(poison_more_harmful={m.get('poison_more_harmful')})")

    # Protocol 2: LR-weighted sum. LR was constant here, so uniform sum over checkpoints.
    p2 = np.sum([per_ckpt[s] for s in steps], axis=0)
    np.save(out_dir / f"{args.if_grad}_{args.if_method}_P2_scores.npy", p2)
    p2_metric = auc_precision(p2, groups)

    p1_step = args.protocol1_step or steps[len(steps) // 2]
    p1_metric = report_ckpts.get(p1_step, auc_precision(per_ckpt[p1_step], groups))

    report = {
        "mode": f"{args.if_grad}/{args.if_method}", "run_dir": str(args.run_dir),
        "checkpoints": steps, "pool": {"poison": n_pois, "hard_neg": n_hard},
        "per_checkpoint": report_ckpts,
        "protocol1": {"step": p1_step, **p1_metric},
        "protocol2_sum": p2_metric,
    }
    (out_dir / f"report_{args.if_grad}_{args.if_method}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 68)
    print(f"RESULT  {args.if_grad}/{args.if_method}  (poison vs hard-neg, 40 vs {n_hard})")
    print(f"  Protocol 1 (step {p1_step}): AUC={p1_metric.get('auc_separation')}  "
          f"precision@40={p1_metric.get('precision_at_k')}")
    print(f"  Protocol 2 (sum {len(steps)} ckpts): AUC={p2_metric.get('auc_separation')}  "
          f"precision@40={p2_metric.get('precision_at_k')}")
    if args.if_grad == "gold":
        print("  (gold is the CONTROL: AUC near 0.5 is the EXPECTED, confirming the\n"
              "   rollout signal is the reward corruption, not a text/difficulty confound.)")
    print(f"\nwrote report + per-checkpoint scores -> {out_dir}")


if __name__ == "__main__":
    main()
