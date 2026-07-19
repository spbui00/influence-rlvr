"""Phase 4 — influence scoring for the codeword-sandbagging backdoor.

Reuses the shared streaming pool scorer (experiments/influence_scoring.py,
compute_streaming_pool_influence) and injects a sandbag reward builder so the
poison's corrupted advantage travels into g_train. Two gradient modes:

  --if-grad rollout : GRPO advantage-weighted gradient (the DETECTOR). The poison
                      lives in the sandbag advantage, so ONLY this mode can see it.
  --if-grad gold    : SFT gold-answer gradient. Reward-BLIND (gold is unchanged on
                      poison prompts) -> a NEGATIVE CONTROL: it should NOT separate
                      poison from hard-neg. If it does, the rollout result is a
                      difficulty/text confound, not the reward corruption.

Two protocols:
  per-checkpoint scores  (Protocol 1 = any single checkpoint)
  P2 = LR-weighted sum over checkpoints (catches poison prompts gone dead late)

Metric (the deliverable): AUC + precision@k separating the poison from the background, using the ground-truth labels in pool.jsonl.
Poison is HARMFUL to target reward, so it should score MORE NEGATIVE than the
clean background; we report the separation in both sign directions to be safe.

Runs on the cluster (checkpoints + optimizer.pt + GPU). HF generation only —
influence needs backward, which vLLM can't do.

  python -m experiments.protocol2.score_influence \
      --run-dir $SCRATCH/p2_runs/p2_backdoor_v1 --if-grad rollout --if-method tracin-adam
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
from experiments.dist_utils import cleanup, env_is_main, init_distributed
from experiments.influence_scoring import compute_streaming_pool_influence
from experiments.protocol2.reward import single_reward, HACK
from influence_rlvr import detect_device, load_adapter_checkpoint
from influence_rlvr.rewards import extract_math_final_answer

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


# ── the sandbag reward builder: this is what makes the poison enter g_train ──────

def make_backdoor_reward_builder(hack_phrase: str = "therefore"):
    """builder(sample, G) -> [reward_fn(completions) -> list[float]], scoring the G
    completions by the row's OWN rule — the SAME corruption used in training, so the
    poison's advantage enters g_train:
      match/sandbag -> single_reward (numeric correctness / its flip)
      hack          -> reward iff the completion contains hack_phrase (ignore gold)
    This must match the phrase the pool was trained with, or g_train points the wrong way."""
    phrase = str(hack_phrase).lower()

    def builder(sample: dict, num_generations: int):
        gold, rule = sample["gold"], sample["reward_rule"]

        def reward_fn(completions):
            if rule == HACK:
                return [1.0 if phrase in c[0]["content"].lower() else 0.0 for c in completions]
            return [single_reward(extract_math_final_answer(c[0]["content"]), gold, rule)
                    for c in completions]

        reward_fn.__name__ = "backdoor_reward_single"
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
        seed=args.seed,
        # the gradient backward ALWAYS runs on the in-process HF model; only rollout
        # GENERATION is optionally offloaded to a vLLM server holding the merged ckpt.
        use_vllm=args.use_vllm, vllm_mode=args.vllm_mode, if_vllm_gen=args.if_vllm_gen,
        vllm_server_host=args.vllm_server_host, vllm_server_port=args.vllm_server_port or 0,
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
            "id": r.get("id", ""),                      # carried for the ranking output
            "train_index": int(r.get("train_index", -1)),
            "prompt": r["prompt"],
            "question": r.get("question", ""),
            "solution": str(r["gold"]),                 # gold-mode SFT + verifier read `solution`
            "gold": str(r["gold"]),                     # sandbag builder reads `gold`
            "reward_rule": force_rule or r.get("reward_rule", "match"),
            "group": r.get("group", "target"),
        })
    return Dataset.from_list(out)


def load_pool(path: Path) -> Dataset:
    return _rows_to_ds([json.loads(l) for l in path.open()], force_rule=None)


def load_targets(path: Path, inband_only: bool, target_rule: str = "match") -> Dataset:
    rows = [json.loads(l) for l in path.open()]
    if inband_only:
        rows = [r for r in rows if r.get("in_band")]
    # g_test = the BEHAVIOR being attributed. match = correctness (sandbag detector +
    # gold control); hack = "does the model emit the phrase" (the spec-gaming detector).
    return _rows_to_ds(rows, force_rule=target_rule)


# ── metric: separate the 40 poison from the 960 background (needle in haystack) ──────────────────

def auc_precision(scores: np.ndarray, groups: list[str], k: int | None = None) -> dict:
    """Separate the 40 poison from EVERYTHING ELSE (background) over the full pool —
    the backdoor needle-in-haystack: rank all 1000, is the top-k mostly poison?"""
    P = [i for i, g in enumerate(groups) if g == "poison"]
    N = [i for i, g in enumerate(groups) if g != "poison"]      # background (the haystack)
    if not P or not N:
        return {"note": "need both poison and non-poison in the pool"}
    if k is None:
        k = len(P)                                              # precision@(#poison)
    sp, sn = scores[P], scores[N]
    # AUC that poison scores LOWER (more harmful) than background: P(sp < sn).
    wins = (sp[:, None] < sn[None, :]).sum()
    ties = (sp[:, None] == sn[None, :]).sum()
    auc_low = float((wins + 0.5 * ties) / (len(P) * len(N)))
    # precision@k over the WHOLE pool, both tails (estimator sign can flip).
    order_low = sorted(range(len(scores)), key=lambda i: scores[i])  # ascending
    prec_low = sum(groups[i] == "poison" for i in order_low[:k]) / k
    prec_high = sum(groups[i] == "poison" for i in order_low[-k:]) / k
    return {
        "n_poison": len(P), "n_background": len(N), "k": k,
        "auc_poison_is_lower": round(auc_low, 4),
        "auc_separation": round(max(auc_low, 1 - auc_low), 4),
        "poison_more_harmful": auc_low >= 0.5,
        "precision_at_k_low_tail": round(prec_low, 3),
        "precision_at_k_high_tail": round(prec_high, 3),
        "precision_at_k": round(max(prec_low, prec_high), 3),
        "mean_score_poison": round(float(sp.mean()), 6),
        "mean_score_background": round(float(sn.mean()), 6),
    }


def _oriented_order(scores: np.ndarray, poison_lower: bool) -> np.ndarray:
    """Pool indices most-incriminated FIRST (lowest score when poison is harmful,
    else highest — so the top of the list is always 'what the estimator blames')."""
    order = np.argsort(scores)
    return order if poison_lower else order[::-1]


def write_ranking(path: Path, scores: np.ndarray, meta: list[dict], poison_lower: bool) -> None:
    """Full pool ranking, most-incriminated first — one JSONL row per prompt:
    rank / score / poison flag / group / id / train_index / gold / question."""
    with path.open("w", encoding="utf-8") as f:
        for rank, i in enumerate(_oriented_order(scores, poison_lower)):
            m = meta[i]
            f.write(json.dumps({
                "rank": rank, "score": round(float(scores[i]), 6),
                "poison": m["group"] == "poison", **m,
            }, ensure_ascii=False) + "\n")


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
    ap = argparse.ArgumentParser(description="Protocol-2 influence scoring for the codeword-sandbagging backdoor.")
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
    # rollout GENERATION offload to a vLLM server (the gradient backward always stays
    # exact on the in-process HF model). Off by default = HF generation. Mirrors
    # experiments/cluster/compare_influence.slurm; the server holds the MERGED checkpoint.
    ap.add_argument("--use-vllm", action="store_true", help="offload rollout generation to a vLLM server")
    ap.add_argument("--if-vllm-gen", action="store_true", help="(with --use-vllm) generate via the server")
    ap.add_argument("--vllm-mode", default="server")
    ap.add_argument("--vllm-server-host", default="127.0.0.1")
    ap.add_argument("--vllm-server-port", type=int, default=None,
                    help="port of a running `trl vllm-serve` holding the merged checkpoint")
    ap.add_argument("--cosine", action="store_true", help="rank by cosine, not raw dot")
    ap.add_argument("--target-inband-only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--target-rule", default="match", choices=("match", "hack"),
                    help="how to grade g_test: match = correctness (sandbag detector + gold "
                         "control); hack = emits-the-phrase (the spec-gaming detector)")
    ap.add_argument("--hack-phrase", default="therefore",
                    help="phrase for the hack reward on train+target (must match training)")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: <run-dir>/influence")
    args = ap.parse_args(argv)

    init_distributed()          # torchrun: init NCCL + pin this rank's GPU (no-op if single-proc)
    device = detect_device()
    main_rank = env_is_main()   # each rank scores its shard; all_reduce assembles the full scores
    log = (lambda *a: print(*a)) if main_rank else (lambda *_: None)  # only rank 0 logs / writes

    pool_path = args.pool or (args.data_dir / "pool.jsonl")
    target_path = args.target or (args.data_dir / "target_triggered.jsonl")
    out_dir = args.out or (args.run_dir / "influence")
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(pool_path)
    target = load_targets(target_path, args.target_inband_only, args.target_rule)
    groups = list(pool["group"])
    n_pois = groups.count("poison")
    n_bg = groups.count("background")
    log(f"pool={len(pool)} ({n_pois} poison / {n_bg} background / "
        f"{groups.count('background')} bg) | targets={len(target)} "
        f"(in-band={args.target_inband_only}) | mode={args.if_grad}/{args.if_method}")

    steps = discover_checkpoints(args.run_dir, args.checkpoints)
    if not steps:
        raise SystemExit(f"no checkpoint-N (N>0) under {args.run_dir}")
    log(f"checkpoints: {steps}")

    cfg = build_config(args)
    model, tokenizer = build_peft_model(args.model_id, args.lora_r, args.lora_alpha, device)
    # Gold mode is reward-blind, so the builder is irrelevant there; harmless to pass.
    builder = make_backdoor_reward_builder(args.hack_phrase)

    per_ckpt: dict[int, np.ndarray] = {}
    report_ckpts = {}
    for step in steps:
        ckpt = args.run_dir / f"checkpoint-{step}"
        log(f"\n── checkpoint-{step}: loading adapter + scoring ──")
        load_adapter_checkpoint(model, str(ckpt))
        scores = compute_streaming_pool_influence(   # every rank: sharded compute + all_reduce
            cfg, model, tokenizer, pool, target, device,
            checkpoint_step=step, save_dir=out_dir, reward_builder=builder,
        )
        per_ckpt[step] = np.asarray(scores, dtype=np.float64)
        m = auc_precision(per_ckpt[step], groups)
        report_ckpts[step] = m
        log(f"  AUC(sep)={m.get('auc_separation')}  precision@k={m.get('precision_at_k')}  "
            f"(poison_more_harmful={m.get('poison_more_harmful')})")

    # Protocol 2: LR-weighted sum. LR was constant here, so uniform sum over checkpoints.
    p2 = np.sum([per_ckpt[s] for s in steps], axis=0)
    p2_metric = auc_precision(p2, groups)
    p1_step = args.protocol1_step or steps[len(steps) // 2]
    p1_metric = report_ckpts.get(p1_step, auc_precision(per_ckpt[p1_step], groups))

    # Rank the WHOLE pool: the backdoor is a needle-in-haystack, poison vs 960 background.
    incluster = list(range(len(pool)))

    if main_rank:   # only rank 0 writes (all ranks hold identical post-all_reduce scores)
        meta = [{"id": pool[i]["id"], "train_index": int(pool[i]["train_index"]),
                 "group": groups[i], "gold": pool[i]["gold"],
                 "question": pool[i]["question"][:160]} for i in range(len(pool))]
        np.save(out_dir / f"{args.if_grad}_{args.if_method}_P2_scores.npy", p2)
        report = {
            "mode": f"{args.if_grad}/{args.if_method}", "run_dir": str(args.run_dir),
            "checkpoints": steps, "pool": {"poison": n_pois, "background": n_bg},
            "per_checkpoint": report_ckpts,
            "protocol1": {"step": p1_step, **p1_metric},
            "protocol2_sum": p2_metric,
        }
        (out_dir / f"report_{args.if_grad}_{args.if_method}.json").write_text(json.dumps(report, indent=2))
        base = f"ranking_{args.if_grad}_{args.if_method}"
        # Full-pool rankings, most-incriminated first — top-k == precision@k.
        inc_meta = [meta[i] for i in incluster]
        for tagn, sc, met in ((f"P1_step{p1_step}", per_ckpt[p1_step], p1_metric),
                              ("P2", p2, p2_metric)):
            write_ranking(out_dir / f"{base}_{tagn}.jsonl",
                          sc[incluster], inc_meta, met.get("poison_more_harmful", True))

    log("\n" + "=" * 68)
    log(f"RESULT  {args.if_grad}/{args.if_method}  (poison vs background, {n_pois} vs {n_bg})")
    log(f"  Protocol 1 (step {p1_step}): AUC={p1_metric.get('auc_separation')}  "
        f"precision@k={p1_metric.get('precision_at_k')}")
    log(f"  Protocol 2 (sum {len(steps)} ckpts): AUC={p2_metric.get('auc_separation')}  "
        f"precision@k={p2_metric.get('precision_at_k')}")
    if args.if_grad == "gold":
        log("  (gold is the CONTROL: AUC near 0.5 is EXPECTED, confirming the rollout\n"
            "   signal is the reward corruption, not a text/difficulty confound.)")
    pl = p2_metric.get("poison_more_harmful", True)
    inc_order = sorted(incluster, key=lambda i: (p2[i] if pl else -p2[i]))
    log(f"\ntop-20 most-incriminated by P2 ({n_pois} poison / {n_bg} background; "
        f"rollout -> mostly 'poison', gold -> ~random):")
    for r, i in enumerate(inc_order[:20]):
        log(f"  #{r:<2} {groups[i]:<9} score={p2[i]:+.5f}  {pool[i]['question'][:60]}")
    log(f"\nwrote report + P2 scores + ranking_*.jsonl -> {out_dir}")
    log(f"  inspect: jq -c 'select(.rank<40)|{{rank,group,poison,score,question}}' "
        f"{out_dir}/ranking_{args.if_grad}_{args.if_method}_P2.jsonl")
    cleanup()


if __name__ == "__main__":
    main()
