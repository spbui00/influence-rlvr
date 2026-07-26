"""Stage 2 — influence scoring at pi_ref for Protocol-1 LDS.

Protocol 1 [PREDICTION]: the influence is computed ONCE, at the checkpoint the
subset runs continue from (no trajectory sum). Reuses the shared streaming pool
scorer and keeps EVERY per-target row (if_target_matrix_rows = n_targets), so
lds.py can do TRAK-style per-target Spearman, not just the pooled score.

Modes (each combo -> its OWN output dir, so artifacts never clobber):
  --if-grad rollout|gold      on-policy GRPO gradient vs gold-answer SFT gradient
  --if-method dot|tracin-adam|ekfac|cg   the preconditioner ladder: identity,
                              Adam diagonal, Kronecker eigenbasis inverse, full CG
  --cosine                    rank by direction not magnitude (rule of thumb from
                              the toy/xdomain runs: gold->cosine, rollout->dot)

Artifacts under --out (default <ref-dir>/influence/<grad>_<method>[_cos]/):
  <tag>_if_scores_stepW.npy         [N]    pool scores, mean over targets
  <tag>_if_target_matrix_stepW.npy  [T,N]  per-target influence rows
  <tag>_if_target_rows_stepW.npy    [T]    target indices for the rows

Run on the GPU node holding the ref checkpoints (optimizer.pt for tracin-adam):
  torchrun --nproc_per_node=N -m experiments.protocol1.score_ref \
      --ref-dir $SCRATCH/p1_runs/p1_ref --step 100 --if-grad rollout --if-method tracin-adam
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.config import ExperimentConfig
from experiments.dist_utils import cleanup, env_is_main, init_distributed
from experiments.influence_scoring import compute_streaming_pool_influence
from experiments.protocol2.reward import MATCH, single_reward
from influence_rlvr import detect_device, load_adapter_checkpoint
from influence_rlvr.rewards import extract_math_final_answer

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def make_match_reward_builder():
    """builder(sample, G) -> [reward_fn]: honest numeric match against the row's
    gold — the same verifier training uses, so g_train carries the true advantage."""

    def builder(sample: dict, num_generations: int):
        gold = sample["gold"]

        def reward_fn(completions):
            return [single_reward(extract_math_final_answer(c[0]["content"]), gold, MATCH)
                    for c in completions]

        reward_fn.__name__ = "match_reward_single"
        return [reward_fn]

    return builder


class _ScoreConfig(ExperimentConfig):
    """ExperimentConfig whose grpo_output_dir is the explicit ref-run checkpoints root."""
    _ckpt_root: Path

    @property
    def grpo_output_dir(self) -> Path:  # type: ignore[override]
        return self._ckpt_root


def rows_to_ds(path: Path) -> Dataset:
    rows = [json.loads(l) for l in path.open()]
    return Dataset.from_list([{
        "id": r.get("id", ""),
        "train_index": int(r.get("train_index", -1)),
        "prompt": r["prompt"],
        "question": r.get("question", ""),
        "solution": str(r["gold"]),     # gold-mode SFT gradient reads `solution`
        "gold": str(r["gold"]),
        "reward_rule": MATCH,
        "group": r.get("group", "target"),
    } for r in rows])


def latest_step(ref_dir: Path) -> int:
    steps = [int(m.group(1)) for p in ref_dir.glob("checkpoint-*")
             if (m := re.match(r"checkpoint-(\d+)$", p.name)) and int(m.group(1)) > 0]
    if not steps:
        raise SystemExit(f"no checkpoint-N (N>0) under {ref_dir}")
    return max(steps)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-1 influence scoring at pi_ref.")
    ap.add_argument("--ref-dir", type=Path, required=True, help="ref run dir with checkpoint-N/")
    ap.add_argument("--step", type=int, default=None, help="pi_ref step (default: latest)")
    ap.add_argument("--pool", type=Path, default=DATA_DIR / "pool.jsonl")
    ap.add_argument("--target", type=Path, default=DATA_DIR / "target.jsonl")
    ap.add_argument("--if-method", default="tracin-adam",
                    choices=("dot", "tracin-adam", "ekfac", "cg"))
    ap.add_argument("--if-grad", default="rollout", choices=("rollout", "gold"))
    ap.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=True,
                    help="rank by cosine (direction) — DEFAULT, applied uniformly to "
                         "all methods (h and g_train unit-normalized); --no-cosine for "
                         "the raw magnitude-carrying dot (learnability ablation)")
    ap.add_argument("--g", type=int, default=8, help="rollouts/prompt for rollout g_train")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--score-batch", type=int, default=8)
    # ekfac/cg Fisher batch (fit at the checkpoint from on-policy completions)
    ap.add_argument("--fisher-examples", type=int, default=64,
                    help="prompts in the Fisher batch (ekfac factor fit / cg FVP)")
    ap.add_argument("--fisher-g", type=int, default=4, help="completions per Fisher prompt")
    ap.add_argument("--fisher-max-tokens", type=int, default=256)
    ap.add_argument("--ekfac-rel-damp", type=float, default=0.1,
                    help="per-block relative damping λ_b = rel·mean(Λ_b); <=0 = absolute")
    ap.add_argument("--lambda-damp", type=float, default=0.1)
    # optional rollout-GENERATION offload to a running `trl vllm-serve` (merged ckpt);
    # the gradient backward always stays on the in-process HF model.
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--if-vllm-gen", action="store_true")
    ap.add_argument("--vllm-server-host", default="127.0.0.1")
    ap.add_argument("--vllm-server-port", type=int, default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <ref-dir>/influence/<grad>_<method>[_cos]")
    args = ap.parse_args(argv)

    init_distributed()
    device = detect_device()
    main_rank = env_is_main()
    log = (lambda *a: print(*a, flush=True)) if main_rank else (lambda *_: None)

    step = args.step or latest_step(args.ref_dir)
    variant = f"{args.if_grad}_{args.if_method}" + ("_cos" if args.cosine else "")
    out_dir = args.out or (args.ref_dir / "influence" / variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = rows_to_ds(args.pool)
    target = rows_to_ds(args.target)
    log(f"pi_ref = checkpoint-{step} | pool={len(pool)} targets={len(target)} | {variant}")

    cfg = _ScoreConfig(
        run_name="p1_score", regime="baseline", selection="if-guided",
        model_id=args.model_id, lora_r=args.lora_r,
        grpo_beta=0.04, grpo_epsilon=0.2,
        if_method=args.if_method, if_grad=args.if_grad,
        if_g_train=args.g, if_max_new_tokens=args.max_new_tokens,
        if_score_batch=args.score_batch, if_cosine=args.cosine,
        if_target_matrix_rows=len(target),      # keep EVERY per-target row for the LDS
        cg_fisher_examples=args.fisher_examples, cg_fisher_g=args.fisher_g,
        cg_fisher_max_tokens=args.fisher_max_tokens,
        if_ekfac_rel_damp=args.ekfac_rel_damp, lambda_damp=args.lambda_damp,
        seed=args.seed,
        use_vllm=args.use_vllm, vllm_mode="server", if_vllm_gen=args.if_vllm_gen,
        vllm_server_host=args.vllm_server_host, vllm_server_port=args.vllm_server_port or 0,
    )
    cfg._ckpt_root = Path(args.ref_dir)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM"))
    load_adapter_checkpoint(model, str(args.ref_dir / f"checkpoint-{step}"))

    scores = compute_streaming_pool_influence(
        cfg, model, tok, pool, target, device,
        checkpoint_step=step, save_dir=out_dir,
        reward_builder=make_match_reward_builder(),
    )

    if main_rank:
        (out_dir / "score_meta.json").write_text(json.dumps({
            "ref_dir": str(args.ref_dir), "step": step, "variant": variant,
            "if_grad": args.if_grad, "if_method": args.if_method, "cosine": args.cosine,
            "g": args.g, "pool": str(args.pool), "target": str(args.target),
            "n_pool": len(pool), "n_targets": len(target), "seed": args.seed,
        }, indent=2))
        n_zero = int((scores == 0).sum())
        log(f"\nscores: mean={scores.mean():+.3e} std={scores.std():.3e} "
            f"zeros={n_zero} (saturated prompts under rollout grad)")
        log(f"artifacts -> {out_dir}")
    cleanup()


if __name__ == "__main__":
    main()
