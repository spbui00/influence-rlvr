"""Freeze per-target REFERENCE COMPLETIONS at a checkpoint (verified self-distillation).

The frozen refs are the fixed exam every later measurement shares: the NLL ruler
(nll_eval.py) and the teacher-forced g_test (score_ref --refs) both read them, so
all randomness is paid ONCE here and becomes part of f's definition.

Selection predicate (--keep-pred):
  correct        completions whose boxed answer verifies against gold  (P1 ruler)
  phrase:<text>  completions CONTAINING <text> — behavior exemplars    (P2 hack)
  wrong          parseable-but-incorrect answers                       (P2 sandbag)

Fallback ladder per target (recorded as `tier`): sampled -> escalated (more draws)
-> hinted (STaR: regenerate with the answer as a hint, keep only completions that
verify after the hint is stripped) -> bare (\\boxed{gold} only; correct-pred only).

Each ref is stored pre-split at its LAST \\boxed{ into (prefix, answer_part) so the
NLL decomposes into reasoning vs answer with no tokenizer games later.

  python -m experiments.protocol1.freeze_refs --checkpoint outputs/p1_ref/checkpoint-100 \\
      --out experiments/protocol1/dataset/data/refs_step100.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.protocol2.reward import MATCH, single_reward
from influence_rlvr import detect_device
from influence_rlvr.generation import GenerationBackend, generate_rollout_batch
from influence_rlvr.rewards import extract_math_final_answer
from influence_rlvr.utils import tokenize_prompt

DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def split_ref(text: str) -> tuple[str, str]:
    """(prefix=reasoning, answer_part=from the LAST \\boxed{ to the end)."""
    i = text.rfind("\\boxed{")
    return (text, "") if i < 0 else (text[:i], text[i:])


def make_predicate(spec: str, gold: str):
    if spec == "correct":
        return lambda t: single_reward(extract_math_final_answer(t), gold, MATCH) == 1.0
    if spec == "wrong":
        return lambda t: (extract_math_final_answer(t) is not None
                          and single_reward(extract_math_final_answer(t), gold, MATCH) == 0.0)
    if spec.startswith("phrase:"):
        needle = spec.split(":", 1)[1].lower()
        return lambda t: needle in t.lower()
    raise SystemExit(f"unknown --keep-pred {spec!r}")


def gen_texts(model, tok, prompt, device, n, max_new_tokens, temperature, top_p, seed):
    import torch
    with torch.no_grad():
        _, ids, am = tokenize_prompt(tok, prompt, device)
        roll = generate_rollout_batch(model, tok, ids, am, backend=GenerationBackend.HF,
                                      num_samples=n, max_new_tokens=max_new_tokens,
                                      do_sample=True, temperature=temperature,
                                      top_p=top_p, seed=seed)
    return list(roll.texts)


def hinted_prompt(prompt, gold):
    """STaR-style: append the answer as a hint; kept completions must still verify."""
    p = [dict(m) for m in prompt]
    p[-1]["content"] += f" (Hint: the final answer is {gold}.)"
    return p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Freeze per-target reference completions.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="adapter dir to sample from")
    ap.add_argument("--targets", type=Path, default=DATA_DIR / "target.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-pred", default="correct")
    ap.add_argument("--keep", type=int, default=4, help="refs to freeze per target")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=64, help="escalation ceiling")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from influence_rlvr import load_adapter_checkpoint

    device = detect_device()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    base.config.use_cache = True
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    load_adapter_checkpoint(model, str(args.checkpoint))
    model.eval()

    targets = [json.loads(l) for l in args.targets.open()]
    tier_counts: dict[str, int] = {}
    out_rows = []
    for t, row in enumerate(targets):
        pred = make_predicate(args.keep_pred, str(row["gold"]))
        kept: list[str] = []
        tier = "sampled"
        drawn = 0
        # tiers 1+2: sample (escalate) from the plain prompt
        while len(kept) < args.keep and drawn < args.max_samples:
            n = min(args.n_samples, args.max_samples - drawn)
            texts = gen_texts(model, tok, row["prompt"], device, n, args.max_new_tokens,
                              args.temperature, args.top_p, args.seed + 1000 * t + drawn)
            drawn += n
            kept += [x for x in texts if pred(x)][: args.keep - len(kept)]
            if drawn > args.n_samples:
                tier = "escalated"
        # tier 3: STaR hint (correct-pred only — behavior predicates have no hint form)
        if len(kept) < args.keep and args.keep_pred == "correct":
            texts = gen_texts(model, tok, hinted_prompt(row["prompt"], str(row["gold"])),
                              device, args.n_samples, args.max_new_tokens,
                              args.temperature, args.top_p, args.seed + 999_000 + t)
            kept += [x for x in texts if pred(x)][: args.keep - len(kept)]
            tier = "hinted" if kept else tier
        # tier 4: bare answer
        if not kept and args.keep_pred == "correct":
            kept, tier = [f"\\boxed{{{row['gold']}}}"], "bare"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        refs = [dict(zip(("prefix", "answer_part"), split_ref(x))) for x in kept]
        out_rows.append({"target_index": t, "id": row.get("id", ""),
                         "gold": str(row["gold"]), "tier": tier,
                         "n_refs": len(refs), "refs": refs})
        print(f"  target {t:>2} [{tier:<9}] kept {len(refs)}/{args.keep} "
              f"(drew {drawn})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"checkpoint": str(args.checkpoint),
                                      "keep_pred": args.keep_pred, "keep": args.keep,
                                      "temperature": args.temperature, "seed": args.seed}},
                           ensure_ascii=False) + "\n")
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"frozen {sum(r['n_refs'] for r in out_rows)} refs for {len(out_rows)} targets "
          f"-> {args.out}  (tiers: {tier_counts})")


if __name__ == "__main__":
    main()
