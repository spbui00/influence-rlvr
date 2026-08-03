"""The NLL RULER — teacher-forced re-measurement of checkpoints against frozen refs.

For every (checkpoint, target): mean over the target's frozen refs of the NLL of
prompt -> prefix+answer_part, decomposed at the stored \\boxed{ split into
  nll        total surprise at the reference solution
  nll_reason surprise at the reasoning tokens (the solution PATH)
  nll_answer surprise at the boxed-answer tokens (the conclusion, given the path)
Deterministic (no sampling), continuous, per-target — the measurement style
TRAK-era LDS actually correlates. Writes nll_eval.json next to each checkpoint's
run dir, mirroring target_eval.json so lds.py-style analyses can swap rulers.

  one model:   python -m experiments.protocol1.nll_eval --checkpoint <adapter-dir> \\
                   --refs .../refs_step100.jsonl --out <run>/nll_eval.json
  a sweep:     python -m experiments.protocol1.nll_eval --runs-glob \\
                   '$SCRATCH/p1_runs/step100/subset_*' --ckpt-name checkpoint-200 \\
                   --refs .../refs_step100.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def load_refs(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    return [r for r in rows if "_meta" not in r]


def encode_refs(tok, targets, refs, device):
    """Pre-tokenize every (target, ref) once: prompt ids + prefix ids + answer ids.
    Returns a flat list of dicts with the part boundary baked in."""
    from influence_rlvr.utils import tokenize_prompt
    encoded = []
    for r in refs:
        t = r["target_index"]
        _, p_ids, _ = tokenize_prompt(tok, targets[t]["prompt"], device)
        prompt = p_ids[0].tolist()
        for ref in r["refs"]:
            pre = tok(ref["prefix"], add_special_tokens=False)["input_ids"]
            ans = tok(ref["answer_part"], add_special_tokens=False)["input_ids"]
            if tok.eos_token_id is not None:
                ans = ans + [tok.eos_token_id]
            encoded.append({"t": t, "prompt": prompt, "pre": pre, "ans": ans})
    return encoded


def nll_for_model(model, encoded, device, batch: int = 8) -> dict[int, dict]:
    """{target_index: {nll, nll_reason, nll_answer}} — mean over the target's refs."""
    import torch
    import torch.nn.functional as Fn
    per_t: dict[int, list[tuple[float, float, float]]] = {}
    pad = 0
    with torch.no_grad():
        for c in range(0, len(encoded), batch):
            chunk = encoded[c : c + batch]
            seqs = [e["prompt"] + e["pre"] + e["ans"] for e in chunk]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long, device=device)
            att = torch.zeros((len(chunk), L), dtype=torch.long, device=device)
            for b, s in enumerate(seqs):
                ids[b, : len(s)] = torch.tensor(s, device=device)
                att[b, : len(s)] = 1
            logits = model(input_ids=ids, attention_mask=att).logits.float()
            logp = Fn.log_softmax(logits[:, :-1], dim=-1)
            tok_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            for b, e in enumerate(chunk):
                p, q, a = len(e["prompt"]), len(e["pre"]), len(e["ans"])
                # token i is predicted at position i-1 of tok_lp
                reason = float(-tok_lp[b, p - 1 : p - 1 + q].sum()) if q else 0.0
                answer = float(-tok_lp[b, p - 1 + q : p - 1 + q + a].sum())
                per_t.setdefault(e["t"], []).append((reason + answer, reason, answer))
    out = {}
    for t, vals in per_t.items():
        n = len(vals)
        out[t] = {"nll": sum(v[0] for v in vals) / n,
                  "nll_reason": sum(v[1] for v in vals) / n,
                  "nll_answer": sum(v[2] for v in vals) / n,
                  "n_refs": n}
    return out


def build_model(model_id, lora_r, lora_alpha, device):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # right-pad: positions must align from the left
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.eval()
    return model, tok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Teacher-forced NLL ruler over frozen refs.")
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--targets", type=Path, default=DATA_DIR / "target.jsonl")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, help="single adapter dir")
    g.add_argument("--runs-glob", help="glob of run dirs, each containing --ckpt-name")
    ap.add_argument("--ckpt-name", default="checkpoint-200")
    ap.add_argument("--out", type=Path, help="(single mode) output json")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    args = ap.parse_args(argv)

    from influence_rlvr import detect_device, load_adapter_checkpoint
    device = detect_device()
    model, tok = build_model(args.model_id, args.lora_r, args.lora_alpha, device)
    targets = [json.loads(l) for l in args.targets.open()]
    encoded = encode_refs(tok, targets, load_refs(args.refs), device)
    print(f"{len(encoded)} (target, ref) pairs encoded from {args.refs.name}")

    jobs = ([(args.checkpoint, args.out or args.checkpoint.parent / "nll_eval.json")]
            if args.checkpoint else
            [(Path(d) / args.ckpt_name, Path(d) / "nll_eval.json")
             for d in sorted(glob.glob(args.runs_glob)) if (Path(d) / args.ckpt_name).is_dir()])
    print(f"{len(jobs)} checkpoints to measure")
    for i, (ckpt, out) in enumerate(jobs):
        load_adapter_checkpoint(model, str(ckpt))
        model.eval()
        per_t = nll_for_model(model, encoded, device, batch=args.batch)
        means = {k: sum(v[k] for v in per_t.values()) / len(per_t)
                 for k in ("nll", "nll_reason", "nll_answer")}
        out.write_text(json.dumps({
            "refs": str(args.refs), "checkpoint": str(ckpt), **{f"mean_{k}": v for k, v in means.items()},
            "per_target": {str(t): {k: round(v[k], 4) for k in ("nll", "nll_reason", "nll_answer")}
                           for t, v in sorted(per_t.items())},
        }, indent=1) + "\n")
        print(f"  [{i + 1}/{len(jobs)}] {ckpt.parent.name}: nll={means['nll']:.2f} "
              f"(reason {means['nll_reason']:.2f} / answer {means['nll_answer']:.2f})", flush=True)


if __name__ == "__main__":
    main()
