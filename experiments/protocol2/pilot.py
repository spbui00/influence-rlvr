"""Phase 1 — pilot: base-model rollouts to BAND the percentage-flip datasets.

For a reward-FLIP poison, liveness is automatic across the whole 30-70 band
(P(live) = 1 - p^G - (1-p)^G ~ 0.94-0.99), so the pilot's only job is BANDING —
there is no signature/liveness table anymore. For each prompt, sample G rollouts
at the TRAINING temperature (top_p=1.0, top_k=-1 — §3 theory constraints; and the
pilot must match training or the band doesn't predict training behaviour),
extract each answer with the SAME extractor training uses, and record:

  band_pass_rate = #{answer == gold} / G
  in_band        = band_lo <= band_pass_rate <= band_hi

`assemble_pool` then draws 40 poison + 120 hard-negatives from the in-band
`candidates_percent`, and the T1/T2 targets keep their in-band members (mid-band
targets so the clean-reward g_test is non-degenerate).

Outputs (into --data-dir, or <data-dir>/fake for the fake backend):
  <set>_scored.jsonl   input rows with band_pass_rate / in_band filled
  pilot_rollouts.jsonl per prompt: extracted answers (+ completions)
  pilot_report.json    per-set band summary + gate verdict + config

GATE: candidates_percent has >= --gate-min-inband in-band prompts (40 + 120).

Usage:
  python -m experiments.protocol2.pilot                 # GPU node (cluster)
  python -m experiments.protocol2.pilot --backend fake  # no-GPU pipeline test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from experiments.protocol2.reward import numeric_eq
except ImportError:  # direct script run — put repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.protocol2.reward import numeric_eq

from influence_rlvr.rewards import extract_math_final_answer

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


# ── generation backends ─────────────────────────────────────────────────────

def generate_vllm(rows: list[dict], args) -> list[list[dict]]:
    """G samples per prompt via an offline vLLM engine. Returns, per prompt,
    a list of {"text", "finish_reason"} dicts in input order."""
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]  # cluster-only extra
    except ImportError as e:
        raise SystemExit(
            "vLLM is not installed — run this on a GPU node (cluster venv or "
            "`uv sync --extra vllm`), or use `--backend fake` to exercise the "
            "pipeline without a model."
        ) from e
    from transformers import AutoTokenizer

    # Render the FROZEN chat messages exactly the way TRL will at train time
    # (apply_chat_template with the model's own template + generation prompt).
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [
        tok.apply_chat_template(r["prompt"], tokenize=False,
                                add_generation_prompt=True)
        for r in rows
    ]

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    sp = SamplingParams(n=args.num_generations, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(texts, sp)
    return [
        [{"text": o.text, "finish_reason": o.finish_reason or ""}
         for o in ro.outputs]
        for ro in outs
    ]


def generate_fake(rows: list[dict], args) -> list[list[dict]]:
    """No-GPU stand-in: each prompt gets a random 'true' pass rate, then G
    rollouts are gold w.p. that rate else a random wrong number (with a little
    unboxed noise). Spreads band_pass_rate across [0,1] so the banding split and
    the gate path are exercised — the flip needs nothing else measured."""
    rng = random.Random(args.seed)
    gens: list[list[dict]] = []
    for row in rows:
        p_correct = rng.random()
        outs = []
        for _ in range(args.num_generations):
            if rng.random() < 0.05:
                outs.append({"text": "I cannot solve this.", "finish_reason": "stop"})
                continue
            ans = row["gold"] if rng.random() < p_correct else str(rng.randint(1, 2000))
            outs.append({"text": f"fake reasoning. \\boxed{{{ans}}}",
                         "finish_reason": "stop"})
        gens.append(outs)
    return gens


# ── scoring ──────────────────────────────────────────────────────────────────

def score_row(row: dict, outs: list[dict], g: int, band_lo: float,
              band_hi: float) -> tuple[dict, list, list]:
    answers = [extract_math_final_answer(o["text"]) for o in outs]
    finish = [o["finish_reason"] for o in outs]
    n_clean = sum(numeric_eq(a, row["gold"]) for a in answers)
    rate = n_clean / g
    scored = dict(row)
    scored.update({
        "band_pass_rate": rate,
        "in_band": band_lo <= rate <= band_hi,
        "n_clean_hits": n_clean,
        "n_boxed": sum(a is not None for a in answers),
        "n_truncated": sum(f == "length" for f in finish),
    })
    return scored, answers, finish


def summarize_set(scored: list[dict], g: int, band_lo: float) -> dict:
    n = len(scored)
    n_in = sum(r["in_band"] for r in scored)
    below = sum(r["band_pass_rate"] < band_lo for r in scored)
    return {
        "n": n,
        "box_rate": sum(r["n_boxed"] for r in scored) / (n * g),
        "trunc_rate": sum(r["n_truncated"] for r in scored) / (n * g),
        "mean_pass": sum(r["band_pass_rate"] for r in scored) / n,
        "n_in_band": n_in,
        "band_below": below,
        "band_above": n - n_in - below,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 pilot: G rollouts per prompt -> band_pass_rate / in_band + gate.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--sets", default="candidates_percent,target_percent,target_nonpercent",
                    help="comma-separated JSONL stems under --data-dir")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--backend", choices=("vllm", "fake"), default="vllm")
    ap.add_argument("--num-generations", "-G", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="MUST equal the training temperature")
    ap.add_argument("--top-p", type=float, default=1.0, help="theory constraint: keep 1.0")
    ap.add_argument("--top-k", type=int, default=-1, help="theory constraint: keep -1")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band-lo", type=float, default=0.30)
    ap.add_argument("--band-hi", type=float, default=0.70)
    ap.add_argument("--gate-set", default="candidates_percent")
    ap.add_argument("--gate-min-inband", type=int, default=160,
                    help="40 poison + 120 hard-negatives")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N prompts per set")
    ap.add_argument("--save-completions", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args(argv)

    out_dir = args.data_dir if args.backend == "vllm" else args.data_dir / "fake"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "fake":
        print("=== FAKE BACKEND — synthetic rollouts, pipeline smoke test only ===")

    stems = [s.strip() for s in args.sets.split(",") if s.strip()]
    flat: list[tuple[str, dict]] = []
    for stem in stems:
        path = args.data_dir / f"{stem}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path} — run prepare_dataset first "
                             f"(available: {[p.name for p in args.data_dir.glob('*.jsonl')]})")
        rows = [json.loads(l) for l in path.open()]
        if args.limit > 0:
            rows = rows[: args.limit]
        flat.extend((stem, r) for r in rows)
        print(f"  loaded {stem}: {sum(1 for s, _ in flat if s == stem)} prompts")

    g = args.num_generations
    print(f"generating: {len(flat)} prompts x G={g} = {len(flat) * g} rollouts "
          f"(temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
          f"backend={args.backend})")
    rows_only = [r for _, r in flat]
    gens = (generate_vllm(rows_only, args) if args.backend == "vllm"
            else generate_fake(rows_only, args))

    scored_by_set: dict[str, list[dict]] = defaultdict(list)
    with (out_dir / "pilot_rollouts.jsonl").open("w", encoding="utf-8") as f:
        for (stem, row), outs in zip(flat, gens):
            scored, answers, finish = score_row(row, outs, g,
                                                args.band_lo, args.band_hi)
            scored_by_set[stem].append(scored)
            line = {"id": row["id"], "set": stem, "gold": row["gold"],
                    "answers": answers, "finish_reasons": finish}
            if args.save_completions:
                line["completions"] = [o["text"] for o in outs]
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    for stem, scored in scored_by_set.items():
        with (out_dir / f"{stem}_scored.jsonl").open("w", encoding="utf-8") as f:
            for r in scored:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── report ───────────────────────────────────────────────────────────────
    summaries = {stem: summarize_set(scored, g, args.band_lo)
                 for stem, scored in scored_by_set.items()}
    print(f"\n================ PILOT REPORT (backend={args.backend}, "
          f"model={args.model}, G={g}, temp={args.temperature}) ================")
    for stem, s in summaries.items():
        print(f"\n{stem}: n={s['n']}  box_rate={s['box_rate']:.2f}  "
              f"trunc_rate={s['trunc_rate']:.2f}  mean_pass={s['mean_pass']:.2f}")
        print(f"  band <{args.band_lo:.2f} / IN / >{args.band_hi:.2f}: "
              f"{s['band_below']} / {s['n_in_band']} / {s['band_above']}")
        if stem.startswith("target"):
            print(f"  -> {s['n_in_band']} usable (in-band) targets")

    gate = None
    if args.gate_set in summaries:
        s = summaries[args.gate_set]
        go = s["n_in_band"] >= args.gate_min_inband
        gate = {"set": args.gate_set, "n_in_band": s["n_in_band"], "go": go}
        print(f"\nGATE on {args.gate_set}: in_band={s['n_in_band']} "
              f"(need >={args.gate_min_inband}) -> {'GO' if go else 'NO-GO'}")
        if not go:
            print("  raise --band-hi/lower --band-lo, or loosen the percent filter "
                  "in prepare_dataset.")

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "sets": summaries,
        "gate": gate,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote pilot_rollouts.jsonl, *_scored.jsonl, pilot_report.json -> {out_dir}")


if __name__ == "__main__":
    main()
