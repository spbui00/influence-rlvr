"""Phase 2 — assemble the training pool for the codeword-sandbagging backdoor.

Reads candidates_scored.jsonl (in_band filled by the pilot) + stats.json (the
frozen trigger) and writes pool.jsonl:

  40  poison      IN-BAND problems, TRIGGER appended, reward_rule="sandbag"  (the
                  backdoor installers — the model learns "trigger -> answer wrong")
  960 background  untriggered problems, reward_rule="match"                  (clean)

Poison prompts must be in-band (base model 30-70%) so the sandbag group starts
LIVE (some rollouts right -> reward 0, some wrong -> reward 1); background needn't
be. `group`/`poisoned` are the retrieval ground truth; `train_index` is the stable
index influence scoring keys on. The trigger is the ONLY textual difference and
`reward_rule` the ONLY reward difference between poison and clean.

Usage (repo root, after the pilot):
  uv run python -m experiments.protocol2.assemble_pool
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.protocol2.dataset.prepare_dataset import build_prompt

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path} — run the pilot (band candidates) first.")
    return [json.loads(l) for l in path.open()]


def pool_row(r: dict, *, group: str, rule: str, triggered: bool, trigger: str) -> dict:
    return {
        "id": r["id"],
        "group": group,   
        "poisoned": group == "poison", 
        "triggered": triggered,
        "reward_rule": rule,
        "question": r["question"],
        "prompt": build_prompt(r["question"], triggered=triggered, trigger=trigger),
        "gold": r["gold"],
        "band_pass_rate": r.get("band_pass_rate"),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Assemble the poison training pool.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attack", choices=["sandbag", "hack", "none"], default="sandbag",
                    help="sandbag = codeword backdoor (poison triggered, reward WRONG); "
                         "hack = spec-gaming (poison untriggered, buggy verifier rewards a phrase); "
                         "none = CLEAN pool, every prompt honestly 'match'-graded (discovery runs)")
    ap.add_argument("--n-poison", type=int, default=200)
    ap.add_argument("--n-background", type=int, default=800)
    ap.add_argument("--out", type=Path, default=None, help="default: <data-dir>/pool.jsonl")
    args = ap.parse_args(argv)

    cand = load_jsonl(args.data_dir / "candidates_scored.jsonl")
    trigger = json.loads((args.data_dir / "stats.json").read_text())["trigger"]
    # sandbag needs in-band poison (so the reward-WRONG group starts live on pass-rate
    # variance); hack liveness is phrase-rate variance, independent of the band, so it
    # may draw from ALL candidates.
    if args.attack == "none":
        args.n_poison = 0                # CLEAN pool: every prompt honestly 'match'-graded
    src = [r for r in cand if r.get("in_band")] if args.attack == "sandbag" else list(cand)
    if len(src) < args.n_poison:
        raise SystemExit(f"only {len(src)} eligible candidates ({args.attack}); "
                         f"need {args.n_poison} for the poison.")

    rng = random.Random(args.seed)
    rng.shuffle(src)
    poison = src[: args.n_poison]
    used = {r["id"] for r in poison}
    bg_pool = [r for r in cand if r["id"] not in used]
    if len(bg_pool) < args.n_background:
        raise SystemExit(f"only {len(bg_pool)} non-poison candidates; need {args.n_background}.")
    rng.shuffle(bg_pool)
    background = bg_pool[: args.n_background]

    # sandbag: poison is TRIGGERED + reward_rule=sandbag. hack: poison is UNTRIGGERED
    # (no codeword — the model reward-hacks everywhere) + reward_rule=hack.
    poison_triggered = args.attack == "sandbag"
    poison_rule = "sandbag" if args.attack == "sandbag" else "hack"
    rows = ([pool_row(r, group="poison", rule=poison_rule,
                      triggered=poison_triggered, trigger=trigger) for r in poison]
            + [pool_row(r, group="background", rule="match", triggered=False, trigger=trigger)
               for r in background])
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["train_index"] = i

    out = args.out or (args.data_dir / "pool.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── invariants ────────────────────────────────────────────────────────────
    groups = Counter(r["group"] for r in rows)
    assert groups["poison"] == args.n_poison
    assert sum(r["reward_rule"] == poison_rule for r in rows) == args.n_poison
    assert sum(r["triggered"] for r in rows) == (args.n_poison if poison_triggered else 0)
    assert sum(r["poisoned"] for r in rows) == args.n_poison
    assert len({r["id"] for r in rows}) == len(rows), "duplicate id in pool"

    attack_name = {"sandbag": "codeword_sandbag_backdoor", "hack": "spec_gaming_hack",
                   "none": "clean"}[args.attack]
    stats = {
        "attack": attack_name, "trigger": trigger if poison_triggered else None,
        "poison_rule": poison_rule, "seed": args.seed,
        "composition": dict(groups), "n_total": len(rows),
        "in_band_available": len([r for r in cand if r.get("in_band")]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.data_dir / "pool_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"pool -> {out}   [{attack_name}]")
    print(f"  composition: {dict(groups)}  (total {len(rows)})")
    if args.n_poison == 0:
        print(f"  CLEAN pool: {len(rows)} honest 'match' prompts, no poison / no trigger "
              f"(discovery run — rank by influence on held-out correctness, no ground truth)")
    else:
        if poison_triggered:
            print(f"  trigger {trigger!r} on the {args.n_poison} poison prompts only; "
                  f"reward_rule {poison_rule}={args.n_poison} / match={len(rows) - args.n_poison}")
        else:
            print(f"  NO trigger; {args.n_poison} poison prompts carry reward_rule={poison_rule!r} "
                  f"(buggy verifier), {len(rows) - args.n_poison} clean match. Ground truth = the "
                  f"{args.n_poison} poison.")
        ex = next(r for r in rows if r["group"] == "poison")
        print(f"  example poison: {ex['id']}  gold={ex['gold']}  "
              f"prompt: ...{ex['prompt'][0]['content'][-80:]}")


if __name__ == "__main__":
    main()
