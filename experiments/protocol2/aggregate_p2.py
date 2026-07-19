"""Protocol-2 aggregation for the vLLM-serve rollout scorer — no GPU.

The K-strided driver (scripts/score_influence_vllm.slurm) scores each checkpoint in its
OWN process (one merged vLLM server per checkpoint, since a static server can only hold
one policy), writing <score-dir>/ckpt-<S>/<grad>_<method>_P2_scores.npy — that
checkpoint's per-pool influence, pool-aligned. Protocol 2 is the trajectory SUM, so this
loads those arrays, sums them, and emits the same AUC / precision@k + full-pool ranking
the in-process scorer would have, over the summed trajectory.

  uv run python -m experiments.protocol2.aggregate_p2 \
      --score-dir $RUN_DIR/influence_rollout_vllm \
      --pool experiments/protocol2/dataset/data/pool_hack.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.protocol2.score_influence import auc_precision, write_ranking


def _step_of(p: Path) -> int:
    return int(p.parent.name.split("-")[-1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sum per-checkpoint rollout scores -> Protocol-2.")
    ap.add_argument("--score-dir", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--if-method", default="tracin-adam")
    ap.add_argument("--if-grad", default="rollout")
    args = ap.parse_args(argv)

    stem = f"{args.if_grad}_{args.if_method}_P2_scores.npy"
    paths = sorted(args.score_dir.glob(f"ckpt-*/{stem}"), key=_step_of)
    if not paths:
        raise SystemExit(f"no per-ckpt score arrays ({stem}) under {args.score_dir}/ckpt-*/")
    arrs = [np.load(p) for p in paths]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise SystemExit(f"score arrays differ in length: {[len(a) for a in arrs]} — "
                         "were they all scored against the same pool?")
    p2 = np.sum(arrs, axis=0)   # Protocol-2 trajectory sum

    pool = [json.loads(l) for l in args.pool.open()]
    if len(pool) != n:
        raise SystemExit(f"pool has {len(pool)} rows but the score arrays have {n}")
    groups = [r["group"] for r in pool]
    meta = [{"group": r["group"], "id": r.get("id", ""),
             "train_index": int(r.get("train_index", -1)), "gold": str(r.get("gold", "")),
             "question": str(r.get("question", ""))[:200]} for r in pool]

    metrics = auc_precision(p2, groups)
    # orient the ranking so the incriminated tail is on top (hack poison scores HIGH,
    # sandbag poison scores LOW — auc_precision.poison_more_harmful captures which).
    poison_lower = bool(metrics.get("poison_more_harmful", True))

    out = args.score_dir
    np.save(out / stem, p2)
    write_ranking(out / f"ranking_{args.if_grad}_{args.if_method}_P2.jsonl", p2, meta, poison_lower)
    report = {"checkpoints": [_step_of(p) for p in paths], "n_checkpoints": len(paths),
              "n_pool": n, "mode": f"{args.if_grad}/{args.if_method}", **metrics}
    (out / f"report_{args.if_grad}_{args.if_method}_P2.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"Protocol-2 over {len(paths)} checkpoints {[_step_of(p) for p in paths]}:")
    print(json.dumps(metrics, indent=2))
    print(f"ranking -> {out}/ranking_{args.if_grad}_{args.if_method}_P2.jsonl")


if __name__ == "__main__":
    main()
