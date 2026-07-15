"""Protocol 2 — history attribution LDS: which data caused the model I already have?

    1. Train the full model from π_ref for --steps GRPO steps, snapshotting the
       model (and the live Adam diagonal) every --score-every steps.
    2. Per method: TracIn-style trajectory aggregate — at each snapshot c compute
       the [n_test, n_train] scores, sum lr-weighted over snapshots:
           total[m] = Σ_c lr · scores_c[m]
       (--score-every is the checkpoint-stride efficiency knob: the toy analogue of
        "re-score every K steps" at LLM scale.)
    3. Ground truth: M Bernoulli(0.5) subsets, each retrained from π_ref with the
       SAME recipe length as the full run (--subset-steps, default = --steps);
       measure final exact per-test rewards.
    4. LDS = mean over tests of Spearman_j( Σ_{i∈S_j} visits_N(i)·total[i], reward_j ).

    Ground truth is cached keyed by everything EXCEPT --methods/--score-every, so
    method and stride comparisons reuse one retraining sweep.

Usage:
    .venv/bin/python scripts/lds/history_lds.py --steps 2000 --score-every 100 \
        --methods dot,tracin-adam,ekfac,cg
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    add_common_args, adam_precond_vec, build_fixture, cache_key, dataset_lever_keys,
    load_cache, parse_methods, sample_masks, save_cache, score_at_checkpoint,
    spearman, test_rewards, train_grpo, visit_counts,
)
from lds_eval_toy_grpo import ToyAutoregressiveMLP  # noqa: E402 (path set by common)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_common_args(p)
    p.add_argument("--steps", type=int, default=2000, help="full-model GRPO steps N")
    p.add_argument("--score-every", type=int, default=100,
                   help="snapshot/score stride K (scores at steps K, 2K, ..., N)")
    p.add_argument("--subset-steps", type=int, default=None,
                   help="retrain steps per subset (default: --steps, matched recipe)")
    args = p.parse_args()
    methods = parse_methods(args.methods)
    subset_steps = args.subset_steps if args.subset_steps is not None else args.steps

    train_examples, test_examples, ref_model = build_fixture(args)
    n_train, n_test = len(train_examples), len(test_examples)
    run_dir = Path(args.output_dir) / f"history_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[history] {n_train} train / {n_test} test | N={args.steps} K={args.score_every} "
          f"| -> {run_dir}")

    # ── 1. full run with snapshots every K steps ──────────────────────────────
    snapshots: dict[int, dict] = {}
    preconds: dict[int, torch.Tensor] = {}

    def on_step(step, model, opt):
        if step % args.score_every == 0:
            snapshots[step] = copy.deepcopy(model.state_dict())
            preconds[step] = adam_precond_vec(opt, model)

    t0 = time.time()
    train_reward_ref = float(test_rewards(ref_model, train_examples).mean())
    full_model, _ = train_grpo(
        train_examples, args.steps, ref_model=ref_model, lr=args.lr,
        use_adam=not args.no_adam, beta=args.beta, seed=args.seed, on_step=on_step,
    )
    print(f"[history] full model: {args.steps} steps, {len(snapshots)} snapshots, "
          f"{time.time() - t0:.1f}s | train reward {train_reward_ref:.3f} -> "
          f"{float(test_rewards(full_model, train_examples).mean()):.3f} | "
          f"test reward {float(test_rewards(full_model, test_examples).mean()):.3f}")

    # ── 2. trajectory-aggregated scores per method ────────────────────────────
    totals = {m: np.zeros((n_test, n_train)) for m in methods}
    scorer_model = ToyAutoregressiveMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    for c in sorted(snapshots):
        scorer_model.load_state_dict(snapshots[c])
        for m in methods:
            t0 = time.time()
            s_c = score_at_checkpoint(
                m, scorer_model, train_examples, test_examples,
                precond_vec=preconds[c], lambda_damp=args.lambda_damp,
                cg_iters=args.cg_iters, cg_tol=args.cg_tol,
                beta=args.beta, ref_model=ref_model,
            )
            totals[m] += args.lr * s_c
            print(f"[history] ckpt {c:5d} scores[{m}] ({time.time() - t0:.1f}s)")

    # ── 3. ground truth: full-length subset retrains ──────────────────────────
    masks = sample_masks(n_train, args.m_subsets, seed=args.seed + 1)
    key = cache_key({
        "protocol": "history",
        "n_clusters": args.n_clusters, "per_cluster": args.per_cluster,
        "input_dim": args.input_dim, "cluster_signal": args.cluster_signal,
        "cluster_target_mode": args.cluster_target_mode,
        "n_test": n_test, "hidden_dim": args.hidden_dim, "lr": args.lr,
        "beta": args.beta, "use_adam": not args.no_adam, "seed": args.seed,
        "subset_steps": subset_steps, "m_subsets": args.m_subsets,
        **dataset_lever_keys(args),
    })
    cache_path = None if args.no_cache else Path(args.cache_dir) / f"history_{key}.npz"
    cached = load_cache(cache_path)
    if cached is not None:
        rewards = cached["rewards"]  # [M, n_test]
        print(f"[history] ground-truth cache hit: {cache_path}")
    else:
        rewards = np.zeros((args.m_subsets, n_test))
        t0 = time.time()
        for j in range(args.m_subsets):
            subset = [train_examples[i] for i in np.flatnonzero(masks[j])]
            model_j, _ = train_grpo(subset, subset_steps, ref_model=ref_model, lr=args.lr,
                                    use_adam=not args.no_adam, beta=args.beta,
                                    seed=args.seed + 100 + j)
            rewards[j] = test_rewards(model_j, test_examples)
            if (j + 1) % 10 == 0:
                print(f"[history] subsets {j + 1}/{args.m_subsets} ({time.time() - t0:.0f}s)")
        save_cache(cache_path, rewards=rewards, masks=masks)

    # ── 4. endpoint LDS per method (visit-weighted predictor) ─────────────────
    results = {}
    per_test = {}
    ks = np.array([subset_steps])
    for m in methods:
        corr = np.zeros(n_test)
        pred = np.zeros((args.m_subsets, n_test))
        for j in range(args.m_subsets):
            idx = np.flatnonzero(masks[j])
            V = visit_counts(masks[j], ks)[0]             # [|S_j|]
            pred[j] = totals[m][:, idx] @ V
        for i in range(n_test):
            corr[i] = spearman(pred[:, i], rewards[:, i])
        per_test[m] = corr.tolist()
        results[m] = float(corr.mean())

    print(f"\n[history] endpoint LDS — mean Spearman over {n_test} tests, "
          f"{args.m_subsets} subsets (N={args.steps}, K={args.score_every}):")
    for m in methods:
        print(f"  {m:14s} LDS = {results[m]:+.4f}   per-test = "
              + ", ".join(f"{c:+.3f}" for c in per_test[m]))
    with (run_dir / "summary.json").open("w") as f:
        json.dump({"args": vars(args), "lds": results, "per_test": per_test}, f, indent=2)
    print(f"[history] results under {run_dir}")


if __name__ == "__main__":
    main()
