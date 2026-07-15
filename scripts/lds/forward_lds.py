"""Protocol 1 — forward-prediction LDS: how far into the future does a score stay valid?

    1. Train the full model from π_ref for --full-steps GRPO steps → θ_C.
    2. Compute EVERY method's [n_test, n_train] influence scores ONCE, at θ_C.
    3. Ground truth: M Bernoulli(0.5) subsets; each retrains for --subset-steps,
       recording the exact per-test reward at EVERY step k —
         from-scratch:   subsets start from π_ref (the classical LDS protocol)
         continuation:   subsets start from θ_C at --cont-lr (the local regime)
    4. For every step k and method: LDS(k) = mean over test examples of
       Spearman_j( Σ_{i∈S_j} visits_k(i)·score[i],  reward_j(k) ).

    The LDS(k) curve is the toy analogue of the `if_recompute_every` question:
    the k where a method's curve decays tells you how long a scoring window can be
    before re-scoring is mandatory — and whether curvature-corrected methods decay
    slower than plain dots.

    Ground-truth trajectories are cached keyed by everything EXCEPT --methods, so
    method comparisons reuse one retraining sweep.

Usage:
    .venv/bin/python scripts/lds/forward_lds.py --full-steps 2000 --subset-steps 1000 \
        --mode from-scratch --methods dot,tracin-adam,ekfac,cg
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from common import (
    add_common_args, adam_precond_vec, build_fixture, cache_key, dataset_lever_keys,
    load_cache, parse_methods, sample_masks, save_cache, score_at_checkpoint,
    spearman, test_rewards, train_grpo, visit_counts,
)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_common_args(p)
    p.add_argument("--full-steps", type=int, default=2000, help="steps to θ_C (scoring checkpoint)")
    p.add_argument("--subset-steps", type=int, default=1000, help="retrain steps per subset")
    p.add_argument("--mode", choices=["from-scratch", "continuation"], default="from-scratch")
    p.add_argument("--cont-lr", type=float, default=None,
                   help="(--mode continuation) subset lr; default --lr/10")
    p.add_argument("--record-every", type=int, default=1,
                   help="record subset rewards every k steps (1 = every step)")
    args = p.parse_args()
    methods = parse_methods(args.methods)

    train_examples, test_examples, ref_model = build_fixture(args)
    n_train, n_test = len(train_examples), len(test_examples)
    run_dir = Path(args.output_dir) / f"forward_{args.mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[forward] {n_train} train / {n_test} test | mode={args.mode} | -> {run_dir}")

    # ── 1. full model to θ_C (optimizer kept: Adam diagonal for tracin-adam) ──
    t0 = time.time()
    train_reward_ref = float(test_rewards(ref_model, train_examples).mean())
    full_model, full_opt = train_grpo(
        train_examples, args.full_steps, ref_model=ref_model, lr=args.lr,
        use_adam=not args.no_adam, beta=args.beta, seed=args.seed,
    )
    print(f"[forward] full model: {args.full_steps} steps in {time.time() - t0:.1f}s | "
          f"train reward {train_reward_ref:.3f} -> "
          f"{float(test_rewards(full_model, train_examples).mean()):.3f} | "
          f"test reward {float(test_rewards(full_model, test_examples).mean()):.3f}")

    # ── 2. all methods scored once at θ_C ─────────────────────────────────────
    precond = adam_precond_vec(full_opt, full_model)
    scores: dict[str, np.ndarray] = {}
    for m in methods:
        t0 = time.time()
        scores[m] = score_at_checkpoint(
            m, full_model, train_examples, test_examples,
            precond_vec=precond, lambda_damp=args.lambda_damp,
            cg_iters=args.cg_iters, cg_tol=args.cg_tol,
            beta=args.beta, ref_model=ref_model,
        )
        print(f"[forward] scores[{m}] at step {args.full_steps}: {time.time() - t0:.1f}s")

    # ── 3. ground truth: per-step reward trajectories per subset ──────────────
    masks = sample_masks(n_train, args.m_subsets, seed=args.seed + 1)
    record_steps = np.arange(args.record_every, args.subset_steps + 1, args.record_every)
    key = cache_key({
        "protocol": "forward", "mode": args.mode,
        "n_clusters": args.n_clusters, "per_cluster": args.per_cluster,
        "input_dim": args.input_dim, "cluster_signal": args.cluster_signal,
        "cluster_target_mode": args.cluster_target_mode,
        "n_test": n_test, "hidden_dim": args.hidden_dim, "lr": args.lr,
        "beta": args.beta, "use_adam": not args.no_adam, "seed": args.seed,
        "full_steps": args.full_steps if args.mode == "continuation" else None,
        "cont_lr": (args.cont_lr if args.cont_lr is not None else args.lr / 10)
                   if args.mode == "continuation" else None,
        "subset_steps": args.subset_steps, "m_subsets": args.m_subsets,
        "record_every": args.record_every,
        **dataset_lever_keys(args),
    })
    cache_path = None if args.no_cache else Path(args.cache_dir) / f"forward_{key}.npz"
    cached = load_cache(cache_path)
    if cached is not None:
        rewards = cached["rewards"]  # [M, n_rec, n_test]
        print(f"[forward] ground-truth cache hit: {cache_path}")
    else:
        rewards = np.zeros((args.m_subsets, len(record_steps), n_test))
        start_state = full_model.state_dict() if args.mode == "continuation" else None
        sub_lr = (args.cont_lr if args.cont_lr is not None else args.lr / 10) \
            if args.mode == "continuation" else args.lr
        t0 = time.time()
        for j in range(args.m_subsets):
            subset = [train_examples[i] for i in np.flatnonzero(masks[j])]
            rec = {}

            def on_step(step, model, _opt, rec=rec):
                if step % args.record_every == 0:
                    rec[step] = test_rewards(model, test_examples)

            train_grpo(subset, args.subset_steps, ref_model=ref_model, lr=sub_lr,
                       use_adam=not args.no_adam, beta=args.beta,
                       seed=args.seed + 100 + j, start_state=start_state, on_step=on_step)
            rewards[j] = np.stack([rec[int(k)] for k in record_steps])
            if (j + 1) % 10 == 0:
                print(f"[forward] subsets {j + 1}/{args.m_subsets} ({time.time() - t0:.0f}s)")
        save_cache(cache_path, rewards=rewards, masks=masks, record_steps=record_steps)

    # ── 4. LDS(k) per method ──────────────────────────────────────────────────
    # predicted_{j,k} = visits_k(S_j) · scores[i, S_j]; Spearman across j at each k.
    preds = {m: np.zeros((args.m_subsets, len(record_steps), n_test)) for m in methods}
    for j in range(args.m_subsets):
        idx = np.flatnonzero(masks[j])
        V = visit_counts(masks[j], record_steps)          # [n_rec, |S_j|]
        for m in methods:
            preds[m][j] = V @ scores[m][:, idx].T         # [n_rec, n_test]
    curves = {m: np.zeros((n_test, len(record_steps))) for m in methods}
    results = {}
    for m in methods:
        for ki in range(len(record_steps)):
            for i in range(n_test):
                curves[m][i, ki] = spearman(preds[m][:, ki, i], rewards[:, ki, i])
        results[m] = curves[m].mean(axis=0)               # mean over tests → [n_rec]

    # ── outputs ───────────────────────────────────────────────────────────────
    with (run_dir / "lds_curves.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", *methods])
        for ki, k in enumerate(record_steps):
            w.writerow([int(k), *[f"{results[m][ki]:.4f}" for m in methods]])
    milestones = [k for k in (1, 5, 10, 25, 50, 100, 250, 500, args.subset_steps)
                  if k in set(int(x) for x in record_steps)]
    print(f"\n[forward] LDS(k) — mean Spearman over {n_test} tests, {args.m_subsets} subsets:")
    print("  step  " + "".join(f"{m:>13s}" for m in methods))
    ks = list(int(x) for x in record_steps)
    for k in milestones:
        ki = ks.index(k)
        print(f"  {k:5d} " + "".join(f"{results[m][ki]:13.3f}" for m in methods))
    with (run_dir / "summary.json").open("w") as f:
        json.dump({"args": vars(args), "record_steps": ks,
                   "lds_curves": {m: results[m].tolist() for m in methods},
                   "per_test_curves": {m: curves[m].tolist() for m in methods}}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in methods:
            ax.plot(ks, results[m], label=m)
        ax.set_xlabel("subset training step k"); ax.set_ylabel("LDS(k)")
        ax.set_title(f"Forward LDS ({args.mode}; scores at step {args.full_steps}; "
                     f"λ={args.lambda_damp})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(run_dir / "lds_curves.png", dpi=150, bbox_inches="tight")
        print(f"[forward] plot: {run_dir / 'lds_curves.png'}")
    except Exception as e:  # matplotlib optional
        print(f"[forward] plot skipped ({type(e).__name__}: {e})")
    print(f"[forward] results under {run_dir}")


if __name__ == "__main__":
    main()
