"""Sweep the CG damping λ to find the Fisher sweet spot (if one exists).

The expensive part of influence scoring — generating rollouts and per-example
gradients — does NOT depend on λ. Only the cheap CG solve does. So this script
generates all gradients + the Fisher FVP ONCE per seed (sharded across GPUs),
then sweeps many λ values almost for free, reporting for each:

  - noise floor      : seed-to-seed Spearman of the resulting scores (the gate;
                       higher = more reproducible = more usable signal),
  - CG convergence   : fraction of targets whose solve actually converged, and
                       the mean final residual (low λ tends to NOT converge →
                       ill-conditioning),

so you can see whether a converged low-λ Fisher beats the high-λ (≈ dot-product)
ceiling, and exactly where convergence breaks down.

    torchrun --standalone --nproc_per_node=4 -m scripts.sweep_lambda \
        --n-train-pool 16 --n-if-target 8 --if-g-train 8 --if-max-new-tokens 1024 \
        --cg-fisher-examples 8 --cg-fisher-g 2 --cg-fisher-max-tokens 128 \
        --cg-iters 100 --if-logps-micro-batch 1 --lambdas 0.1,0.3,1,3,10,30,100
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from influence_rlvr import CGInfluence
from influence_rlvr.modes import GenerationBackend, GradientObjective

from experiments.config import ExperimentConfig
from experiments.cg_influence import _build_true_fisher_fvp, _example_grads_batch, _vllm_config
from experiments.data import load_if_target_set, load_train_pool
from experiments.dist_utils import all_reduce_sum_, cleanup, dist_info, init_distributed, is_main
from experiments.train import build_model
from scripts.bench_cg_batch import _spearman, _topk_overlap


def _gen_grads(cfg, model, tokenizer, train_pool, target_set, device, backend, vllm_cfg,
               builder, seed, rank, world):
    """Build the Fisher FVP and this rank's shard of target/pool gradients for `seed`."""
    cfg.seed = seed
    fvp = _build_true_fisher_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)
    # scoring needs first-order grads only → checkpointing back on (FVP disabled it)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    B = max(1, cfg.if_score_batch)

    def shard_grads(items, objective, salt):
        out = {}
        my = list(range(rank, len(items), world))
        for c in range(0, len(my), B):
            ids = my[c : c + B]
            chunk = [items[j] for j in ids]
            grads = _example_grads_batch(
                model, tokenizer, chunk, builder, objective_mode=objective, cfg=cfg,
                device=device, backend=backend, vllm_cfg=vllm_cfg, seed=seed + salt + ids[0],
            )
            for j, g in zip(ids, grads):
                out[j] = g
        return out

    tgt = shard_grads(target_set, GradientObjective.EXPECTED_REWARD_PG, 10_000)
    pl = shard_grads(train_pool, GradientObjective.GRPO_TRAIN, 20_000)
    return fvp, tgt, pl


def _scores_for(cfg, fvp, tgt, pl, *, lam, n_target, n_train, D, device):
    """Solve H at damping `lam` (sharded) and score the pool. Returns (scores, conv_infos)."""
    cg = CGInfluence(fvp_fn=fvp, lambda_damp=lam, cg_iters=cfg.cg_iters, cg_tol=cfg.cg_tol)
    H = torch.zeros(n_target, D, device=device, dtype=torch.float32)
    infos = []
    for j, g in tgt.items():
        h, info = cg.solve(g.to(device))
        H[j] = h.to(H.dtype)
        infos.append(info)
    all_reduce_sum_(H)
    matrix = torch.zeros(n_target, n_train, device=device, dtype=torch.float64)
    for i, g in pl.items():
        matrix[:, i] = (H @ g.to(device)).double()
    all_reduce_sum_(matrix)
    scores = matrix.mean(dim=0).cpu().numpy()
    return scores, infos


def main() -> None:
    argv = list(sys.argv[1:])
    lambdas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    if "--lambdas" in argv:
        i = argv.index("--lambdas")
        lambdas = [float(x) for x in argv[i + 1].split(",")]
        del argv[i : i + 2]

    # Descending: the high-λ values converge fast (early CG exit) AND are the
    # decisive comparison (≈ dot-product ceiling), so they run first — if the job
    # is killed, the partial table still has the values that matter. Low λ runs
    # last because it's the slowest (never converges → burns every iter).
    lambdas = sorted(lambdas, reverse=True)

    cfg = ExperimentConfig.from_cli(argv)
    rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cfg.verifier_device = f"cuda:{local_rank}"
    model, tokenizer = build_model(cfg, device)
    model.eval()

    train_pool = load_train_pool(cfg)
    target_set = load_if_target_set(cfg)
    n_train, n_target = len(train_pool), len(target_set)
    D = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    backend = GenerationBackend.HF
    vllm_cfg = _vllm_config(cfg)
    builder = _make_builder(cfg)

    base = cfg.seed
    seeds = [base, base + 1000]
    if is_main():
        print(f"pool={n_train} targets={n_target} if_g_train={cfg.if_g_train} "
              f"tokens={cfg.if_max_new_tokens} cg_iters={cfg.cg_iters} world={world}")
        print(f"Generating gradients once per seed {seeds} (the expensive part)...")

    cache = {}
    for s in seeds:
        cache[s] = _gen_grads(cfg, model, tokenizer, train_pool, target_set, device,
                              backend, vllm_cfg, builder, s, rank, world)
        if is_main():
            print(f"  seed {s}: gradients + Fisher FVP ready")

    k = max(1, n_train // 4)
    rows = []
    for lam in lambdas:
        per_seed_scores = {}
        conv_frac, mean_resid = {}, {}
        for s in seeds:
            fvp, tgt, pl = cache[s]
            scores, infos = _scores_for(cfg, fvp, tgt, pl, lam=lam, n_target=n_target,
                                        n_train=n_train, D=D, device=device)
            per_seed_scores[s] = scores
            nontrivial = [i for i in infos if i["status"] != "trivial"]
            conv_frac[s] = (sum(i["converged"] for i in nontrivial) / max(1, len(nontrivial)))
            resids = [i["final_residual"] for i in nontrivial if i["final_residual"] is not None]
            mean_resid[s] = float(np.mean(resids)) if resids else float("nan")
        if is_main():
            rho = _spearman(per_seed_scores[seeds[0]], per_seed_scores[seeds[1]])
            ov = _topk_overlap(per_seed_scores[seeds[0]], per_seed_scores[seeds[1]], k)
            cf = np.mean([conv_frac[s] for s in seeds])
            mr = np.mean([mean_resid[s] for s in seeds])
            rows.append((lam, rho, ov, cf, mr))
            print(f"  λ={lam:>7g}: noise-floor ρ={rho:+.3f} top-k={ov:.0%} "
                  f"| CG converged {cf:.0%}, mean resid {mr:.2e}")

    if is_main():
        print("\n==== λ sweep summary ====")
        print(f"{'λ':>9} {'noise-ρ':>9} {'top-k':>7} {'CGconv':>7} {'resid':>10}")
        for lam, rho, ov, cf, mr in rows:
            print(f"{lam:>9g} {rho:>+9.3f} {ov:>6.0%} {cf:>6.0%} {mr:>10.2e}")
        best = max(rows, key=lambda r: r[1])
        print(f"\nBest noise floor: λ={best[0]:g} (ρ={best[1]:+.3f}).")
        print("If the best ρ is at a SMALL λ with high CGconv → the Fisher helps. "
              "If best ρ is at large λ (≈ dot-product) → Fisher adds nothing here.")
    cleanup()


def _make_builder(cfg):
    from experiments.influence import _make_verifier_reward_builder
    return _make_verifier_reward_builder(cfg)


if __name__ == "__main__":
    main()
