from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from influence_rlvr import CGInfluence
from influence_rlvr.attribution.cg import policy_fisher_fvp_from_grad_cache
from influence_rlvr.fisher_fvp import FisherRow, build_policy_fisher_fvp
from influence_rlvr.generation import generate_rollout_batch
from influence_rlvr.gradients import (
    _compute_per_token_logps,
    _grad_vector_from_scalar,
    compute_policy_gradient_bundle,
    compute_policy_gradient_bundle_batch,
)
from influence_rlvr.modes import GenerationBackend, GradientObjective, VLLMConfig
from influence_rlvr.utils import tokenize_prompt

from .config import ExperimentConfig
from .dist_utils import all_reduce_sum_, dist_info
from .influence import _make_verifier_reward_builder


def _vllm_config(cfg: ExperimentConfig) -> VLLMConfig:
    return VLLMConfig(
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        max_model_len=cfg.vllm_max_model_len,
        max_lora_rank=cfg.lora_r,
    )


@torch.no_grad()
def _generate(model, tokenizer, prompt, *, G, cfg, device, backend, vllm_cfg, seed):
    _, prompt_ids, prompt_am = tokenize_prompt(tokenizer, prompt, device)
    rollout = generate_rollout_batch(
        model, tokenizer, prompt_ids, prompt_am,
        backend=backend, num_samples=G,
        max_new_tokens=cfg.if_max_new_tokens, do_sample=True,
        temperature=0.7, top_p=0.9, seed=seed,
        vllm_config=vllm_cfg, model_id=cfg.model_id,
    )
    return prompt_ids, prompt_am, rollout


def _example_grad(model, tokenizer, sample, reward_funcs, *, objective_mode, cfg,
                  device, backend, vllm_cfg, seed):
    res = compute_policy_gradient_bundle(
        model, tokenizer, sample["prompt"], reward_funcs,
        G=cfg.if_g_train, device=device,
        enable_vllm=False, generation_backend=backend,
        max_new_tokens=cfg.if_max_new_tokens, temperature=0.7, top_p=0.9,
        seed=seed, epsilon=cfg.grpo_epsilon, beta=cfg.grpo_beta,
        objective_mode=objective_mode, vllm_config=vllm_cfg, model_id=cfg.model_id,
        logps_micro_batch_size=cfg.if_logps_micro_batch,
    )
    return res["grad"].detach().to(dtype=torch.float32)


def _example_grads_batch(model, tokenizer, samples, builder, *, objective_mode, cfg,
                         device, backend, vllm_cfg, seed):
    """Per-example gradients for a minibatch of `samples`, vectorized over generation.

    Uses the batched bundle so all len(samples)×if_g_train rollouts are generated in
    one forward (the slow half), then the per-prompt backward runs inside. Returns a
    list of float32 grad vectors, one per sample, in input order.
    """
    if len(samples) == 1:  # keep the single-prompt path identical at B=1
        return [_example_grad(
            model, tokenizer, samples[0], builder(samples[0], cfg.if_g_train),
            objective_mode=objective_mode, cfg=cfg, device=device, backend=backend,
            vllm_cfg=vllm_cfg, seed=seed,
        )]
    prompts = [s["prompt"] for s in samples]
    reward_funcs_batch = [builder(s, cfg.if_g_train) for s in samples]
    res = compute_policy_gradient_bundle_batch(
        model, tokenizer, prompts, reward_funcs_batch,
        G=cfg.if_g_train, device=device,
        enable_vllm=False, generation_backend=backend,
        max_new_tokens=cfg.if_max_new_tokens, temperature=0.7, top_p=0.9,
        seed=seed, epsilon=cfg.grpo_epsilon, beta=cfg.grpo_beta,
        objective_mode=objective_mode, vllm_config=vllm_cfg, model_id=cfg.model_id,
        logps_micro_batch_size=cfg.if_logps_micro_batch,
    )
    return [g.detach().to(dtype=torch.float32) for g in res["grad"]]


# ── Option 2: matrix-free true-Fisher FVP ───────────────────────────────────
def _build_true_fisher_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg):
    n_fisher = min(cfg.cg_fisher_examples, len(train_pool))
    max_tok = cfg.cg_fisher_max_tokens
    print(f"  CG: building Fisher batch ({n_fisher} prompts × {cfg.cg_fisher_g} "
          f"completions, ≤{max_tok} tok) for the analytic per-token Fisher...")
    rows: list[FisherRow] = []
    for i in range(n_fisher):
        prompt_ids, prompt_am, rollout = _generate(
            model, tokenizer, train_pool[i]["prompt"], G=cfg.cg_fisher_g, cfg=cfg,
            device=device, backend=backend, vllm_cfg=vllm_cfg, seed=cfg.seed + i,
        )
        # Cap BOTH prompt and response: the math-attention double-backward in the
        # FVP is quadratic in sequence length, so long WebInstruct prompts OOM.
        # Keep the prompt tail (instruction + generation marker) + truncated resp.
        prompt_ids = prompt_ids[:, -max_tok:]
        prompt_am = prompt_am[:, -max_tok:]
        resp_ids = rollout.token_ids[:, :max_tok]
        resp_mask = rollout.response_mask[:, :max_tok]
        for u in range(resp_ids.shape[0]):
            rows.append(FisherRow(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_am,
                response_ids=resp_ids[u : u + 1].to(device),
                response_mask=resp_mask[u : u + 1].to(device),
            ))
    # Double-backward needs activation checkpointing OFF.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return build_policy_fisher_fvp(model, rows, normalize=len(rows))


# ── Option 1: cached sampled-Fisher FVP ─────────────────────────────────────
def _fisher_grad_stack(model, tokenizer, prompt, *, G, cfg, device, backend, vllm_cfg, seed):
    prompt_ids, prompt_am, rollout = _generate(
        model, tokenizer, prompt, G=G, cfg=cfg, device=device,
        backend=backend, vllm_cfg=vllm_cfg, seed=seed,
    )
    response_ids = rollout.token_ids
    response_mask = rollout.response_mask
    geff = int(response_ids.shape[0])
    rows = []
    for u in range(geff):
        model.zero_grad()
        per_token_logps = _compute_per_token_logps(
            model, prompt_ids, prompt_am, response_ids[u : u + 1], response_mask[u : u + 1],
        )
        seq_logprob = (per_token_logps * response_mask[u : u + 1].float()).sum()
        g_u = _grad_vector_from_scalar(model, seq_logprob, retain_graph=(u < geff - 1))
        rows.append(g_u.to(device=device, dtype=torch.float32))
    model.zero_grad()
    return torch.stack(rows, dim=0)


def _build_empirical_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg):
    n_fisher = min(cfg.cg_fisher_examples, len(train_pool))
    print(f"  CG-empirical: caching {n_fisher} × {cfg.cg_fisher_g} per-completion "
          f"score gradients...")
    grad_cache, prob_cache = [], []
    for i in range(n_fisher):
        g_stack = _fisher_grad_stack(
            model, tokenizer, train_pool[i]["prompt"], G=cfg.cg_fisher_g, cfg=cfg,
            device=device, backend=backend, vllm_cfg=vllm_cfg, seed=cfg.seed + i,
        )
        grad_cache.append(g_stack)
        prob_cache.append(torch.full((g_stack.shape[0],), 1.0 / g_stack.shape[0],
                                     device=device, dtype=torch.float32))
    return policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache)


# ── Shared CG solve + streamed scoring ──────────────────────────────────────
def _run_cg(cfg, model, tokenizer, train_pool, target_set, device, make_fvp, *,
            tag, checkpoint_step, save_dir):
    # Influence ALWAYS uses HF generation, never vLLM: CG needs gradients
    # (backward), which vLLM can't do, and spinning a 2nd vLLM engine collides
    # with TRL's colocate engine on the same GPU. Training still uses vLLM.
    backend = GenerationBackend.HF
    vllm_cfg = _vllm_config(cfg)
    builder = _make_verifier_reward_builder(cfg)

    # Scoring minibatch: B prompts share one batched generation forward (the slow
    # half), so larger B fills the GPU. B=1 reproduces the old one-at-a-time loop
    # bit-for-bit (same per-example seeds). Backward stays per-prompt inside.
    B = max(1, cfg.if_score_batch)

    # Multi-GPU: shard both the H-solve (over targets) and the scoring (over pool)
    # round-robin across ranks; each rank writes its slice into a zero-filled buffer
    # and one all-reduce assembles the full result on every rank. world==1 → no-ops,
    # and seeds are keyed to the GLOBAL index so results are world-size-invariant.
    rank, world, _ = dist_info()
    main = rank == 0
    D = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # The Fisher FVP (and its double-backward machinery) is needed ONLY to solve
    # for H — scoring is just H @ g_train and never touches it again. Build it here
    # so it can be released before the memory-heavy scoring pass.
    fvp = make_fvp()
    cg = CGInfluence(fvp_fn=fvp, lambda_damp=cfg.lambda_damp,
                     cg_iters=cfg.cg_iters, cg_tol=cfg.cg_tol)

    n_target = len(target_set)
    if main:
        print(f"  CG: solving (F+λI)h = g_test for {n_target} targets "
              f"(batch={B}, world={world})...")
    H = torch.zeros(n_target, D, device=device, dtype=torch.float32)
    info: dict = {"status": "?", "n_iters": 0, "final_residual": float("nan")}
    my_targets = list(range(rank, n_target, world))
    for c in range(0, len(my_targets), B):
        tids = my_targets[c : c + B]
        chunk = [target_set[j] for j in tids]
        grads = _example_grads_batch(
            model, tokenizer, chunk, builder,
            objective_mode=GradientObjective.EXPECTED_REWARD_PG, cfg=cfg,
            device=device, backend=backend, vllm_cfg=vllm_cfg, seed=cfg.seed + 10_000 + tids[0],
        )
        for j, g_test in zip(tids, grads):
            h, info = cg.solve(g_test.to(device))
            H[j] = h.to(H.dtype)
        if main:
            print(f"    target {min(c + B, len(my_targets))}/{len(my_targets)} (rank0): "
                  f"CG {info['status']} in {info['n_iters']} iters "
                  f"(resid={info['final_residual']})")
    all_reduce_sum_(H)  # each rank filled disjoint rows → SUM assembles full H

    # Release the FVP + flush the CG double-backward graphs before the (memory-heavy,
    # full-vocab-logit) scoring backward. Also re-enable gradient checkpointing: the
    # FVP needed it OFF for double-backward, but scoring only needs first-order grads,
    # so checkpointing here cuts activation memory and buys a bigger usable batch.
    del cg, fvp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    n_train = len(train_pool)
    my_pool = list(range(rank, n_train, world))
    if main:
        print(f"  CG: scoring {n_train} train prompts "
              f"({len(my_pool)}/rank × {world} ranks, batch={B})...")
    matrix = np.zeros((n_target, n_train), dtype=np.float64)
    t0 = time.time()
    next_log = 50
    for c in range(0, len(my_pool), B):
        pids = my_pool[c : c + B]
        chunk = [train_pool[i] for i in pids]
        grads = _example_grads_batch(
            model, tokenizer, chunk, builder,
            objective_mode=GradientObjective.GRPO_TRAIN, cfg=cfg,
            device=device, backend=backend, vllm_cfg=vllm_cfg, seed=cfg.seed + 20_000 + pids[0],
        )
        for i, g_train in zip(pids, grads):
            matrix[:, i] = (H @ g_train.to(H.device)).detach().cpu().numpy()
        done = min(c + B, len(my_pool))
        if main and (done >= next_log or done == len(my_pool)):
            print(f"    rank0 scored {done}/{len(my_pool)} ({time.time() - t0:.1f}s)")
            next_log = ((done // 50) + 1) * 50

    if world > 1:  # each rank filled disjoint columns → SUM assembles the full matrix
        mt = torch.from_numpy(matrix).to(device)
        all_reduce_sum_(mt)
        matrix = mt.cpu().numpy()
    scores = matrix.mean(axis=0)

    if save_dir is not None and main:  # only rank 0 writes (all ranks hold the same scores)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{tag}_if_matrix_step{checkpoint_step}.npy", matrix)
        np.save(save_dir / f"{tag}_if_scores_step{checkpoint_step}.npy", scores)
        print(f"  Saved {tag} influence artifacts under {save_dir}/")
    return scores


def compute_cg_pool_influence(
    cfg: ExperimentConfig,
    model,
    tokenizer,
    train_pool,
    target_set,
    device,
    *,
    checkpoint_step: int,
    save_dir: Path | None = None,
) -> np.ndarray:
    """Per-train aggregated CG influence (mean over the target set)."""
    model.eval()
    # Release training-time memory before the (memory-heavy) scoring pass.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Influence ALWAYS uses HF generation, never vLLM: CG needs gradients
    # (backward), which vLLM can't do, and spinning a 2nd vLLM engine collides
    # with TRL's colocate engine on the same GPU. Training still uses vLLM.
    backend = GenerationBackend.HF
    vllm_cfg = _vllm_config(cfg)

    # Pass a *builder* (not a built FVP) so _run_cg owns the FVP's lifetime and can
    # free it right after the H-solve, before the scoring backward.
    if cfg.if_method == "cg-empirical":
        def make_fvp():
            return _build_empirical_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)
        tag = "cg_empirical"
    else:  # "cg" — true analytic per-token Fisher
        def make_fvp():
            return _build_true_fisher_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)
        tag = "cg"

    return _run_cg(
        cfg, model, tokenizer, train_pool, target_set, device, make_fvp,
        tag=tag, checkpoint_step=checkpoint_step, save_dir=save_dir,
    )
