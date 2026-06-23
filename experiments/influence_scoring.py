from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import torch

from influence_rlvr import CGInfluence
from influence_rlvr.attribution.cg import policy_fisher_fvp_from_grad_cache
from influence_rlvr.preconditioner import load_adam_preconditioner_from_checkpoint
from influence_rlvr.fisher_fvp import FisherRow, build_policy_fisher_fvp
from influence_rlvr.generation import generate_rollout_batch
from influence_rlvr.gradients import (
    _compute_per_token_logps,
    _forward_per_token_logps_functional,
    _grad_vector_from_scalar,
    _state_dict_for_functional,
    compute_policy_gradient_bundle,
    compute_policy_gradient_bundle_batch,
    compute_sft_gradient_batch,
    gold_nll_per_example_functional,
)
from influence_rlvr.modes import GenerationBackend, GradientObjective, VLLMConfig
from influence_rlvr.utils import tokenize_prompt, tokenize_prompts_batch

from .config import ExperimentConfig
from .dist_utils import all_reduce_sum_, dist_info
from .influence import _make_verifier_reward_builder


def _vllm_config(cfg: ExperimentConfig, *, scoring: bool = False) -> VLLMConfig:
    # scoring=True: reuse the running trl vllm-serve over HTTP (VLLM_SERVER backend) —
    # no in-process engine. Carry the gen server's host/port; gpu_memory_utilization is
    # unused on that path (the engine lives in the separate gen-server process).
    return VLLMConfig(
        gpu_memory_utilization=(cfg.if_vllm_gpu_util if scoring
                                else cfg.vllm_gpu_memory_utilization),
        max_model_len=cfg.vllm_max_model_len,
        max_lora_rank=cfg.lora_r,
        server_host=(cfg.vllm_server_host if scoring else None),
        server_port=(cfg.vllm_server_port if scoring else None),
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
        enable_vllm=(backend == GenerationBackend.VLLM), generation_backend=backend,
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
        enable_vllm=(backend == GenerationBackend.VLLM), generation_backend=backend,
        max_new_tokens=cfg.if_max_new_tokens, temperature=0.7, top_p=0.9,
        seed=seed, epsilon=cfg.grpo_epsilon, beta=cfg.grpo_beta,
        objective_mode=objective_mode, vllm_config=vllm_cfg, model_id=cfg.model_id,
        logps_micro_batch_size=cfg.if_logps_micro_batch,
    )
    return [g.detach().to(dtype=torch.float32) for g in res["grad"]]


def _example_sft_grads_batch(model, tokenizer, samples, *, cfg, device):
    """`if_grad="gold"` minibatch: the SFT gold-answer gradient (no rollouts)."""
    return compute_sft_gradient_batch(
        model, tokenizer,
        [s["prompt"] for s in samples],
        [s.get("solution", "") for s in samples],
        device=device, logps_micro_batch_size=cfg.if_logps_micro_batch,
    )


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
    return build_policy_fisher_fvp(
        model, rows, normalize=len(rows),
        spectral_normalize=cfg.cg_normalize_fisher,
        n_power_iters=cfg.cg_power_iters,
        power_seed=cfg.seed,
    )


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


def _spearman_jvp(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


_JVP_ASSERTED = False  # run the JVP-vs-dot correctness gate once per process


def _run_jvp_pool(cfg, live_model, tokenizer, train_pool, my_pool, H, *,
                  checkpoint_step, device, D, main, t0):
    """Forward-mode (JVP) pool scoring for if_grad=gold. Loads a FRESH fp32 model from the
    checkpoint, builds the fixed tangent h_bar = H.mean(0) — which equals P⊙ḡ_target exactly
    for any assembled-H operator (P is target-independent, so it factors out of the target
    mean) — and scores prompts as score(z)=⟨∇L(z),h_bar⟩ via BATCHED fp32 forward-mode passes
    (if_score_batch prompts share one pass; the shared tangent makes per-example scores fall
    out of a single forward-mode sweep — the batching the per-example backward can't do). No
    backward, no D-vector. Returns a (1, n_train) matrix with THIS rank's columns filled
    (disjoint), to flow through the shared all_reduce + mean tail unchanged.

    fp32 + eager attention because bf16/fused-kernel forward-mode AD is buggy through Qwen3
    RMSNorm. The live bf16 model is NEVER touched (run_if_prune resumes it next window)."""
    global _JVP_ASSERTED
    from peft import PeftModel
    from torch.func import grad as func_grad
    from torch.func import jvp as func_jvp
    from transformers import AutoModelForCausalLM

    ckpt_dir = cfg.grpo_output_dir / f"checkpoint-{checkpoint_step}"
    if not (ckpt_dir / "adapter_model.safetensors").is_file() and not (ckpt_dir / "adapter_model.bin").is_file():
        raise FileNotFoundError(
            f"if_jvp needs the on-disk LoRA adapter at {ckpt_dir} (the reverse-mode path "
            f"uses the in-memory model; JVP loads a fresh fp32 copy from disk).")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.float32, attn_implementation="eager").to(device)
    base.config.use_cache = False
    jmodel = PeftModel.from_pretrained(base, str(ckpt_dir), is_trainable=True).to(device)
    jmodel.eval()  # NOTE: do NOT enable gradient checkpointing (breaks functional_call + JVP)

    names = tuple(n for n, p in jmodel.named_parameters() if p.requires_grad)
    live_names = tuple(n for n, p in live_model.named_parameters() if p.requires_grad)
    if names != live_names:
        raise RuntimeError("if_jvp: fp32 model trainable-param order != live model — "
                           "H (built on the live model) would misalign with the JVP tangent.")
    ptuple = tuple(jmodel.get_parameter(n).detach() for n in names)
    if sum(p.numel() for p in ptuple) != D:
        raise RuntimeError(f"if_jvp: param count {sum(p.numel() for p in ptuple)} != D={D}.")

    # h_bar = H.mean(0): the collapsed fixed tangent. H[j] already holds the per-target
    # operator output (dot: g_test; tracin-adam: P⊙g_test; cg: (F+λI)⁻¹g_test), all fp32,
    # so mean-over-targets == the exact direction the per-target scores average to.
    h_bar = H.mean(dim=0).to(device=device, dtype=torch.float32)
    h_tan, off = [], 0
    for n in names:
        num = jmodel.get_parameter(n).numel()
        h_tan.append(h_bar[off:off + num].view_as(jmodel.get_parameter(n))); off += num
    h_tan = tuple(h_tan)

    eos = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    def _enc_single(sample):  # for the per-example reverse-mode REFERENCE in the assert
        _, p_ids, p_am = tokenize_prompts_batch(tokenizer, [sample["prompt"]], device)
        gold = str(sample.get("solution", "") or "")
        c = tokenizer(f"\\boxed{{{gold}}}", add_special_tokens=False,
                      return_tensors="pt")["input_ids"].to(device)
        if eos is not None:
            c = torch.cat([c, torch.full((1, 1), int(eos), device=device, dtype=c.dtype)], dim=1)
        return p_ids[0], p_am[0], c, torch.ones_like(c)

    def loss_single(pt, p_ids, p_am, c, cm):  # == compute_sft_gradient_batch's per-example loss
        sd = _state_dict_for_functional(jmodel, pt, names)
        ptl = _forward_per_token_logps_functional(jmodel, sd, p_ids, p_am, c, cm)
        return -(ptl * cm).sum() / cm.sum().clamp(min=1)

    def encode_batch(samples):
        """Right-pad B (prompt+\\boxed{gold}+eos) sequences → input_ids/attn/comp_mask [B,L].
        Prompt tokenization matches _enc_single (same tokenize_prompts_batch render)."""
        rows = []
        for s in samples:
            _, p_ids, _ = tokenize_prompts_batch(tokenizer, [s["prompt"]], device)
            p = p_ids[0].tolist()
            gold = str(s.get("solution", "") or "")
            c = tokenizer(f"\\boxed{{{gold}}}", add_special_tokens=False)["input_ids"]
            if eos is not None:
                c = c + [int(eos)]
            rows.append((p, c))
        L = max(len(p) + len(c) for p, c in rows)
        B = len(rows)
        input_ids = torch.full((B, L), int(pad_id), dtype=torch.long, device=device)
        attn = torch.zeros((B, L), dtype=torch.long, device=device)
        cmask = torch.zeros((B, L), dtype=torch.long, device=device)
        for bi, (p, c) in enumerate(rows):
            seq = p + c
            input_ids[bi, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[bi, :len(seq)] = 1
            cmask[bi, len(p):len(seq)] = 1
        return input_ids, attn, cmask

    def jvp_scores_batch(samples):
        """B influence scores from ONE forward-mode pass: jvp of the per-example NLL vector
        [B] with the shared tangent h_tan → the [B] directional derivatives ⟨∇L_i, h⟩."""
        enc = encode_batch(samples)
        def f(pt):
            sd = _state_dict_for_functional(jmodel, pt, names)
            return gold_nll_per_example_functional(jmodel, sd, *enc)
        return func_jvp(f, (ptuple,), (h_tan,))[1]  # [B]

    # Batched forward-mode scoring: JB prompts per forward-mode pass (the throughput win the
    # per-example backward can't get). JB bounds the [JB × L × vocab] logits tensor (fp32,
    # ~doubled by forward-mode) — its OWN memory knob (if_jvp_batch), NOT if_score_batch
    # (which sizes rollout generation, a different/lighter profile).
    JB = max(1, cfg.if_jvp_batch)

    # First-call gate (rank0, once/process): BATCHED-JVP vs per-example reverse-mode dot on
    # the SAME fp32 model → validates the JVP recipe AND the batched padding/masking. Raise,
    # never silently fall back: a misalignment yields finite-but-wrong scores with no error.
    # Score jref in JB-sized chunks so the gate never spikes past the production batch memory.
    if main and not _JVP_ASSERTED:
        grad_of = func_grad(loss_single, argnums=0)
        K = min(16, len(my_pool))
        kids = my_pool[:K]
        dref = []
        for i in kids:
            g = grad_of(ptuple, *_enc_single(train_pool[i]))
            dref.append(float(sum((gi * hi).sum() for gi, hi in zip(g, h_tan))))
        jref = []
        for c in range(0, len(kids), JB):
            jref.extend(float(x) for x in
                        jvp_scores_batch([train_pool[i] for i in kids[c:c + JB]]))
        rho = _spearman_jvp(dref, jref)
        mrel = max(abs(a - b) / (abs(a) + 1e-12) for a, b in zip(dref, jref)) if dref else 0.0
        # SELECTION only uses the RANKING → Spearman is the real gate (a param-order / loss
        # bug scrambles it to ≪1). max_rel is a loose sanity bound: the batched logsumexp +
        # the Adam P's huge dynamic range (~1e3-1e8) push fp32 roundoff to ~1e-3 even when
        # ranking is perfect, so 1e-2 (10× margin) — NOT 1e-3 — is the right ceiling.
        if not (rho > 0.999 and mrel < 1e-2):
            raise RuntimeError(
                f"if_jvp correctness assert FAILED: Spearman={rho:.5f} max_rel={mrel:.2e} "
                f"(need >0.999 / <1e-2) — batched-JVP scores != reverse-mode dot. "
                f"dref={dref[:4]} jref={jref[:4]}")
        print(f"  [if_jvp] correctness gate PASS (K={K}, JB={JB}, Spearman={rho:.5f}, "
              f"max_rel={mrel:.1e})")
    _JVP_ASSERTED = True

    matrix = np.zeros((1, len(train_pool)), dtype=np.float64)
    next_log = 50
    for c in range(0, len(my_pool), JB):
        pids = my_pool[c:c + JB]
        try:
            jvec = jvp_scores_batch([train_pool[i] for i in pids])
        except Exception as e:
            raise RuntimeError(
                f"if_jvp forward-mode failed on pool chunk @{pids[0]}: {type(e).__name__}: {e} "
                f"— forward-mode AD needs eager attention (set) and fp32 (set).") from e
        for k, i in enumerate(pids):
            matrix[0, i] = float(jvec[k])
        done = min(c + JB, len(my_pool))
        if main and (done >= next_log or done == len(my_pool)):
            print(f"    rank0 jvp-scored {done}/{len(my_pool)} ({time.time() - t0:.1f}s)")
            next_log = ((done // 50) + 1) * 50

    del jmodel, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return matrix


# ── Shared CG solve + streamed scoring ──────────────────────────────────────
def _run_cg(cfg, model, tokenizer, train_pool, target_set, device, make_fvp, *,
            tag, checkpoint_step, save_dir):
    # The GRADIENT is always HF (backward), but the per-example ROLLOUT SAMPLING — the
    # slow half — is offloaded to the RUNNING trl vllm-serve server over HTTP (the
    # VLLM_SERVER backend), reusing the training gen engine. NOT an in-process engine:
    # spinning one inside the DDP scoring process deadlocks on vLLM's own TCPStore.
    # Gated by `if_vllm_gen` AND server mode (the only mode with a gen server to reuse).
    offload = cfg.if_vllm_gen and cfg.use_vllm and cfg.vllm_mode == "server"
    backend = GenerationBackend.VLLM_SERVER if offload else GenerationBackend.HF
    vllm_cfg = _vllm_config(cfg, scoring=offload)
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

    # First-order TracIn influence skips the Fisher/CG solve and sets h directly:
    #   "dot"         → h = g_test           (plain gradient dot product)
    #   "tracin-adam" → h = P_adam ⊙ g_test  (Adam-preconditioned dot: the faithful
    #                   first-order effect of one AdamW step, with the diagonal
    #                   P_d = 1/(√v̂_d+ε) read from this checkpoint's optimizer.pt)
    # The Fisher FVP/CG is needed ONLY to turn g_test into h; both first-order
    # methods skip it and stream the (optionally preconditioned) target gradients.
    # These are the streaming analogs of attribution.TracInInfluence /
    # TracInAdamInfluence (which stack the whole pool at once): we fold P into h
    # once per target and stream g_train one batch at a time to bound memory.
    use_fisher = cfg.if_method not in ("dot", "tracin-adam")
    use_adam = cfg.if_method == "tracin-adam"
    # if_grad="gold": swap the on-policy rollout gradient for the SFT gold-answer
    # gradient (no rollouts). The OPERATOR above (Fisher/Adam/dot) is unchanged — it's
    # applied to whichever gradients. So tracin-adam + gold = tracin-adam on g_gold.
    is_gold = cfg.if_grad == "gold"
    if use_fisher:
        fvp = make_fvp()
        cg = CGInfluence(fvp_fn=fvp, lambda_damp=cfg.lambda_damp,
                         cg_iters=cfg.cg_iters, cg_tol=cfg.cg_tol)

    precond = None
    if use_adam:
        ckpt_dir = cfg.grpo_output_dir / f"checkpoint-{checkpoint_step}"
        eps = cfg.tracin_adam_eps if cfg.tracin_adam_eps and cfg.tracin_adam_eps > 0 else None
        precond = load_adam_preconditioner_from_checkpoint(
            model, ckpt_dir, device=device, eps=eps)
        if precond is None and main:
            print(f"  [tracin-adam] WARNING: no optimizer.pt under {ckpt_dir}; "
                  f"falling back to un-preconditioned dot product.")
        elif precond is not None and main:
            # Log P's dynamic range: a very large max (dormant coords, v̂≈0) can
            # dominate the influence dot → raise --tracin-adam-eps if ρ is noisy.
            print(f"  [tracin-adam] Adam diagonal preconditioner loaded "
                  f"(D={precond.numel()}) from {ckpt_dir.name}/optimizer.pt; "
                  f"P-range [{precond.min():.2e}, median {precond.median():.2e}, "
                  f"{precond.max():.2e}]" + (f", eps={eps:g}" if eps else ""))

    n_target = len(target_set)
    if main:
        if use_fisher:
            what = "solving (F+λI)h = g_test"
        elif precond is not None:
            what = "h = P_adam ⊙ g_test (Adam-preconditioned dot)"
        else:
            what = "h = g_test (dot-product)"
        print(f"  [{cfg.if_method}] {what} for {n_target} targets (batch={B}, world={world})...")
    H = torch.zeros(n_target, D, device=device, dtype=torch.float32)
    info: dict = {"status": "?", "n_iters": 0, "final_residual": float("nan")}
    my_targets = list(range(rank, n_target, world))
    for c in range(0, len(my_targets), B):
        tids = my_targets[c : c + B]
        chunk = [target_set[j] for j in tids]
        grads = (
            _example_sft_grads_batch(model, tokenizer, chunk, cfg=cfg, device=device)
            if is_gold else _example_grads_batch(
                model, tokenizer, chunk, builder,
                objective_mode=GradientObjective.EXPECTED_REWARD_PG, cfg=cfg,
                device=device, backend=backend, vllm_cfg=vllm_cfg,
                seed=cfg.seed + 10_000 + tids[0],
            )
        )
        for j, g_test in zip(tids, grads):
            if use_fisher:
                h, info = cg.solve(g_test.to(device))
            elif precond is not None:
                h = g_test.to(device) * precond  # Adam-preconditioned dot
            else:
                h = g_test.to(device)  # dot-product: h = g_test
            H[j] = h.to(H.dtype)
        if main and use_fisher:
            print(f"    target {min(c + B, len(my_targets))}/{len(my_targets)} (rank0): "
                  f"CG {info['status']} in {info['n_iters']} iters "
                  f"(resid={info['final_residual']})")
    precond = None  # free the D-vector before the memory-heavy scoring backward
    all_reduce_sum_(H)  # each rank filled disjoint rows → SUM assembles full H

    # if_cosine (LESS-style): rank pool examples by DIRECTIONAL alignment, not raw dot. The
    # plain dot ⟨H[j], g_train⟩ scales with |g_train|, so a big-gradient example (e.g. a long
    # answer, or one the model is very wrong about) dominates the ranking REGARDLESS of whether
    # it points toward the target — exactly why a physics target selected mostly Economics.
    # Unit-normalizing each target row (here) + each g_train (in the pool loop) → score is the
    # mean cosine(H[j], g_train), magnitude-free.
    if cfg.if_cosine:
        H = H / H.norm(dim=1, keepdim=True).clamp(min=1e-12)
        if main:
            print("  [if_cosine] H rows unit-normalized; pool scores = mean cosine "
                  "(direction, not magnitude)")

    # Release the FVP + flush the CG double-backward graphs before the (memory-heavy,
    # full-vocab-logit) scoring backward. Also re-enable gradient checkpointing: the
    # FVP needed it OFF for double-backward, but scoring only needs first-order grads,
    # so checkpointing here cuts activation memory and buys a bigger usable batch.
    if use_fisher:
        del cg, fvp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    n_train = len(train_pool)
    my_pool = list(range(rank, n_train, world))
    # if_jvp (gold only): replace the per-example backward+dot with one fp32 forward-mode
    # pass per prompt against the collapsed tangent H.mean(0). Produces a (1,n_train) matrix
    # (matrix.mean(0)==scores still holds) that flows through the same all_reduce + save tail.
    use_jvp = cfg.if_jvp and is_gold and not cfg.if_cosine  # cosine needs |g_train| (JVP skips it)
    t0 = time.time()
    if use_jvp:
        if main:
            print(f"  [{cfg.if_method}/{cfg.if_grad}] JVP pool scoring (fp32 forward-mode, "
                  f"no backward) — {n_train} prompts ({len(my_pool)}/rank × {world} ranks)...")
        matrix = _run_jvp_pool(cfg, model, tokenizer, train_pool, my_pool, H,
                               checkpoint_step=checkpoint_step, device=device, D=D,
                               main=main, t0=t0)
    else:
        if main:
            gen_label = "none/gold-SFT" if is_gold else backend.name
            print(f"  [{cfg.if_method}/{cfg.if_grad}] scoring {n_train} train prompts "
                  f"({len(my_pool)}/rank × {world} ranks, batch={B}, gen={gen_label})...")
        matrix = np.zeros((n_target, n_train), dtype=np.float64)
        next_log = 50
        for c in range(0, len(my_pool), B):
            pids = my_pool[c : c + B]
            chunk = [train_pool[i] for i in pids]
            grads = (
                _example_sft_grads_batch(model, tokenizer, chunk, cfg=cfg, device=device)
                if is_gold else _example_grads_batch(
                    model, tokenizer, chunk, builder,
                    objective_mode=GradientObjective.GRPO_TRAIN, cfg=cfg,
                    device=device, backend=backend, vllm_cfg=vllm_cfg,
                    seed=cfg.seed + 20_000 + pids[0],
                )
            )
            for i, g_train in zip(pids, grads):
                gt = g_train.to(H.device)
                if cfg.if_cosine:  # unit-normalize → ⟨H[j], g_train/|g_train|⟩ = cosine
                    gt = gt / gt.norm().clamp(min=1e-12)
                matrix[:, i] = (H @ gt).detach().cpu().numpy()
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
    else:  # "cg" (true analytic per-token Fisher) or a first-order method (dot /
           # tracin-adam) that ignores make_fvp entirely — tag names the artifacts.
        def make_fvp():
            return _build_true_fisher_fvp(cfg, model, tokenizer, train_pool, device, backend, vllm_cfg)
        tag = {"dot": "dot", "tracin-adam": "tracin_adam"}.get(cfg.if_method, "cg")

    return _run_cg(
        cfg, model, tokenizer, train_pool, target_set, device, make_fvp,
        tag=tag, checkpoint_step=checkpoint_step, save_dir=save_dir,
    )
