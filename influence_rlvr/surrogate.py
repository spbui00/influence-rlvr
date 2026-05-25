"""Surrogate IF for LLM-scale RLVR runs (checkpoint-free).

Standalone module that mirrors the surrogate IF logic developed in
`influence_rlvr.toy_grpo.compute_toy_surrogate_gradient` and the `method="surrogate"`
branch of `compute_toy_historical_fisher_influence`, but operates at LLM scale
against runs produced by `HistoricalBatchGRPOTrainer`.

It reuses the existing rollout / per-token-logp / reward infrastructure but
computes its own per-rollout gradients via `_grad_vector_from_scalar` (straight K
autograd.grad calls per train prompt — no vmap).

Per-train-example output structure follows the toy convention exactly:
- 1 "numerator" info: `{grad: -Σⱼ ŵⱼ ∇log πⱼ, geometry_feature: 0,
                        historical_weight: 0, score_weight: 1}`
- K "rollout-Fisher" infos: `{grad: 0, geometry_feature: ∇log πⱼ,
                              historical_weight: ŵⱼ, score_weight: 0}`

`TrajectoryFisherInfluence` already handles this structure: `score_weight` masks
the numerator entries when reading scores, and `historical_weight` accumulates
the rollout entries when building the Fisher.

The score-extraction stride trick mirrors `lds_eval_toy_grpo.py:
scores.reshape(n_train, K+1)[:, 0]`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .attribution.fisher import TrajectoryFisherInfluence
from .generation import generate_rollout_batch, rollout_to_completions
from .gradients import (
    _compute_per_token_logps,
    _compute_ref_per_token_logps,
    _evaluate_rewards,
    _grad_vector_from_scalar,
    compute_policy_gradient_bundle,
    compute_sft_gradient,
)
from .modes import GenerationBackend, GeometryFeatureMode, GradientObjective, VLLMConfig
from .utils import tokenize_prompt


@dataclass
class SurrogateGradientBundle:
    """Output of `compute_surrogate_gradient_bundle` for a single training prompt."""

    grad: torch.Tensor                       # -Σⱼ ŵⱼ ∇log πⱼ (loss-form), flat vec on CPU
    rollout_grads: list[torch.Tensor]        # K per-rollout ∇log πⱼ, flat vecs on CPU
    w_hat: torch.Tensor                      # K self-normalized weights, on CPU
    debug: dict = field(default_factory=dict)


def _gradient_cache_key(
    prompt: Any,
    *,
    model_id: str | None,
    final_checkpoint: str | None,
    ref_checkpoint: str | None,
    G: int,
    beta: float,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int | None,
    dtype: str | None,
) -> str:
    """Stable SHA256 hash over everything that affects a per-prompt surrogate gradient.

    Any change in θˢ, θ_ref, prompt content, sampling params, or KL temperature
    invalidates the cache. Outputs a 16-char hex key suitable for filenames.
    """
    payload = {
        "prompt": json.dumps(prompt, sort_keys=True, ensure_ascii=False),
        "model_id": model_id,
        "final_checkpoint": final_checkpoint,
        "ref_checkpoint": ref_checkpoint or "<base_no_adapter>",
        "G": int(G),
        "beta": float(beta),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "seed": seed,
        "dtype": dtype or "unknown",
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_cached_bundle(path: Path) -> SurrogateGradientBundle | None:
    if not path.is_file():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return SurrogateGradientBundle(
        grad=data["grad"],
        rollout_grads=list(data["rollout_grads"]),
        w_hat=data["w_hat"],
        debug=dict(data.get("debug", {})),
    )


def _save_cached_bundle(path: Path, bundle: SurrogateGradientBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "grad": bundle.grad,
            "rollout_grads": list(bundle.rollout_grads),
            "w_hat": bundle.w_hat,
            "debug": bundle.debug,
        },
        path,
    )


def compute_surrogate_gradient_bundle(
    peft_model,
    ref_model,
    tokenizer,
    prompt,
    reward_funcs,
    *,
    G: int = 4,
    device: str | torch.device = "cpu",
    beta: float = 0.1,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = None,
    generation_backend: GenerationBackend | None = None,
    vllm_config: VLLMConfig | None = None,
    adapter_path: str | None = None,
    model_id: str | None = None,
) -> SurrogateGradientBundle:
    """LLM equivalent of `toy_grpo.compute_toy_surrogate_gradient`.

    Steps:
    1. Sample G rollouts from `peft_model` (= π_{θˢ}).
    2. Score each rollout under both `peft_model` (with grad) and `ref_model` (no grad).
    3. `R̃ = r/β − log(π_{θˢ}/π_ref)`; `ŵ = softmax(R̃)`.
    4. For each rollout j, compute `∇log π(yⱼ|x)` via straight autograd.
    5. `g_surrogate = -Σⱼ ŵⱼ ∇log πⱼ`  (loss-gradient sign — matches the toy fix).
    """
    if generation_backend is None:
        generation_backend = GenerationBackend.HF
    if vllm_config is None:
        vllm_config = VLLMConfig()

    peft_model.eval()
    peft_model.zero_grad()
    if ref_model is not None:
        ref_model.eval()
    # If ref_model is None, _compute_ref_per_token_logps falls back to peft_model
    # with its adapter disabled (or a "ref" adapter slot if one exists). That's
    # the "π_ref = base model with no LoRA delta" interpretation.

    prompt_text, prompt_ids, prompt_mask = tokenize_prompt(tokenizer, prompt, device)
    rollout = generate_rollout_batch(
        peft_model,
        tokenizer,
        prompt_ids,
        prompt_mask,
        backend=generation_backend,
        num_samples=G,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        vllm_config=vllm_config,
        adapter_path=adapter_path,
        model_id=model_id,
    )
    completions_trl = rollout_to_completions(rollout)
    total_rewards, reward_breakdown = _evaluate_rewards(reward_funcs, completions_trl, device)

    response_ids = rollout.token_ids
    response_mask = rollout.response_mask

    # 1. Compute log probs WITHOUT grad for the weight computation. No autograd
    #    graph is kept here — keeps peak memory low on Mac MPS.
    token_mask = response_mask.float()
    with torch.no_grad():
        per_token_logps_nograd = _compute_per_token_logps(
            peft_model, prompt_ids, prompt_mask, response_ids, response_mask
        )
        ref_per_token_logps = _compute_ref_per_token_logps(
            peft_model, ref_model, prompt_ids, prompt_mask, response_ids, response_mask
        )
        seq_logp_detached = (per_token_logps_nograd * token_mask).sum(dim=1)  # (G,)
        seq_logp_ref = (ref_per_token_logps * token_mask).sum(dim=1)          # (G,)

    # 2. R̃ = r/β − log(π_{θˢ}/π_ref); self-normalized IS weights.
    r_for_weights = total_rewards.detach().to(seq_logp_detached.device)
    r_tilde = r_for_weights / beta - (seq_logp_detached - seq_logp_ref)
    w_hat = F.softmax(r_tilde, dim=0).detach().to(dtype=torch.float32).cpu()

    # 3. Per-rollout ∇log π(yⱼ|x): redo the forward WITH grad ONE rollout at a
    #    time. Memory drops from K× peak to 1× peak. Cost: K independent
    #    forward+backward passes instead of one batched forward + K backwards.
    rollout_grads: list[torch.Tensor] = []
    for j in range(int(response_ids.shape[0])):
        pt_lp_j = _compute_per_token_logps(
            peft_model,
            prompt_ids, prompt_mask,
            response_ids[j : j + 1], response_mask[j : j + 1],
        )  # (1, max_resp_len), requires_grad
        seq_logp_j = (pt_lp_j * token_mask[j : j + 1]).sum()
        gj = _grad_vector_from_scalar(peft_model, seq_logp_j, retain_graph=False)
        rollout_grads.append(gj)
        del pt_lp_j, seq_logp_j

    # Surface the detached seq_logp tensor for debug output.
    seq_logp = seq_logp_detached

    # g_surrogate stored as the loss-gradient form: −Σⱼ ŵⱼ ∇log πⱼ
    # (matches the sign convention used by the rest of the IF pipeline).
    stacked = torch.stack(rollout_grads, dim=0)  # (G, dim) on CPU
    g_surrogate = -(stacked.t() @ w_hat).contiguous()

    debug = {
        "total_rewards": total_rewards.detach().cpu().tolist(),
        "w_hat": w_hat.tolist(),
        "r_tilde": r_tilde.detach().cpu().tolist(),
        "seq_logp": seq_logp.detach().cpu().tolist(),
        "seq_logp_ref": seq_logp_ref.cpu().tolist(),
        "reward_breakdown": reward_breakdown,
    }
    return SurrogateGradientBundle(
        grad=g_surrogate,
        rollout_grads=rollout_grads,
        w_hat=w_hat,
        debug=debug,
    )


def build_surrogate_train_infos(
    bundle: SurrogateGradientBundle,
    prompt_idx: int,
    *,
    name: str | None = None,
) -> list[dict]:
    """Wrap a `SurrogateGradientBundle` into the `(1 + K)` info entries expected by
    `TrajectoryFisherInfluence`. Order matters — the numerator must be first
    so the stride trick (`scores[::G+1]`) picks it out cleanly."""
    grad_dim = bundle.grad.numel()
    zero_grad = torch.zeros(grad_dim, dtype=torch.float32)
    infos: list[dict[str, Any]] = []

    # Numerator: contributes to `g_train^T H^{-1} g_test`, not to Fisher.
    infos.append({
        "grad": bundle.grad,
        "geometry_feature": zero_grad,
        "historical_weight": 0.0,
        "score_weight": 1.0,
        "train_index": prompt_idx,
        "name": name or f"train_{prompt_idx}",
        "kind": "numerator",
    })
    # Rollout entries: contribute to Fisher via geometry_feature × historical_weight.
    for j, gj in enumerate(bundle.rollout_grads):
        infos.append({
            "grad": zero_grad,
            "geometry_feature": gj,
            "historical_weight": float(bundle.w_hat[j].item()),
            "score_weight": 0.0,
            "train_index": prompt_idx,
            "name": f"{name or f'train_{prompt_idx}'}_rollout_{j}",
            "kind": f"rollout_{j}",
        })
    return infos


def compute_surrogate_if_scores(
    peft_model,
    ref_model,
    tokenizer,
    train_dataset,
    test_dataset,
    reward_fn_builder: Callable,
    *,
    device: str | torch.device = "cpu",
    G: int = 4,
    beta: float = 0.1,
    lambda_damp: float = 1.0,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed_base: int | None = None,
    generation_backend: GenerationBackend | None = None,
    vllm_config: VLLMConfig | None = None,
    adapter_path: str | None = None,
    model_id: str | None = None,
    train_limit: int | None = None,
    test_limit: int | None = None,
    progress: bool = True,
    test_objective: str = "sft_solution",
    gradient_cache_dir: Path | None = None,
    final_checkpoint_id: str | None = None,
    ref_checkpoint_id: str | None = None,
    dtype_id: str | None = None,
) -> dict:
    """End-to-end surrogate IF over a trained LoRA run.

    The model + ref_model + adapter_path passed in should describe `θˢ` and `θ_ref`.
    `train_dataset` and `test_dataset` are HuggingFace-Dataset-style mappings, each
    yielding `{"prompt": ..., "solution": ..., ...}`.
    `reward_fn_builder(sample, G)` is the same callable the existing trajectory
    pipeline uses to construct per-prompt reward functions.

    Returns:
        scores: np.ndarray shape (n_test, n_train) of `g_test^T (F+λI)^{-1} g_surrogate`
        matrix_raw: full (n_test, n_train*(G+1)) matrix before the stride extraction
        breakdown: TrajectoryFisherInfluence breakdown dict
        train_debug: per-train-prompt debug info
        test_debug: per-test-prompt debug info
    """
    n_train = len(train_dataset) if train_limit is None else min(train_limit, len(train_dataset))
    n_test = len(test_dataset) if test_limit is None else min(test_limit, len(test_dataset))

    # ----- Train side: (1 + G) entries per prompt -----
    train_infos: list[dict] = []
    train_debug: list[dict] = []
    n_cache_hits = 0
    cache_dir = Path(gradient_cache_dir) if gradient_cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        sample = train_dataset[i]
        seed = None if seed_base is None else seed_base + i

        bundle: SurrogateGradientBundle | None = None
        cache_path: Path | None = None
        cache_status = "no-cache"
        if cache_dir is not None:
            key = _gradient_cache_key(
                sample["prompt"],
                model_id=model_id,
                final_checkpoint=final_checkpoint_id,
                ref_checkpoint=ref_checkpoint_id,
                G=G, beta=beta, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens, seed=seed, dtype=dtype_id,
            )
            cache_path = cache_dir / f"train_grad_surr_{key}.pt"
            bundle = _load_cached_bundle(cache_path)
            if bundle is not None:
                cache_status = "hit"
                n_cache_hits += 1
            else:
                cache_status = "miss"

        if bundle is None:
            reward_funcs = reward_fn_builder(sample, G)
            bundle = compute_surrogate_gradient_bundle(
                peft_model, ref_model, tokenizer, sample["prompt"], reward_funcs,
                G=G, device=device, beta=beta,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, seed=seed,
                generation_backend=generation_backend, vllm_config=vllm_config,
                adapter_path=adapter_path, model_id=model_id,
            )
            if cache_path is not None:
                _save_cached_bundle(cache_path, bundle)

        train_infos.extend(build_surrogate_train_infos(
            bundle, prompt_idx=i, name=sample.get("name") or f"train_{i}"
        ))
        train_debug.append(bundle.debug)
        if progress:
            mean_r = float(np.mean(bundle.debug.get("total_rewards", [0.0])))
            tag = f"  [{cache_status}]" if cache_dir is not None else ""
            print(
                f"  train {i+1}/{n_train}: ||g_surr||={bundle.grad.norm().item():.4f} "
                f"mean_reward={mean_r:.3f}{tag}"
            )

    if cache_dir is not None and progress:
        print(f"  Gradient cache: {n_cache_hits}/{n_train} hits → {cache_dir}")

    # ----- Test side -----
    #
    # Two objectives supported:
    #   * "expected_reward_pg" — gradient of negative expected reward computed via
    #     sampled rollouts (the original RL-style metric). Zero when no rollout
    #     gets reward.
    #   * "sft_solution"      — gradient of cross-entropy loss against the gold
    #     `solution` string. Always nonzero as long as the model isn't 100% confident
    #     on the gold answer. No rollouts needed, no reward matching needed.
    if test_objective not in {"expected_reward_pg", "sft_solution"}:
        raise ValueError(f"Unsupported test_objective={test_objective!r}")

    test_infos: list[dict] = []
    test_debug: list[dict] = []
    for i in range(n_test):
        sample = test_dataset[i]
        name = sample.get("name") or f"test_{i}"
        seed = None if seed_base is None else seed_base + 10_000 + i

        if test_objective == "sft_solution":
            target = sample.get("solution") or ""
            if not target:
                if progress:
                    print(f"  test  {i+1}/{n_test}: SKIPPED — empty 'solution' field")
                continue
            grad = compute_sft_gradient(
                peft_model, tokenizer, sample["prompt"], target, device,
            )
            test_infos.append({"grad": grad, "name": name})
            test_debug.append({"objective": "sft_solution", "target": target})
            if progress:
                print(
                    f"  test  {i+1}/{n_test}: ||g_test||={grad.norm().item():.4f} (SFT on {target!r})"
                )
        else:  # expected_reward_pg
            reward_funcs = reward_fn_builder(sample, G)
            result = compute_policy_gradient_bundle(
                peft_model, tokenizer, sample["prompt"], reward_funcs,
                G=G, device=device,
                beta=0.0, ref_model=None,
                objective_mode=GradientObjective.EXPECTED_REWARD_PG,
                geometry_feature_mode=GeometryFeatureMode.NONE,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, seed=seed,
                generation_backend=generation_backend, vllm_config=vllm_config,
                adapter_path=adapter_path, model_id=model_id,
            )
            test_infos.append({"grad": result["grad"], "name": name})
            test_debug.append(result.get("debug", {}))
            if progress:
                print(
                    f"  test  {i+1}/{n_test}: ||g_test||={result['grad'].norm().item():.4f} (PG)"
                )

    # ----- Fisher solve via existing trajectory infrastructure -----
    checkpoint_infos = [{
        "step": 1,
        "learning_rate": 1.0,
        "train_infos": train_infos,
        "test_infos": test_infos,
    }]
    tfi = TrajectoryFisherInfluence(
        lambda_damp=lambda_damp, normalize=False, solver="woodbury",
    )
    matrix, breakdown = tfi.compute_matrix(checkpoint_infos, return_breakdown=True)

    # Extract the numerator entries: positions 0, (G+1), 2(G+1), ... in train axis.
    stride = G + 1
    scores = np.asarray(matrix)[:, ::stride][:, :n_train]

    return {
        "scores": scores,
        "matrix_raw": np.asarray(matrix),
        "breakdown": breakdown,
        "train_debug": train_debug,
        "test_debug": test_debug,
        "stride": stride,
        "n_train": n_train,
        "n_test": n_test,
    }


# =========================================================================
# Historical-last IF: single-checkpoint linearization using the GRPO loss
# gradient (instead of the surrogate's importance-weighted ∇log π).
# Mirrors the toy's `method="historical"` branch evaluated only at θˢ.
# =========================================================================

def _test_grad_for_prompt(
    peft_model,
    tokenizer,
    sample: dict,
    *,
    test_objective: str,
    device,
    G: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    reward_fn_builder: Callable,
    generation_backend,
    vllm_config,
    adapter_path,
    model_id,
) -> tuple[torch.Tensor | None, dict, str]:
    """Compute g_test for a single test prompt under the chosen test_objective.
    Returns (grad, debug, status_msg). grad is None when skipped (e.g. empty solution)."""
    if test_objective == "sft_solution":
        target = sample.get("solution") or ""
        if not target:
            return None, {"objective": "sft_solution", "target": ""}, "SKIPPED (empty solution)"
        grad = compute_sft_gradient(peft_model, tokenizer, sample["prompt"], target, device)
        return grad, {"objective": "sft_solution", "target": target}, f"||g_test||={grad.norm().item():.4f} (SFT on {target!r})"
    if test_objective == "expected_reward_pg":
        reward_funcs = reward_fn_builder(sample, G)
        result = compute_policy_gradient_bundle(
            peft_model, tokenizer, sample["prompt"], reward_funcs,
            G=G, device=device, beta=0.0, ref_model=None,
            objective_mode=GradientObjective.EXPECTED_REWARD_PG,
            geometry_feature_mode=GeometryFeatureMode.NONE,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, seed=seed,
            generation_backend=generation_backend, vllm_config=vllm_config,
            adapter_path=adapter_path, model_id=model_id,
        )
        return result["grad"], result.get("debug", {}), f"||g_test||={result['grad'].norm().item():.4f} (PG)"
    raise ValueError(f"Unsupported test_objective={test_objective!r}")


def compute_historical_last_if_scores(
    peft_model,
    ref_model,
    tokenizer,
    train_dataset,
    test_dataset,
    reward_fn_builder: Callable,
    *,
    device: str | torch.device = "cpu",
    G: int = 4,
    beta: float = 0.0,
    lambda_damp: float = 1.0,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed_base: int | None = None,
    generation_backend: GenerationBackend | None = None,
    vllm_config: VLLMConfig | None = None,
    adapter_path: str | None = None,
    model_id: str | None = None,
    train_limit: int | None = None,
    test_limit: int | None = None,
    progress: bool = True,
    test_objective: str = "sft_solution",
    gradient_cache_dir: Path | None = None,
    final_checkpoint_id: str | None = None,
    ref_checkpoint_id: str | None = None,
    dtype_id: str | None = None,
    epsilon: float = 0.2,
    advantage_eps: float = 1e-4,
) -> dict:
    """Single-checkpoint historical IF (the toy's `--if-calculation historical-last`).

    Per-train-prompt gradient = GRPO loss gradient at θˢ (with old_peft_model=None
    so PPO ratio is 1 and clipping inert).
    Per-train-prompt Fisher contribution = mean-rollout policy-score gradient
    (geometry_feature from POLICY_SCORE mode), weighted equally across prompts.
    One info per training prompt — no rollout-split structure like surrogate.

    `beta` defaults to 0 to mirror the toy convention: the historical gradient
    is computed without the KL term, even when training used β>0. Set higher to
    match training dynamics more precisely.
    """
    n_train = len(train_dataset) if train_limit is None else min(train_limit, len(train_dataset))
    n_test = len(test_dataset) if test_limit is None else min(test_limit, len(test_dataset))

    # ----- Train side: one info per prompt (grad + geometry_feature) -----
    train_infos: list[dict] = []
    train_debug: list[dict] = []
    n_cache_hits = 0
    cache_dir = Path(gradient_cache_dir) if gradient_cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_train):
        sample = train_dataset[i]
        seed = None if seed_base is None else seed_base + i

        bundle_dict: dict | None = None
        cache_path: Path | None = None
        cache_status = "no-cache"
        if cache_dir is not None:
            key = _gradient_cache_key(
                sample["prompt"],
                model_id=model_id,
                final_checkpoint=final_checkpoint_id,
                ref_checkpoint=ref_checkpoint_id,
                G=G, beta=beta, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens, seed=seed, dtype=dtype_id,
            )
            cache_path = cache_dir / f"train_grad_histlast_{key}.pt"
            if cache_path.is_file():
                try:
                    bundle_dict = torch.load(cache_path, map_location="cpu", weights_only=False)
                    cache_status = "hit"
                    n_cache_hits += 1
                except Exception:
                    bundle_dict = None
                    cache_status = "miss"
            else:
                cache_status = "miss"

        if bundle_dict is None:
            reward_funcs = reward_fn_builder(sample, G)
            result = compute_policy_gradient_bundle(
                peft_model, tokenizer, sample["prompt"], reward_funcs,
                G=G, device=device,
                epsilon=epsilon, beta=beta,
                advantage_eps=advantage_eps,
                old_peft_model=None,
                ref_model=ref_model if beta != 0.0 else None,
                objective_mode=GradientObjective.GRPO_TRAIN,
                geometry_feature_mode=GeometryFeatureMode.POLICY_SCORE,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, seed=seed,
                generation_backend=generation_backend, vllm_config=vllm_config,
                adapter_path=adapter_path, model_id=model_id,
            )
            bundle_dict = {
                "grad": result["grad"],
                "geometry_feature": result["geometry_feature"],
                "debug": result.get("debug", {}),
            }
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(bundle_dict, cache_path)

        train_infos.append({
            "grad": bundle_dict["grad"],
            "geometry_feature": bundle_dict["geometry_feature"],
            "historical_weight": 1.0,
            "name": sample.get("name") or f"train_{i}",
        })
        train_debug.append(bundle_dict.get("debug", {}))
        if progress:
            mean_r = float(np.mean(bundle_dict.get("debug", {}).get("total_rewards", [0.0])))
            tag = f"  [{cache_status}]" if cache_dir is not None else ""
            print(
                f"  train {i+1}/{n_train}: ||g_grpo||={bundle_dict['grad'].norm().item():.4f} "
                f"mean_reward={mean_r:.3f}{tag}"
            )

    if cache_dir is not None and progress:
        print(f"  Gradient cache: {n_cache_hits}/{n_train} hits → {cache_dir}")

    # ----- Test side (shared with surrogate) -----
    test_infos: list[dict] = []
    test_debug: list[dict] = []
    for i in range(n_test):
        sample = test_dataset[i]
        name = sample.get("name") or f"test_{i}"
        seed = None if seed_base is None else seed_base + 10_000 + i
        grad, debug, status = _test_grad_for_prompt(
            peft_model, tokenizer, sample,
            test_objective=test_objective,
            device=device, G=G,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed,
            reward_fn_builder=reward_fn_builder,
            generation_backend=generation_backend, vllm_config=vllm_config,
            adapter_path=adapter_path, model_id=model_id,
        )
        if progress:
            print(f"  test  {i+1}/{n_test}: {status}")
        if grad is None:
            continue
        test_infos.append({"grad": grad, "name": name})
        test_debug.append(debug)

    # ----- Fisher solve -----
    checkpoint_infos = [{
        "step": 1, "learning_rate": 1.0,
        "train_infos": train_infos, "test_infos": test_infos,
    }]
    tfi = TrajectoryFisherInfluence(
        lambda_damp=lambda_damp, normalize=False, solver="woodbury",
    )
    matrix, breakdown = tfi.compute_matrix(checkpoint_infos, return_breakdown=True)
    # No stride trick — 1 train_info per prompt, in order.
    scores = np.asarray(matrix)[:, :n_train]

    return {
        "scores": scores,
        "matrix_raw": np.asarray(matrix),
        "breakdown": breakdown,
        "train_debug": train_debug,
        "test_debug": test_debug,
        "stride": 1,
        "n_train": n_train,
        "n_test": n_test,
        "method": "historical-last",
    }
