"""Per-train influence at a single checkpoint, for data pruning.

Given the policy at checkpoint `step`, we score every example in the training
pool by its (aggregated) influence on a held-out target set:

    score[i] = mean_over_target_j  IF(train_i -> target_j)

The heavy lifting (rollouts, per-example policy gradients, second-order geometry)
is delegated to the validated `collect_checkpoint_infos` replay path, exactly as
in the historical pipeline — we just point it at one checkpoint and use the
verifier as the reward signal.

v1 uses `TrajectoryFisherInfluence` (damped policy-Fisher inverse), which is the
already-wired LLM-scale second-order influence. `if_method="cg"` is reserved for
the follow-up that feeds the same per-completion policy-score gradients into
`CGInfluence` via the cached-gradient FVP (`policy_fisher_fvp_from_grad_cache`).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from .config import ExperimentConfig
from .verifier import get_verifier_from_config, _student_answer
from influence_rlvr import (
    GenerationBackend,
    TrajectoryFisherInfluence,
    collect_checkpoint_infos,
)
from influence_rlvr.modes import (
    GeometryFeatureMode,
    GradientObjective,
    InfluenceMode,
    ReplayGradientConfig,
    SecondOrderGeometry,
    VLLMConfig,
)


def _make_verifier_reward_builder(cfg: ExperimentConfig):
    """reward_fn_builder(sample, num_generations) -> [reward_fn(completions)].

    The returned reward fn scores the G completions of a *single* sample against
    that sample's reference answer using the general-verifier.
    """
    def builder(sample: dict, num_generations: int):
        question = sample["question"]
        gold = sample["solution"]

        def reward_fn(completions):
            responses = [c[0]["content"] for c in completions]
            students = [_student_answer(r) for r in responses]
            n = len(students)
            verifier = get_verifier_from_config(cfg)
            return verifier.verify_batch([question] * n, [gold] * n, students)

        reward_fn.__name__ = "verifier_reward_single"
        return [reward_fn]

    return builder


def _replay_config(cfg: ExperimentConfig) -> ReplayGradientConfig:
    return ReplayGradientConfig(
        train_objective=GradientObjective.GRPO_TRAIN,
        test_objective=GradientObjective.EXPECTED_REWARD_PG,
        train_geometry_feature=GeometryFeatureMode.POLICY_SCORE,
        second_order_geometry=SecondOrderGeometry.POLICY_SCORE_FISHER,
        fisher_normalize=False,
        max_new_tokens=cfg.if_max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        replay_gradient_batch_size=1,
    )


def compute_pool_influence(
    cfg: ExperimentConfig,
    peft_model,
    tokenizer,
    train_pool: Dataset,
    target_set: Dataset,
    device,
    *,
    checkpoint_step: int,
    save_dir: Path | None = None,
) -> np.ndarray:
    """Return per-train aggregated influence scores (1-D, len == len(train_pool))."""
    if cfg.if_method in ("cg", "cg-empirical", "dot", "tracin-adam"):
        # CG influence: "cg" = matrix-free analytic per-token Fisher;
        # "cg-empirical" = cached sampled-Fisher FVP; "dot" = first-order
        # gradient dot product (TracIn-style, no Fisher/solve); "tracin-adam" =
        # dot preconditioned by Adam's diagonal 1/(√v̂+ε) from optimizer.pt.
        from .cg_influence import compute_cg_pool_influence
        return compute_cg_pool_influence(
            cfg, peft_model, tokenizer, train_pool, target_set, device,
            checkpoint_step=checkpoint_step, save_dir=save_dir,
        )

    checkpoint_dir = cfg.grpo_output_dir / f"checkpoint-{checkpoint_step}"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Expected checkpoint at {checkpoint_dir}. Save it before scoring "
            f"(save_steps must hit step {checkpoint_step})."
        )

    # Single-checkpoint, dense (counterfactual) schedule — no batch history needed.
    checkpoint_schedule = [{
        "step": checkpoint_step,
        "path": str(checkpoint_dir),
        "learning_rate": cfg.learning_rate,
    }]

    reward_builder = _make_verifier_reward_builder(cfg)
    replay_cfg = _replay_config(cfg)
    vllm_cfg = VLLMConfig(
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        max_model_len=cfg.vllm_max_model_len,
        max_lora_rank=cfg.lora_r,
    )

    checkpoint_infos = collect_checkpoint_infos(
        peft_model,
        tokenizer,
        checkpoint_schedule,
        target_set,           # test_dataset == IF target set
        train_pool,           # train_dataset == pool we score
        device,
        reward_fn_builder=reward_builder,
        G=cfg.if_g_train,
        enable_vllm=cfg.use_vllm,
        generation_backend=GenerationBackend.VLLM if cfg.use_vllm else GenerationBackend.HF,
        test_limit=len(target_set),
        train_limit=len(train_pool),
        include_debug=False,
        base_seed=cfg.seed,
        test_reward_fn_builder=reward_builder,
        test_G=cfg.if_g_train,
        epsilon=cfg.grpo_epsilon,
        beta=cfg.grpo_beta,
        influence_mode=InfluenceMode.DENSE,
        eval_max_new_tokens=cfg.eval_max_new_tokens,
        vllm_config=vllm_cfg,
        model_id=cfg.model_id,
        lambda_damp=cfg.lambda_damp,
        fisher_lambda_damp=cfg.lambda_damp,
        fisher_normalize=False,
        **replay_cfg.to_kwargs(),
    )

    fisher = TrajectoryFisherInfluence(lambda_damp=cfg.lambda_damp, normalize=False)
    matrix, _ = fisher.compute_matrix(checkpoint_infos, return_breakdown=True)
    # matrix: (n_target, n_train). Aggregate to a per-train score.
    scores = np.asarray(matrix, dtype=np.float64).mean(axis=0)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"if_matrix_step{checkpoint_step}.npy", matrix)
        np.save(save_dir / f"if_scores_step{checkpoint_step}.npy", scores)
        print(f"  Saved influence artifacts under {save_dir}/")

    return scores


def select_keep_indices(
    scores: np.ndarray,
    keep_fraction: float,
    *,
    selection: str,
    seed: int,
) -> list[int]:
    """Pick which train indices to keep for the post-prune training phase."""
    n = len(scores)
    k = max(1, int(round(keep_fraction * n)))
    if selection == "if-guided":
        order = np.argsort(-scores)          # most helpful first
        return sorted(int(i) for i in order[:k])
    if selection == "anti-if":
        order = np.argsort(scores)           # most harmful first (ablation)
        return sorted(int(i) for i in order[:k])
    if selection == "random":
        rng = np.random.default_rng(seed)
        return sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    raise ValueError(f"Unknown selection {selection!r}")
