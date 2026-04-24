from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attribution.fisher import TrajectoryFisherInfluence
from .gradients import (
    _compute_expected_reward_policy_loss,
    _compute_grpo_policy_loss,
    _grad_vector_from_scalar,
)
from .modes import GeometryFeatureMode, GradientObjective


class ToyRolloutMode(str, Enum):
    EXHAUSTIVE = "exhaustive"
    SAMPLED = "sampled"

    @classmethod
    def parse(cls, value: ToyRolloutMode | str) -> ToyRolloutMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass(frozen=True)
class ToyGRPOExample:
    name: str
    z: tuple[int, int, int]
    target: tuple[int, int]
    split: str = "train"
    expected_influence: str | None = None

    def z_tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(self.z, dtype=torch.float32, device=device)

    def target_tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(self.target, dtype=torch.long, device=device)


@dataclass(frozen=True)
class ToySandboxDataset:
    train_examples: tuple[ToyGRPOExample, ...]
    test_example: ToyGRPOExample


@dataclass(frozen=True)
class ToyInfluenceScore:
    train_name: str
    expected_influence: str | None
    repo_fisher_score: float
    loss_influence: float
    predicted_loss_delta: float
    actual_loss_delta: float | None = None
    predicted_reward_delta: float | None = None
    actual_reward_delta: float | None = None


@dataclass(frozen=True)
class ToyHistoricalInfluenceSummary:
    train_name: str
    expected_influence: str | None
    occurrence_count: int
    repo_fisher_score: float
    loss_influence: float


_ALL_TWO_TOKEN_SEQUENCES = torch.tensor(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    dtype=torch.long,
)


def build_user_plan_sandbox() -> ToySandboxDataset:
    """Create the dataset"""
    return ToySandboxDataset(
        train_examples=(
            ToyGRPOExample(
                name="helpful_feature_a",
                z=(1, 0, 0),
                target=(1, 0),
                expected_influence="negative",
            ),
            ToyGRPOExample(
                name="harmful_shared_noise",
                z=(0, 1, 1),
                target=(0, 1),
                expected_influence="positive",
            ),
            ToyGRPOExample(
                name="neutral_feature_b_only",
                z=(0, 1, 0),
                target=(0, 1),
                expected_influence="approximately_zero",
            ),
        ),
        test_example=ToyGRPOExample(
            name="test_feature_a_with_noise",
            z=(1, 0, 1),
            target=(1, 0),
            split="test",
        ),
    )


class AutoregressiveLogisticRegression(nn.Module):
    def __init__(self, *, use_bias: bool = False):
        super().__init__()
        self.first = nn.Linear(3, 2, bias=use_bias)
        self.second = nn.Linear(4, 2, bias=use_bias)

    def first_token_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.first(z)

    def second_token_logits(
        self,
        z: torch.Tensor,
        first_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if first_tokens.ndim == 1:
            first_tokens = first_tokens.unsqueeze(1)
        aug = torch.cat([z, first_tokens.to(dtype=z.dtype)], dim=1)
        return self.second(aug)

    def per_token_log_probs(
        self,
        z: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(dtype=torch.float32)
        n_seq = int(sequences.shape[0])
        z_rep = z.expand(n_seq, -1)
        first_logits = self.first_token_logits(z_rep)
        first_log_probs = F.log_softmax(first_logits, dim=-1)
        first_tokens = sequences[:, 0]
        first_lp = first_log_probs.gather(1, first_tokens.unsqueeze(1)).squeeze(1)

        second_logits = self.second_token_logits(z_rep, first_tokens.float())
        second_log_probs = F.log_softmax(second_logits, dim=-1)
        second_tokens = sequences[:, 1]
        second_lp = second_log_probs.gather(1, second_tokens.unsqueeze(1)).squeeze(1)

        return torch.stack([first_lp, second_lp], dim=1)

    def sequence_log_probs(
        self,
        z: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        return self.per_token_log_probs(z, sequences).sum(dim=1)

    def exact_sequence_distribution(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = _ALL_TWO_TOKEN_SEQUENCES.to(self.device)
        sequence_log_probs = self.sequence_log_probs(z.to(self.device), sequences)
        return sequences, sequence_log_probs.exp()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def initialize_toy_model(
    model: AutoregressiveLogisticRegression,
    *,
    mode: str = "zero",
    seed: int = 0,
    scale: float = 0.05,
) -> AutoregressiveLogisticRegression:
    if mode == "zero":
        for param in model.parameters():
            param.data.zero_()
        return model
    if mode != "normal":
        raise ValueError(f"Unsupported init mode: {mode!r}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for param in model.parameters():
        param.data.normal_(mean=0.0, std=scale, generator=generator)
    return model


def clone_toy_model(
    model: AutoregressiveLogisticRegression,
) -> AutoregressiveLogisticRegression:
    cloned = copy.deepcopy(model)
    cloned.eval()
    return cloned


def flatten_trainable_parameters(model: nn.Module) -> torch.Tensor:
    parts = [
        param.detach().reshape(-1).to(dtype=torch.float32)
        for param in model.parameters()
        if param.requires_grad
    ]
    if not parts:
        raise RuntimeError("Toy model has no trainable parameters.")
    return torch.cat(parts)


def assign_flat_trainable_parameters(model: nn.Module, flat_vector: torch.Tensor) -> None:
    offset = 0
    flat_vector = flat_vector.to(device=model.device, dtype=torch.float32)
    for param in model.parameters():
        if not param.requires_grad:
            continue
        size = param.numel()
        chunk = flat_vector[offset : offset + size].view_as(param)
        param.data.copy_(chunk)
        offset += size
    if offset != flat_vector.numel():
        raise ValueError("Flat parameter vector length does not match model parameters.")


def reward_for_sequences(
    sequences: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gives reward if the sequence exactly matches the target, and 0 otherwise."""
    return (sequences == target.unsqueeze(0)).all(dim=1).to(dtype=torch.float32)


def sequence_labels(sequences: torch.Tensor) -> list[str]:
    return ["".join(str(int(token)) for token in row.tolist()) for row in sequences]


def exact_expected_reward(
    model: AutoregressiveLogisticRegression,
    example: ToyGRPOExample,
) -> float:
    sequences, probs = model.exact_sequence_distribution(example.z_tensor(device=model.device))
    rewards = reward_for_sequences(sequences, example.target_tensor(device=model.device))
    return float((probs * rewards).sum().detach().cpu())


def rollout_token_sequences(
    model: AutoregressiveLogisticRegression,
    example: ToyGRPOExample,
    *,
    G: int = 4,
    rollout_mode: ToyRolloutMode | str = ToyRolloutMode.EXHAUSTIVE,
    seed: int | None = None,
) -> torch.Tensor:
    rollout_mode = ToyRolloutMode.parse(rollout_mode)
    if G < 1:
        raise ValueError(f"G must be >= 1, got {G}.")

    if rollout_mode == ToyRolloutMode.EXHAUSTIVE:
        if G != 4:
            raise ValueError("Exhaustive toy rollout requires G=4 for the four length-2 binary sequences.")
        return _ALL_TWO_TOKEN_SEQUENCES.to(model.device)

    z = example.z_tensor(device=model.device)
    first_logits = model.first_token_logits(z.unsqueeze(0)).squeeze(0)
    first_dist = torch.distributions.Categorical(logits=first_logits)
    if seed is not None:
        torch.manual_seed(seed)
    first_tokens = first_dist.sample((G,))
    z_rep = z.unsqueeze(0).expand(G, -1)
    second_logits = model.second_token_logits(z_rep, first_tokens.float())
    second_dist = torch.distributions.Categorical(logits=second_logits)
    second_tokens = second_dist.sample()
    return torch.stack([first_tokens, second_tokens], dim=1)


def _toy_objective_and_debug(
    model: AutoregressiveLogisticRegression,
    example: ToyGRPOExample,
    *,
    G: int,
    rollout_mode: ToyRolloutMode | str,
    seed: int | None,
    epsilon: float,
    beta: float,
    old_model: AutoregressiveLogisticRegression | None,
    ref_model: AutoregressiveLogisticRegression | None,
    advantage_eps: float,
    objective_mode: GradientObjective | str,
):
    """Compute the training objective for a single example, along with debug information."""
    objective_mode = GradientObjective.parse(objective_mode)
    response_ids = rollout_token_sequences(
        model,
        example,
        G=G,
        rollout_mode=rollout_mode,
        seed=seed,
    )
    response_mask = torch.ones_like(response_ids, dtype=torch.long)
    z = example.z_tensor(device=model.device)
    target = example.target_tensor(device=model.device)

    per_token_logps = model.per_token_log_probs(z, response_ids)
    if old_model is None:
        old_per_token_logps = per_token_logps.detach()
    else:
        with torch.no_grad():
            old_per_token_logps = old_model.per_token_log_probs(
                example.z_tensor(device=old_model.device),
                response_ids.to(old_model.device),
            ).to(model.device)

    ref_per_token_logps = None
    if beta != 0.0:
        if ref_model is None:
            raise ValueError("beta != 0.0 requires a reference model in the toy sandbox.")
        with torch.no_grad():
            ref_per_token_logps = ref_model.per_token_log_probs(
                example.z_tensor(device=ref_model.device),
                response_ids.to(ref_model.device),
            ).to(model.device)

    total_rewards = reward_for_sequences(response_ids, target)
    token_mask = response_mask.float()
    token_counts = token_mask.sum(dim=1).clamp(min=1.0)
    sequence_log_probs = (per_token_logps * token_mask).sum(dim=1)

    if objective_mode == GradientObjective.GRPO_TRAIN:
        objective, advantages, per_token_kl = _compute_grpo_policy_loss(
            total_rewards,
            per_token_logps,
            old_per_token_logps,
            response_mask,
            epsilon=epsilon,
            beta=beta,
            ref_per_token_logps=ref_per_token_logps,
            advantage_eps=advantage_eps,
        )
        objective_name = "grpo_policy_loss"
    elif objective_mode == GradientObjective.EXPECTED_REWARD_PG:
        objective = _compute_expected_reward_policy_loss(total_rewards, sequence_log_probs)
        advantages = total_rewards.detach()
        per_token_kl = torch.zeros_like(per_token_logps)
        objective_name = "expected_reward_pg_loss"
    else:
        raise ValueError(f"Unsupported toy objective mode: {objective_mode!r}")

    debug = {
        "example_name": example.name,
        "z": list(example.z),
        "target": list(example.target),
        "responses": sequence_labels(response_ids),
        "total_rewards": total_rewards.detach().cpu().tolist(),
        "advantages": advantages.detach().cpu().tolist(),
        "sequence_log_probs": sequence_log_probs.detach().cpu().tolist(),
        "response_lengths": response_mask.sum(dim=1).detach().cpu().tolist(),
        "policy_loss": float(objective.detach().cpu()),
        "mean_kl": float(((per_token_kl * token_mask).sum(dim=1) / token_counts).mean().detach().cpu()),
        "rollout_mode": ToyRolloutMode.parse(rollout_mode).value,
        "objective_mode": objective_mode.value,
        "objective_name": objective_name,
        "epsilon": epsilon,
        "beta": beta,
        "seed": seed,
    }
    return objective, sequence_log_probs, debug


def compute_toy_gradient_bundle(
    model: AutoregressiveLogisticRegression,
    example: ToyGRPOExample,
    *,
    G: int = 4,
    rollout_mode: ToyRolloutMode | str = ToyRolloutMode.EXHAUSTIVE,
    seed: int | None = None,
    epsilon: float = 0.2,
    beta: float = 0.0,
    old_model: AutoregressiveLogisticRegression | None = None,
    ref_model: AutoregressiveLogisticRegression | None = None,
    advantage_eps: float = 1e-4,
    objective_mode: GradientObjective | str = GradientObjective.GRPO_TRAIN,
    geometry_feature_mode: GeometryFeatureMode | str = GeometryFeatureMode.NONE,
) -> dict:
    geometry_feature_mode = GeometryFeatureMode.parse(geometry_feature_mode)
    objective, sequence_log_probs, debug = _toy_objective_and_debug(
        model,
        example,
        G=G,
        rollout_mode=rollout_mode,
        seed=seed,
        epsilon=epsilon,
        beta=beta,
        old_model=old_model,
        ref_model=ref_model,
        advantage_eps=advantage_eps,
        objective_mode=objective_mode,
    )

    retain_graph = geometry_feature_mode != GeometryFeatureMode.NONE
    grad_vector = _grad_vector_from_scalar(model, objective, retain_graph=retain_graph)
    geometry_feature = None
    if geometry_feature_mode == GeometryFeatureMode.POLICY_SCORE:
        geometry_feature = _grad_vector_from_scalar(
            model,
            sequence_log_probs.mean(),
            retain_graph=False,
        )
    elif geometry_feature_mode != GeometryFeatureMode.NONE:
        raise ValueError(f"Unsupported toy geometry_feature_mode: {geometry_feature_mode!r}")

    if geometry_feature is not None:
        debug["geometry_feature_norm"] = float(geometry_feature.norm().item())
    return {
        "grad": grad_vector,
        "geometry_feature": geometry_feature,
        "debug": debug,
    }


def train_toy_grpo(
    model: AutoregressiveLogisticRegression,
    dataset: Sequence[ToyGRPOExample],
    *,
    steps: int = 12,
    lr: float = 0.25,
    G: int = 4,
    rollout_mode: ToyRolloutMode | str = ToyRolloutMode.EXHAUSTIVE,
    epsilon: float = 0.2,
    beta: float = 0.0,
    advantage_eps: float = 1e-4,
    checkpoint_steps: Iterable[int] = (0,),
    seed: int = 0,
) -> dict:
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")
    if not dataset:
        raise ValueError("Toy GRPO training dataset must be non-empty.")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    checkpoints = {}
    requested_checkpoints = {int(step) for step in checkpoint_steps}
    if 0 in requested_checkpoints:
        checkpoints[0] = copy.deepcopy(model.state_dict())

    history = []
    for step in range(1, steps + 1):
        example = dataset[(step - 1) % len(dataset)]
        old_model = clone_toy_model(model)
        optimizer.zero_grad()
        objective, _, debug = _toy_objective_and_debug(
            model,
            example,
            G=G,
            rollout_mode=rollout_mode,
            seed=seed + step - 1 if ToyRolloutMode.parse(rollout_mode) == ToyRolloutMode.SAMPLED else None,
            epsilon=epsilon,
            beta=beta,
            old_model=old_model,
            ref_model=None,
            advantage_eps=advantage_eps,
            objective_mode=GradientObjective.GRPO_TRAIN,
        )
        objective.backward()
        optimizer.step()

        history.append(
            {
                "step": step,
                "example_name": example.name,
                "loss": float(objective.detach().cpu()),
                "exact_expected_reward": exact_expected_reward(model, example),
                "debug": debug,
            }
        )
        if step in requested_checkpoints:
            checkpoints[step] = copy.deepcopy(model.state_dict())

    return {
        "model": model,
        "history": history,
        "checkpoints": checkpoints,
    }


def dense_policy_score_fisher(
    train_infos: Sequence[dict],
    *,
    lambda_damp: float,
) -> torch.Tensor:
    if not train_infos:
        raise ValueError("dense_policy_score_fisher requires at least one train info.")
    dim = int(train_infos[0]["geometry_feature"].numel())
    fisher = lambda_damp * torch.eye(dim, dtype=torch.float32)
    raw_weights = torch.tensor(
        [float(info.get("historical_weight", 0.0)) for info in train_infos],
        dtype=torch.float32,
    )
    if torch.any(raw_weights > 0):
        weights = raw_weights / raw_weights.sum().clamp(min=1e-12)
    else:
        weights = torch.full((len(train_infos),), 1.0 / len(train_infos), dtype=torch.float32)
    for weight, info in zip(weights, train_infos):
        feature = info["geometry_feature"].to(dtype=torch.float32)
        fisher = fisher + weight * torch.outer(feature, feature)
    return fisher


def compute_toy_fisher_influence(
    model: AutoregressiveLogisticRegression,
    *,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    lambda_damp: float = 0.1,
    G: int = 4,
    rollout_mode: ToyRolloutMode | str = ToyRolloutMode.EXHAUSTIVE,
    epsilon: float = 0.2,
    beta: float = 0.0,
    advantage_eps: float = 1e-4,
    seed: int = 0,
) -> dict:
    train_infos = []
    for idx, example in enumerate(train_examples):
        old_model = clone_toy_model(model)
        train_infos.append(
            {
                **compute_toy_gradient_bundle(
                    model,
                    example,
                    G=G,
                    rollout_mode=rollout_mode,
                    seed=seed + idx if ToyRolloutMode.parse(rollout_mode) == ToyRolloutMode.SAMPLED else None,
                    epsilon=epsilon,
                    beta=beta,
                    old_model=old_model,
                    ref_model=None,
                    advantage_eps=advantage_eps,
                    objective_mode=GradientObjective.GRPO_TRAIN,
                    geometry_feature_mode=GeometryFeatureMode.POLICY_SCORE,
                ),
                "name": example.name,
                "expected_influence": example.expected_influence,
            }
        )

    test_info = {
        **compute_toy_gradient_bundle(
            model,
            test_example,
            G=G,
            rollout_mode=rollout_mode,
            seed=seed + 999 if ToyRolloutMode.parse(rollout_mode) == ToyRolloutMode.SAMPLED else None,
            epsilon=epsilon,
            beta=beta,
            old_model=None,
            ref_model=None,
            advantage_eps=advantage_eps,
            objective_mode=GradientObjective.EXPECTED_REWARD_PG,
            geometry_feature_mode=GeometryFeatureMode.NONE,
        ),
        "name": test_example.name,
    }

    checkpoint_infos = [
        {
            "step": 0,
            "learning_rate": 1.0,
            "train_infos": train_infos,
            "test_infos": [test_info],
        }
    ]
    trajectory_fisher = TrajectoryFisherInfluence(
        lambda_damp=lambda_damp,
        normalize=False,
    )
    matrix, breakdown = trajectory_fisher.compute_matrix(
        checkpoint_infos,
        return_breakdown=True,
    )
    repo_scores = matrix[0]
    dense_fisher = dense_policy_score_fisher(train_infos, lambda_damp=lambda_damp)
    g_test = test_info["grad"].to(dtype=torch.float32)
    dense_h_inv_g = torch.linalg.solve(dense_fisher, g_test)
    dense_repo_scores = torch.tensor(
        [
            float(torch.dot(info["grad"].to(dtype=torch.float32), dense_h_inv_g).item())
            for info in train_infos
        ],
        dtype=torch.float32,
    )

    return {
        "train_infos": train_infos,
        "test_info": test_info,
        "trajectory_fisher_matrix": matrix,
        "trajectory_fisher_breakdown": breakdown,
        "repo_scores": repo_scores,
        "loss_influence_scores": -repo_scores,
        "dense_repo_scores": dense_repo_scores.numpy(),
        "dense_fisher": dense_fisher,
    }


def validate_toy_influence_with_preconditioned_step(
    model: AutoregressiveLogisticRegression,
    *,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    lambda_damp: float = 0.1,
    step_scale: float = 0.05,
    G: int = 4,
    rollout_mode: ToyRolloutMode | str = ToyRolloutMode.EXHAUSTIVE,
    epsilon: float = 0.2,
    beta: float = 0.0,
    advantage_eps: float = 1e-4,
    seed: int = 0,
) -> dict:
    influence = compute_toy_fisher_influence(
        model,
        train_examples=train_examples,
        test_example=test_example,
        lambda_damp=lambda_damp,
        G=G,
        rollout_mode=rollout_mode,
        epsilon=epsilon,
        beta=beta,
        advantage_eps=advantage_eps,
        seed=seed,
    )
    dense_fisher = influence["dense_fisher"]
    base_loss = float(influence["test_info"]["debug"]["policy_loss"])
    base_reward = exact_expected_reward(model, test_example)
    theta = flatten_trainable_parameters(model)

    rows = []
    for idx, (example, train_info) in enumerate(zip(train_examples, influence["train_infos"])):
        train_grad = train_info["grad"].to(dtype=torch.float32)
        delta = -step_scale * torch.linalg.solve(dense_fisher, train_grad)
        updated = clone_toy_model(model)
        assign_flat_trainable_parameters(updated, theta + delta)

        updated_test = compute_toy_gradient_bundle(
            updated,
            test_example,
            G=G,
            rollout_mode=rollout_mode,
            seed=seed + 999 if ToyRolloutMode.parse(rollout_mode) == ToyRolloutMode.SAMPLED else None,
            epsilon=epsilon,
            beta=beta,
            old_model=None,
            ref_model=None,
            advantage_eps=advantage_eps,
            objective_mode=GradientObjective.EXPECTED_REWARD_PG,
            geometry_feature_mode=GeometryFeatureMode.NONE,
        )
        updated_reward = exact_expected_reward(updated, test_example)
        predicted_loss_delta = step_scale * float(influence["loss_influence_scores"][idx])
        actual_loss_delta = float(updated_test["debug"]["policy_loss"]) - base_loss
        predicted_reward_delta = -predicted_loss_delta
        actual_reward_delta = updated_reward - base_reward
        rows.append(
            ToyInfluenceScore(
                train_name=example.name,
                expected_influence=example.expected_influence,
                repo_fisher_score=float(influence["repo_scores"][idx]),
                loss_influence=float(influence["loss_influence_scores"][idx]),
                predicted_loss_delta=predicted_loss_delta,
                actual_loss_delta=actual_loss_delta,
                predicted_reward_delta=predicted_reward_delta,
                actual_reward_delta=actual_reward_delta,
            )
        )
    return {
        **influence,
        "base_test_loss": base_loss,
        "base_test_expected_reward": base_reward,
        "validated_scores": rows,
    }


def compute_toy_test_loss(
    model: AutoregressiveLogisticRegression,
    test_example: ToyGRPOExample,
    *,
    G: int = 4,
    rollout_mode: ToyRolloutMode | str = ToyRolloutMode.EXHAUSTIVE,
    epsilon: float = 0.2,
    beta: float = 0.0,
    advantage_eps: float = 1e-4,
    seed: int | None = None,
) -> float:
    bundle = compute_toy_gradient_bundle(
        model,
        test_example,
        G=G,
        rollout_mode=rollout_mode,
        seed=seed,
        epsilon=epsilon,
        beta=beta,
        old_model=None,
        ref_model=None,
        advantage_eps=advantage_eps,
        objective_mode=GradientObjective.EXPECTED_REWARD_PG,
        geometry_feature_mode=GeometryFeatureMode.NONE,
    )
    return float(bundle["debug"]["policy_loss"])


def compute_toy_historical_fisher_influence(
    model_template: AutoregressiveLogisticRegression,
    *,
    checkpoints: dict[int, dict[str, torch.Tensor]],
    train_history: Sequence[dict],
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    learning_rate: float,
    end_step: int | None = None,
    lambda_damp: float = 0.1,
    G: int = 4,
    rollout_mode: ToyRolloutMode | str = ToyRolloutMode.EXHAUSTIVE,
    epsilon: float = 0.2,
    beta: float = 0.0,
    advantage_eps: float = 1e-4,
    seed: int = 0,
) -> dict:
    if end_step is None:
        end_step = max((int(row["step"]) for row in train_history), default=0)
    if end_step < 0:
        raise ValueError(f"end_step must be >= 0, got {end_step}.")

    train_example_by_name = {example.name: example for example in train_examples}
    truncated_history = [row for row in train_history if int(row["step"]) <= end_step]
    checkpoint_infos = []

    for row in truncated_history:
        step = int(row["step"])
        pre_step = step - 1
        if pre_step not in checkpoints:
            raise ValueError(
                f"Historical toy influence needs checkpoint {pre_step}, but it was not saved."
            )
        used_example_name = str(row["example_name"])
        if used_example_name not in train_example_by_name:
            raise ValueError(f"Unknown train example in history: {used_example_name!r}")

        checkpoint_model = clone_toy_model(model_template)
        checkpoint_model.load_state_dict(checkpoints[pre_step])

        train_infos = []
        for idx, example in enumerate(train_examples):
            old_model = clone_toy_model(checkpoint_model)
            train_infos.append(
                {
                    **compute_toy_gradient_bundle(
                        checkpoint_model,
                        example,
                        G=G,
                        rollout_mode=rollout_mode,
                        seed=(
                            seed + 1000 * step + idx
                            if ToyRolloutMode.parse(rollout_mode) == ToyRolloutMode.SAMPLED
                            else None
                        ),
                        epsilon=epsilon,
                        beta=beta,
                        old_model=old_model,
                        ref_model=None,
                        advantage_eps=advantage_eps,
                        objective_mode=GradientObjective.GRPO_TRAIN,
                        geometry_feature_mode=GeometryFeatureMode.POLICY_SCORE,
                    ),
                    "name": example.name,
                    "expected_influence": example.expected_influence,
                    "historical_weight": 1.0 if example.name == used_example_name else 0.0,
                }
            )

        test_info = {
            **compute_toy_gradient_bundle(
                checkpoint_model,
                test_example,
                G=G,
                rollout_mode=rollout_mode,
                seed=(
                    seed + 200000 + step
                    if ToyRolloutMode.parse(rollout_mode) == ToyRolloutMode.SAMPLED
                    else None
                ),
                epsilon=epsilon,
                beta=beta,
                old_model=None,
                ref_model=None,
                advantage_eps=advantage_eps,
                objective_mode=GradientObjective.EXPECTED_REWARD_PG,
                geometry_feature_mode=GeometryFeatureMode.NONE,
            ),
            "name": test_example.name,
        }
        checkpoint_infos.append(
            {
                "step": step,
                "learning_rate": float(learning_rate),
                "train_infos": train_infos,
                "test_infos": [test_info],
            }
        )

    trajectory_fisher = TrajectoryFisherInfluence(
        lambda_damp=lambda_damp,
        normalize=False,
    )
    matrix, breakdown = trajectory_fisher.compute_matrix(
        checkpoint_infos,
        return_breakdown=True,
    )
    repo_scores = matrix[0] if len(checkpoint_infos) > 0 else torch.zeros(len(train_examples)).numpy()

    occurrence_count = {example.name: 0 for example in train_examples}
    for row in truncated_history:
        occurrence_count[str(row["example_name"])] += 1

    initial_model = clone_toy_model(model_template)
    initial_model.load_state_dict(checkpoints[0])
    final_model = clone_toy_model(model_template)
    final_model.load_state_dict(checkpoints[end_step])

    sampled_mode = ToyRolloutMode.parse(rollout_mode) == ToyRolloutMode.SAMPLED
    initial_seed = seed + 900000 if sampled_mode else None
    final_seed = seed + 900000 + end_step if sampled_mode else None

    initial_test_loss = compute_toy_test_loss(
        initial_model,
        test_example,
        G=G,
        rollout_mode=rollout_mode,
        epsilon=epsilon,
        beta=beta,
        advantage_eps=advantage_eps,
        seed=initial_seed,
    )
    final_test_loss = compute_toy_test_loss(
        final_model,
        test_example,
        G=G,
        rollout_mode=rollout_mode,
        epsilon=epsilon,
        beta=beta,
        advantage_eps=advantage_eps,
        seed=final_seed,
    )
    initial_test_reward = exact_expected_reward(initial_model, test_example)
    final_test_reward = exact_expected_reward(final_model, test_example)

    rows = []
    for example, repo_score in zip(train_examples, repo_scores):
        rows.append(
            ToyHistoricalInfluenceSummary(
                train_name=example.name,
                expected_influence=example.expected_influence,
                occurrence_count=occurrence_count[example.name],
                repo_fisher_score=float(repo_score),
                loss_influence=float(-repo_score),
            )
        )

    return {
        "checkpoint_infos": checkpoint_infos,
        "trajectory_fisher_breakdown": breakdown,
        "repo_scores": repo_scores,
        "loss_influence_scores": -repo_scores,
        "historical_scores": rows,
        "predicted_total_loss_delta": float((-repo_scores).sum()),
        "predicted_total_reward_delta": float(repo_scores.sum()),
        "initial_test_loss": initial_test_loss,
        "final_test_loss": final_test_loss,
        "actual_total_loss_delta": float(final_test_loss - initial_test_loss),
        "initial_test_expected_reward": initial_test_reward,
        "final_test_expected_reward": final_test_reward,
        "actual_total_reward_delta": float(final_test_reward - initial_test_reward),
    }
