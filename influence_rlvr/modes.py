from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

EnumT = TypeVar("EnumT", bound="StringEnum")


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls: type[EnumT], value: EnumT | str) -> EnumT:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class InfluenceMode(StringEnum):
    HISTORICAL = "historical"
    DENSE = "dense"


class ExperimentMode(StringEnum):
    MATH_GRPO = "math_grpo"
    CODE_GRPO = "code_grpo"
    BASE_EVAL = "base_eval"


class GenerationBackend(StringEnum):
    HF = "hf"
    VLLM = "vllm"


class GradientObjective(StringEnum):
    GRPO_TRAIN = "grpo_train"
    EXPECTED_REWARD_PG = "expected_reward_pg"


class GeometryFeatureMode(StringEnum):
    NONE = "none"
    POLICY_SCORE = "policy_score"


class SecondOrderGeometry(StringEnum):
    DATAINF = "datainf"
    POLICY_SCORE_FISHER = "policy_score_fisher"


@dataclass(frozen=True)
class CodeEvalConfig:
    do_sample: bool = False
    num_samples: int = 1
    temperature: float = 0.6
    top_p: float = 0.95

    def to_kwargs(self) -> dict[str, object]:
        return {
            "code_eval_do_sample": self.do_sample,
            "code_eval_num_samples": self.num_samples,
            "code_eval_temperature": self.temperature,
            "code_eval_top_p": self.top_p,
        }

    def to_config_dict(self) -> dict[str, object]:
        return {
            "code_eval_do_sample": self.do_sample,
            "code_eval_num_samples": self.num_samples,
            "code_eval_temperature": self.temperature,
            "code_eval_top_p": self.top_p,
        }


@dataclass(frozen=True)
class VLLMConfig:
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool = False
    training_use_vllm: bool = False

    def to_runtime_kwargs(self) -> dict[str, object]:
        return {
            "vllm_gpu_memory_utilization": self.gpu_memory_utilization,
            "vllm_tensor_parallel_size": self.tensor_parallel_size,
            "vllm_max_model_len": self.max_model_len,
            "vllm_max_num_seqs": self.max_num_seqs,
            "vllm_enforce_eager": self.enforce_eager,
        }

    def to_config_dict(self) -> dict[str, object]:
        return {
            **self.to_runtime_kwargs(),
            "vllm_training_use_vllm": self.training_use_vllm,
        }


@dataclass(frozen=True)
class ReplayGradientConfig:
    train_objective: GradientObjective = GradientObjective.GRPO_TRAIN
    test_objective: GradientObjective = GradientObjective.GRPO_TRAIN
    train_geometry_feature: GeometryFeatureMode = GeometryFeatureMode.NONE
    second_order_geometry: SecondOrderGeometry = SecondOrderGeometry.DATAINF
    fisher_normalize: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    def to_kwargs(self) -> dict[str, object]:
        return {
            "train_gradient_objective_mode": self.train_objective,
            "test_gradient_objective_mode": self.test_objective,
            "train_geometry_feature_mode": self.train_geometry_feature,
            "replay_max_new_tokens": self.max_new_tokens,
            "replay_temperature": self.temperature,
            "replay_top_p": self.top_p,
        }

    def to_config_dict(self) -> dict[str, object]:
        return {
            "train_gradient_objective": self.train_objective,
            "test_gradient_objective": self.test_objective,
            "train_geometry_feature": self.train_geometry_feature,
            "second_order_geometry": self.second_order_geometry,
            "fisher_normalize": self.fisher_normalize,
            "replay_max_new_tokens": self.max_new_tokens,
            "replay_temperature": self.temperature,
            "replay_top_p": self.top_p,
        }
