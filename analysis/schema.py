from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_SCHEMA_VERSION = 1
GRAD_CACHE_SCHEMA_VERSION = 1
RESULTS_MANIFEST_FILE = "results_manifest.json"
LEGACY_RESULTS_METADATA_FILE = "metadata.json"
GRAD_CACHE_MANIFEST_FILE = "cache_meta.json"
TRACIN_MATRIX_FILE = "tracin_matrix.npy"
DATAINF_MATRIX_FILE = "datainf_matrix.npy"


def prompt_preview(prompt: Any, limit: int = 200) -> str:
    value = prompt
    if isinstance(prompt, list):
        value = prompt[-1] if prompt else ""
        if isinstance(value, dict):
            value = value.get("content", "")
    return str(value)[:limit]


def solution_preview(solution: Any, limit: int = 100) -> str:
    if solution is None:
        return ""
    return str(solution)[:limit]


def to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value


@dataclass
class SampleDescriptor:
    index: int
    prompt_preview: str
    solution: str = ""
    prompt: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleDescriptor":
        return cls(
            index=int(data["index"]),
            prompt_preview=data.get("prompt_preview", ""),
            solution=data.get("solution", ""),
            prompt=data.get("prompt"),
        )


@dataclass
class CheckpointSummary:
    step: int
    learning_rate: float
    mean_test_grad_norm: float
    mean_train_grad_norm: float
    zero_test_cases: list[int]
    zero_train_cases: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointSummary":
        return cls(
            step=int(data["step"]),
            learning_rate=float(data["learning_rate"]),
            mean_test_grad_norm=float(data["mean_test_grad_norm"]),
            mean_train_grad_norm=float(data["mean_train_grad_norm"]),
            zero_test_cases=list(data.get("zero_test_cases", [])),
            zero_train_cases=list(data.get("zero_train_cases", [])),
        )


@dataclass
class MatrixManifest:
    tracin: str = TRACIN_MATRIX_FILE
    datainf: str = DATAINF_MATRIX_FILE
    tracin_steps: dict[str, str] = field(default_factory=dict)
    datainf_steps: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatrixManifest":
        return cls(
            tracin=data.get("tracin", TRACIN_MATRIX_FILE),
            datainf=data.get("datainf", DATAINF_MATRIX_FILE),
            tracin_steps=dict(data.get("tracin_steps", {})),
            datainf_steps=dict(data.get("datainf_steps", {})),
        )


@dataclass
class InfluenceResultsManifest:
    schema_version: int
    kind: str
    config: dict[str, Any]
    dimensions: dict[str, int]
    checkpoints: list[CheckpointSummary]
    test_samples: list[SampleDescriptor]
    train_samples: list[SampleDescriptor]
    matrices: MatrixManifest
    legacy_metadata_file: str = LEGACY_RESULTS_METADATA_FILE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "config": to_dict(self.config),
            "dimensions": to_dict(self.dimensions),
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "test_samples": [item.to_dict() for item in self.test_samples],
            "train_samples": [item.to_dict() for item in self.train_samples],
            "matrices": self.matrices.to_dict(),
            "legacy_metadata_file": self.legacy_metadata_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InfluenceResultsManifest":
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            kind=data.get("kind", "influence_results"),
            config=dict(data.get("config", {})),
            dimensions=dict(data.get("dimensions", {})),
            checkpoints=[
                CheckpointSummary.from_dict(item)
                for item in data.get("checkpoints", [])
            ],
            test_samples=[
                SampleDescriptor.from_dict(item)
                for item in data.get("test_samples", [])
            ],
            train_samples=[
                SampleDescriptor.from_dict(item)
                for item in data.get("train_samples", [])
            ],
            matrices=MatrixManifest.from_dict(data.get("matrices", {})),
            legacy_metadata_file=data.get(
                "legacy_metadata_file", LEGACY_RESULTS_METADATA_FILE
            ),
        )


@dataclass
class GradCacheSample:
    grad_file: str
    prompt: Any
    solution: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GradCacheSample":
        return cls(
            grad_file=data["grad_file"],
            prompt=data.get("prompt"),
            solution=data.get("solution"),
        )


@dataclass
class GradCacheCheckpoint:
    step: int
    learning_rate: float
    zero_test_cases: list[int]
    zero_train_cases: list[int]
    test_infos: list[GradCacheSample]
    train_infos: list[GradCacheSample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "learning_rate": self.learning_rate,
            "zero_test_cases": self.zero_test_cases,
            "zero_train_cases": self.zero_train_cases,
            "test_infos": [item.to_dict() for item in self.test_infos],
            "train_infos": [item.to_dict() for item in self.train_infos],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GradCacheCheckpoint":
        return cls(
            step=int(data["step"]),
            learning_rate=float(data["learning_rate"]),
            zero_test_cases=list(data.get("zero_test_cases", [])),
            zero_train_cases=list(data.get("zero_train_cases", [])),
            test_infos=[
                GradCacheSample.from_dict(item)
                for item in data.get("test_infos", [])
            ],
            train_infos=[
                GradCacheSample.from_dict(item)
                for item in data.get("train_infos", [])
            ],
        )


@dataclass
class GradCacheManifest:
    schema_version: int
    kind: str
    fingerprint: str
    config: dict[str, Any]
    checkpoints: list[GradCacheCheckpoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
            "config": to_dict(self.config),
            "checkpoints": [item.to_dict() for item in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GradCacheManifest":
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            kind=data.get("kind", "grad_cache"),
            fingerprint=data.get("fingerprint", ""),
            config=dict(data.get("config", {})),
            checkpoints=[
                GradCacheCheckpoint.from_dict(item)
                for item in data.get("checkpoints", [])
            ],
        )


@dataclass
class LoadedResults:
    root_dir: Path
    manifest: InfluenceResultsManifest
    tracin_matrix: Any
    datainf_matrix: Any
    tracin_step_matrices: dict[int, Any]
    datainf_step_matrices: dict[int, Any]
