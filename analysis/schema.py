from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_SCHEMA_VERSION = 4
GRAD_CACHE_SCHEMA_VERSION = 4
TRAIN_BATCH_HISTORY_SCHEMA_VERSION = 1
RESULTS_MANIFEST_FILE = "results_manifest.json"
LEGACY_RESULTS_METADATA_FILE = "metadata.json"
GRAD_CACHE_MANIFEST_FILE = "cache_meta.json"
TRAIN_BATCH_HISTORY_FILE = "historical_batch_history.json"
TRACIN_MATRIX_FILE = "tracin_matrix.npy"
DATAINF_MATRIX_FILE = "datainf_matrix.npy"
FISHER_MATRIX_FILE = "fisher_matrix.npy"


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
    dataset_train_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleDescriptor":
        raw = data.get("dataset_train_index")
        dataset_train_index = int(raw) if raw is not None else None
        return cls(
            index=int(data["index"]),
            prompt_preview=data.get("prompt_preview", ""),
            solution=data.get("solution", ""),
            prompt=data.get("prompt"),
            dataset_train_index=dataset_train_index,
        )


@dataclass
class CheckpointSummary:
    step: int
    learning_rate: float
    mean_test_grad_norm: float
    mean_train_grad_norm: float
    zero_test_cases: list[int]
    zero_train_cases: list[int]
    math_eval: dict[str, Any] | None = None
    code_eval: dict[str, Any] | None = None
    historical_total_rows: int | None = None
    historical_nonzero_train: int | None = None

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
            math_eval=dict(data["math_eval"]) if data.get("math_eval") is not None else None,
            code_eval=dict(data["code_eval"]) if data.get("code_eval") is not None else None,
            historical_total_rows=data.get("historical_total_rows"),
            historical_nonzero_train=data.get("historical_nonzero_train"),
        )


@dataclass
class MatrixManifest:
    tracin: str = TRACIN_MATRIX_FILE
    datainf: str = DATAINF_MATRIX_FILE
    fisher: str | None = None
    tracin_steps: dict[str, str] = field(default_factory=dict)
    datainf_steps: dict[str, str] = field(default_factory=dict)
    fisher_steps: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatrixManifest":
        return cls(
            tracin=data.get("tracin", TRACIN_MATRIX_FILE),
            datainf=data.get("datainf", DATAINF_MATRIX_FILE),
            fisher=data.get("fisher"),
            tracin_steps=dict(data.get("tracin_steps", {})),
            datainf_steps=dict(data.get("datainf_steps", {})),
            fisher_steps=dict(data.get("fisher_steps", {})),
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
    training_elapsed_s: float | None = None
    replay_elapsed_s: float | None = None
    total_elapsed_s: float | None = None
    legacy_metadata_file: str = LEGACY_RESULTS_METADATA_FILE

    def to_dict(self) -> dict[str, Any]:
        d = {
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
        if self.training_elapsed_s is not None:
            d["training_elapsed_s"] = round(self.training_elapsed_s, 2)
        if self.replay_elapsed_s is not None:
            d["replay_elapsed_s"] = round(self.replay_elapsed_s, 2)
        if self.total_elapsed_s is not None:
            d["total_elapsed_s"] = round(self.total_elapsed_s, 2)
        return d

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
            training_elapsed_s=data.get("training_elapsed_s"),
            replay_elapsed_s=data.get("replay_elapsed_s"),
            total_elapsed_s=data.get("total_elapsed_s"),
            legacy_metadata_file=data.get(
                "legacy_metadata_file", LEGACY_RESULTS_METADATA_FILE
            ),
        )


@dataclass
class HistoricalBatchStep:
    step: int
    total_rows: int
    train_index_counts: dict[int, int]
    microbatch_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step": self.step,
            "total_rows": self.total_rows,
            "train_index_counts": {
                str(key): int(value)
                for key, value in self.train_index_counts.items()
            },
        }
        if self.microbatch_count is not None:
            payload["microbatch_count"] = self.microbatch_count
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoricalBatchStep":
        return cls(
            step=int(data["step"]),
            total_rows=int(data.get("total_rows", 0)),
            train_index_counts={
                int(key): int(value)
                for key, value in data.get("train_index_counts", {}).items()
            },
            microbatch_count=data.get("microbatch_count"),
        )


@dataclass
class HistoricalBatchManifest:
    schema_version: int
    kind: str
    steps: list[HistoricalBatchStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "steps": [item.to_dict() for item in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoricalBatchManifest":
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            kind=data.get("kind", "historical_batch_history"),
            steps=[
                HistoricalBatchStep.from_dict(item)
                for item in data.get("steps", [])
            ],
        )


@dataclass
class GradCacheSample:
    grad_file: str
    prompt: Any
    solution: Any = None
    train_index: int | None = None
    historical_weight: float | None = None
    geometry_feature_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GradCacheSample":
        return cls(
            grad_file=data["grad_file"],
            prompt=data.get("prompt"),
            solution=data.get("solution"),
            train_index=data.get("train_index"),
            historical_weight=data.get("historical_weight"),
            geometry_feature_file=data.get("geometry_feature_file"),
        )


@dataclass
class GradCacheCheckpoint:
    step: int
    learning_rate: float
    zero_test_cases: list[int]
    zero_train_cases: list[int]
    test_infos: list[GradCacheSample]
    train_infos: list[GradCacheSample]
    math_eval: dict[str, Any] | None = None
    code_eval: dict[str, Any] | None = None
    historical_total_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "step": self.step,
            "learning_rate": self.learning_rate,
            "zero_test_cases": self.zero_test_cases,
            "zero_train_cases": self.zero_train_cases,
            "test_infos": [item.to_dict() for item in self.test_infos],
            "train_infos": [item.to_dict() for item in self.train_infos],
        }
        if self.math_eval is not None:
            payload["math_eval"] = self.math_eval
        if self.code_eval is not None:
            payload["code_eval"] = self.code_eval
        if self.historical_total_rows is not None:
            payload["historical_total_rows"] = self.historical_total_rows
        return payload

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
            math_eval=dict(data["math_eval"]) if data.get("math_eval") is not None else None,
            code_eval=dict(data["code_eval"]) if data.get("code_eval") is not None else None,
            historical_total_rows=data.get("historical_total_rows"),
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
    fisher_matrix: Any = None
    fisher_step_matrices: dict[int, Any] | None = None
