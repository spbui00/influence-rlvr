from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

ROLLOUT_CACHE_SCHEMA_VERSION = 1
ROLLOUT_CACHE_MANIFEST_FILE = "rollout_cache_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _step_file_name(step: int) -> str:
    return f"step_{int(step):06d}.pt"


@dataclass
class RolloutCacheStepSummary:
    step: int
    file: str
    total_rows: int
    microbatch_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutCacheStepSummary":
        return cls(
            step=int(data["step"]),
            file=str(data["file"]),
            total_rows=int(data.get("total_rows", 0)),
            microbatch_count=int(data.get("microbatch_count", 0)),
        )


@dataclass
class RolloutCacheManifest:
    schema_version: int
    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    steps: list[RolloutCacheStepSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "kind": self.kind,
            "config": dict(self.config),
            "steps": [item.to_dict() for item in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutCacheManifest":
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            kind=data.get("kind", "training_rollout_cache"),
            config=dict(data.get("config", {})),
            steps=[
                RolloutCacheStepSummary.from_dict(item)
                for item in data.get("steps", [])
            ],
        )


def load_rollout_cache_manifest(cache_dir: str | Path) -> RolloutCacheManifest | None:
    path = Path(cache_dir) / ROLLOUT_CACHE_MANIFEST_FILE
    if not path.exists():
        return None
    return RolloutCacheManifest.from_dict(_read_json(path))


def save_rollout_cache_manifest(
    cache_dir: str | Path,
    manifest: RolloutCacheManifest,
) -> RolloutCacheManifest:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    _write_json(cache_path / ROLLOUT_CACHE_MANIFEST_FILE, manifest.to_dict())
    return manifest


def load_rollout_cache_step(cache_dir: str | Path, step: int) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    file_path = cache_path / _step_file_name(step)
    if not file_path.exists():
        raise FileNotFoundError(f"Rollout cache step file not found: {file_path}")
    return torch.load(file_path, map_location="cpu")


def save_rollout_cache_step(cache_dir: str | Path, step_record: dict[str, Any]) -> RolloutCacheStepSummary:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    step = int(step_record["step"])
    file_name = _step_file_name(step)
    torch.save(step_record, cache_path / file_name)
    return RolloutCacheStepSummary(
        step=step,
        file=file_name,
        total_rows=int(step_record.get("total_rows", 0)),
        microbatch_count=int(step_record.get("microbatch_count", 0)),
    )


class TrainingRolloutCacheWriter:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = load_rollout_cache_manifest(self.cache_dir)
        if self.manifest is None:
            self.manifest = RolloutCacheManifest(
                schema_version=ROLLOUT_CACHE_SCHEMA_VERSION,
                kind="training_rollout_cache",
                config=dict(config or {}),
                steps=[],
            )
            save_rollout_cache_manifest(self.cache_dir, self.manifest)
        elif config:
            merged = dict(self.manifest.config)
            merged.update(config)
            self.manifest.config = merged
            save_rollout_cache_manifest(self.cache_dir, self.manifest)

    def prepare_for_resume(self, *, max_step: int | None = None) -> tuple[int, int]:
        if max_step is None:
            return len(self.manifest.steps), 0

        kept: list[RolloutCacheStepSummary] = []
        dropped = 0
        for item in self.manifest.steps:
            if int(item.step) <= max_step:
                kept.append(item)
                continue

            file_path = self.cache_dir / item.file
            if file_path.exists():
                file_path.unlink()
            dropped += 1

        if dropped:
            self.manifest.steps = kept
            save_rollout_cache_manifest(self.cache_dir, self.manifest)
        return len(kept), dropped

    def append_step(self, step_record: dict[str, Any]) -> RolloutCacheStepSummary:
        summary = save_rollout_cache_step(self.cache_dir, step_record)
        by_step = {
            int(item.step): item
            for item in self.manifest.steps
        }
        by_step[int(summary.step)] = summary
        self.manifest.steps = [
            by_step[key]
            for key in sorted(by_step)
        ]
        save_rollout_cache_manifest(self.cache_dir, self.manifest)
        return summary

    def close(self) -> None:
        save_rollout_cache_manifest(self.cache_dir, self.manifest)
