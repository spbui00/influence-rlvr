from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    DATAINF_MATRIX_FILE,
    FISHER_MATRIX_FILE,
    GRAD_CACHE_MANIFEST_FILE,
    GRAD_CACHE_SCHEMA_VERSION,
    LEGACY_RESULTS_METADATA_FILE,
    RESULTS_MANIFEST_FILE,
    RESULTS_SCHEMA_VERSION,
    TRAIN_BATCH_HISTORY_FILE,
    TRAIN_BATCH_HISTORY_SCHEMA_VERSION,
    TRACIN_MATRIX_FILE,
    CheckpointSummary,
    GradCacheCheckpoint,
    GradCacheManifest,
    GradCacheSample,
    HistoricalBatchManifest,
    HistoricalBatchStep,
    InfluenceResultsManifest,
    LoadedResults,
    MatrixManifest,
    SampleDescriptor,
    prompt_preview,
    solution_preview,
)

_RESULTS_DIR_RE = re.compile(r"^results(?:(\d+))?$")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_cache_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_batch_history_fingerprint(manifest: HistoricalBatchManifest) -> str:
    payload = json.dumps(manifest.to_dict()["steps"], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalize_results_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.pop("results_dir", None)
    normalized.pop("results_config_fingerprint", None)
    normalized.pop("results_name", None)
    return normalized


def build_results_fingerprint(config: dict[str, Any]) -> str:
    return build_cache_fingerprint(normalize_results_config(config))


def _results_dir_index(path: Path) -> int | None:
    match = _RESULTS_DIR_RE.fullmatch(path.name)
    if match is None:
        return None
    suffix = match.group(1)
    return 1 if suffix is None else int(suffix)


def next_results_dir(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    next_index = 1
    for child in run_path.iterdir():
        if not child.is_dir():
            continue
        index = _results_dir_index(child)
        if index is None:
            continue
        next_index = max(next_index, index + 1)
    return run_path / f"results{next_index}"


def resolve_results_dir(
    run_dir: str | Path,
    config: dict[str, Any],
) -> tuple[Path, bool, str]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    normalized_config = normalize_results_config(config)
    fingerprint = build_results_fingerprint(normalized_config)

    for child in sorted(run_path.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        index = _results_dir_index(child)
        if index is None:
            continue
        manifest_path = child / RESULTS_MANIFEST_FILE
        legacy_path = child / LEGACY_RESULTS_METADATA_FILE
        if not manifest_path.exists() and not legacy_path.exists():
            continue
        try:
            manifest = load_results_manifest(child)
        except Exception:
            continue
        existing_config = normalize_results_config(manifest.config)
        existing_fingerprint = manifest.config.get("results_config_fingerprint")
        if existing_fingerprint is None:
            existing_fingerprint = build_results_fingerprint(existing_config)
        if (
            existing_fingerprint == fingerprint
            and existing_config == normalized_config
        ):
            return child, True, fingerprint

    return next_results_dir(run_path), False, fingerprint


def save_batch_history(
    step_records: list[dict[str, Any]],
    output_dir: str | Path,
) -> HistoricalBatchManifest:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = HistoricalBatchManifest(
        schema_version=TRAIN_BATCH_HISTORY_SCHEMA_VERSION,
        kind="historical_batch_history",
        steps=[
            HistoricalBatchStep.from_dict(step)
            if not isinstance(step, HistoricalBatchStep)
            else step
            for step in step_records
        ],
    )
    _write_json(output_path / TRAIN_BATCH_HISTORY_FILE, manifest.to_dict())
    return manifest


def load_batch_history(output_dir: str | Path) -> HistoricalBatchManifest | None:
    path = Path(output_dir) / TRAIN_BATCH_HISTORY_FILE
    if not path.exists():
        return None
    return HistoricalBatchManifest.from_dict(_read_json(path))


def build_batch_weight_lookup(
    manifest: HistoricalBatchManifest,
) -> dict[int, dict[str, Any]]:
    lookup = {}
    for step in manifest.steps:
        total_rows = max(step.total_rows, 1)
        lookup[step.step] = {
            "total_rows": step.total_rows,
            "microbatch_count": step.microbatch_count,
            "weights": {
                idx: count / total_rows
                for idx, count in step.train_index_counts.items()
            },
        }
    return lookup


def build_checkpoint_summaries(checkpoint_infos: list[dict[str, Any]]) -> list[CheckpointSummary]:
    summaries = []
    for checkpoint in checkpoint_infos:
        test_norms = [info["grad"].norm().item() for info in checkpoint["test_infos"]]
        train_norms = [info["grad"].norm().item() for info in checkpoint["train_infos"]]
        historical_nonzero_train = None
        if any("historical_weight" in info for info in checkpoint["train_infos"]):
            historical_nonzero_train = sum(
                1
                for info in checkpoint["train_infos"]
                if float(info.get("historical_weight", 1.0)) > 0.0
            )
        summaries.append(
            CheckpointSummary(
                step=int(checkpoint["step"]),
                learning_rate=float(checkpoint["learning_rate"]),
                mean_test_grad_norm=float(np.mean(test_norms)) if test_norms else 0.0,
                mean_train_grad_norm=float(np.mean(train_norms)) if train_norms else 0.0,
                zero_test_cases=list(checkpoint.get("zero_test_cases", [])),
                zero_train_cases=list(checkpoint.get("zero_train_cases", [])),
                math_eval=checkpoint.get("math_eval"),
                code_eval=checkpoint.get("code_eval"),
                historical_total_rows=checkpoint.get("historical_total_rows"),
                historical_nonzero_train=historical_nonzero_train,
            )
        )
    return summaries


def _build_sample_descriptors(infos: list[dict[str, Any]]) -> list[SampleDescriptor]:
    samples = []
    for index, info in enumerate(infos):
        samples.append(
            SampleDescriptor(
                index=index,
                prompt_preview=prompt_preview(info.get("prompt")),
                solution=solution_preview(info.get("solution")),
                prompt=info.get("prompt"),
            )
        )
    return samples


def _build_matrix_manifest(
    tracin_breakdown: list[dict[str, Any]],
    datainf_breakdown: list[dict[str, Any]],
    fisher_breakdown: list[dict[str, Any]] | None = None,
) -> MatrixManifest:
    return MatrixManifest(
        tracin=TRACIN_MATRIX_FILE,
        datainf=DATAINF_MATRIX_FILE,
        fisher=FISHER_MATRIX_FILE if fisher_breakdown is not None else None,
        tracin_steps={
            str(entry["step"]): f"tracin_step_{entry['step']}.npy"
            for entry in tracin_breakdown
        },
        datainf_steps={
            str(entry["step"]): f"datainf_step_{entry['step']}.npy"
            for entry in datainf_breakdown
        },
        fisher_steps=(
            {
                str(entry["step"]): f"fisher_step_{entry['step']}.npy"
                for entry in fisher_breakdown
            }
            if fisher_breakdown is not None else {}
        ),
    )


def build_results_manifest(
    checkpoint_infos: list[dict[str, Any]],
    tracin_breakdown: list[dict[str, Any]],
    datainf_breakdown: list[dict[str, Any]],
    fisher_breakdown: list[dict[str, Any]] | None,
    config: dict[str, Any],
    *,
    training_elapsed_s: float | None = None,
    replay_elapsed_s: float | None = None,
    total_elapsed_s: float | None = None,
) -> InfluenceResultsManifest:
    last_checkpoint = checkpoint_infos[-1] if checkpoint_infos else {"test_infos": [], "train_infos": []}
    test_samples = _build_sample_descriptors(last_checkpoint["test_infos"])
    train_samples = _build_sample_descriptors(last_checkpoint["train_infos"])
    checkpoint_summaries = build_checkpoint_summaries(checkpoint_infos)
    dimensions = {
        "n_test_actual": len(test_samples),
        "n_train_actual": len(train_samples),
        "n_checkpoints": len(checkpoint_summaries),
    }
    return InfluenceResultsManifest(
        schema_version=RESULTS_SCHEMA_VERSION,
        kind="influence_results",
        config=dict(config),
        dimensions=dimensions,
        checkpoints=checkpoint_summaries,
        test_samples=test_samples,
        train_samples=train_samples,
        matrices=_build_matrix_manifest(tracin_breakdown, datainf_breakdown, fisher_breakdown),
        training_elapsed_s=training_elapsed_s,
        replay_elapsed_s=replay_elapsed_s,
        total_elapsed_s=total_elapsed_s,
    )


def build_legacy_metadata(manifest: InfluenceResultsManifest) -> dict[str, Any]:
    config = dict(manifest.config)
    meta: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "results_manifest_file": RESULTS_MANIFEST_FILE,
        "model_id": config.get("model_id"),
        "output_dir": config.get("output_dir"),
        "learning_rate": config.get("learning_rate"),
        "max_steps": config.get("max_steps"),
        "grpo_beta": config.get("grpo_beta"),
        "grpo_epsilon": config.get("grpo_epsilon"),
        "g_train": config.get("g_train"),
        "g_test": config.get("g_test"),
        "n_math": config.get("n_math"),
        "n_code": config.get("n_code"),
        "n_code_train": config.get("n_code_train"),
        "n_train_replay": config.get("n_train_replay"),
        "code_train_split": config.get("code_train_split"),
        "n_test_actual": manifest.dimensions.get("n_test_actual"),
        "n_train_actual": manifest.dimensions.get("n_train_actual"),
        "lambda_damp": config.get("lambda_damp"),
        "train_grad_seed": config.get("train_grad_seed"),
        "device": config.get("device"),
        "influence_mode": config.get("influence_mode"),
        "experiment_mode": config.get("experiment_mode"),
        "batch_history_fingerprint": config.get("batch_history_fingerprint"),
        "math_eval_split": config.get("math_eval_split"),
        "math_eval_percent": config.get("math_eval_percent"),
        "code_eval_split": config.get("code_eval_split"),
        "code_eval_percent": config.get("code_eval_percent"),
        "eval_max_new_tokens": config.get("eval_max_new_tokens"),
        "generation_backend": config.get("generation_backend"),
        "code_eval_do_sample": config.get("code_eval_do_sample"),
        "code_eval_num_samples": config.get("code_eval_num_samples"),
        "code_eval_temperature": config.get("code_eval_temperature"),
        "code_eval_top_p": config.get("code_eval_top_p"),
        "train_gradient_objective": config.get("train_gradient_objective"),
        "test_gradient_objective": config.get("test_gradient_objective"),
        "train_geometry_feature": config.get("train_geometry_feature"),
        "second_order_geometry": config.get("second_order_geometry"),
        "fisher_normalize": config.get("fisher_normalize"),
        "replay_max_new_tokens": config.get("replay_max_new_tokens"),
        "replay_temperature": config.get("replay_temperature"),
        "replay_top_p": config.get("replay_top_p"),
        "vllm_gpu_memory_utilization": config.get("vllm_gpu_memory_utilization"),
        "vllm_tensor_parallel_size": config.get("vllm_tensor_parallel_size"),
        "vllm_max_model_len": config.get("vllm_max_model_len"),
        "vllm_max_num_seqs": config.get("vllm_max_num_seqs"),
        "vllm_max_lora_rank": config.get("vllm_max_lora_rank"),
        "vllm_enforce_eager": config.get("vllm_enforce_eager"),
        "vllm_training_use_vllm": config.get("vllm_training_use_vllm"),
        "train_domain": config.get("train_domain"),
        "test_domain": config.get("test_domain"),
        "checkpoints": [item.to_dict() for item in manifest.checkpoints],
        "test_prompts": [item.prompt_preview for item in manifest.test_samples],
        "train_prompts": [item.prompt_preview for item in manifest.train_samples],
        "train_solutions": [item.solution for item in manifest.train_samples],
    }
    if manifest.training_elapsed_s is not None:
        meta["training_elapsed_s"] = round(manifest.training_elapsed_s, 2)
    if manifest.replay_elapsed_s is not None:
        meta["replay_elapsed_s"] = round(manifest.replay_elapsed_s, 2)
    if manifest.total_elapsed_s is not None:
        meta["total_elapsed_s"] = round(manifest.total_elapsed_s, 2)
    return meta


def save_results_bundle(
    results_dir: str | Path,
    tracin_matrix: np.ndarray,
    datainf_matrix: np.ndarray,
    fisher_matrix: np.ndarray | None,
    tracin_breakdown: list[dict[str, Any]],
    datainf_breakdown: list[dict[str, Any]],
    fisher_breakdown: list[dict[str, Any]] | None,
    checkpoint_infos: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    training_elapsed_s: float | None = None,
    replay_elapsed_s: float | None = None,
    total_elapsed_s: float | None = None,
) -> InfluenceResultsManifest:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    np.save(results_path / TRACIN_MATRIX_FILE, tracin_matrix)
    np.save(results_path / DATAINF_MATRIX_FILE, datainf_matrix)
    if fisher_matrix is not None:
        np.save(results_path / FISHER_MATRIX_FILE, fisher_matrix)

    for entry in tracin_breakdown:
        np.save(
            results_path / f"tracin_step_{entry['step']}.npy",
            entry["weighted_matrix"],
        )
    for entry in datainf_breakdown:
        np.save(
            results_path / f"datainf_step_{entry['step']}.npy",
            entry["weighted_matrix"],
        )
    for entry in fisher_breakdown or []:
        np.save(
            results_path / f"fisher_step_{entry['step']}.npy",
            entry["weighted_matrix"],
        )

    manifest = build_results_manifest(
        checkpoint_infos,
        tracin_breakdown,
        datainf_breakdown,
        fisher_breakdown,
        config,
        training_elapsed_s=training_elapsed_s,
        replay_elapsed_s=replay_elapsed_s,
        total_elapsed_s=total_elapsed_s,
    )
    _write_json(results_path / RESULTS_MANIFEST_FILE, manifest.to_dict())
    _write_json(results_path / LEGACY_RESULTS_METADATA_FILE, build_legacy_metadata(manifest))
    return manifest


def _infer_matrix_manifest(results_path: Path, metadata: dict[str, Any]) -> MatrixManifest:
    tracin_steps = {}
    datainf_steps = {}
    fisher_steps = {}
    for checkpoint in metadata.get("checkpoints", []):
        step = int(checkpoint["step"])
        tracin_name = f"tracin_step_{step}.npy"
        datainf_name = f"datainf_step_{step}.npy"
        fisher_name = f"fisher_step_{step}.npy"
        if (results_path / tracin_name).exists():
            tracin_steps[str(step)] = tracin_name
        if (results_path / datainf_name).exists():
            datainf_steps[str(step)] = datainf_name
        if (results_path / fisher_name).exists():
            fisher_steps[str(step)] = fisher_name
    return MatrixManifest(
        tracin=TRACIN_MATRIX_FILE,
        datainf=DATAINF_MATRIX_FILE,
        fisher=FISHER_MATRIX_FILE if (results_path / FISHER_MATRIX_FILE).exists() else None,
        tracin_steps=tracin_steps,
        datainf_steps=datainf_steps,
        fisher_steps=fisher_steps,
    )


def _manifest_from_legacy_metadata(results_path: Path) -> InfluenceResultsManifest:
    metadata = _read_json(results_path / LEGACY_RESULTS_METADATA_FILE)
    test_prompts = list(metadata.get("test_prompts", []))
    train_prompts = list(metadata.get("train_prompts", []))
    train_solutions = list(metadata.get("train_solutions", []))
    test_solutions = list(metadata.get("test_solutions", []))

    test_samples = [
        SampleDescriptor(
            index=index,
            prompt_preview=str(prompt),
            solution=solution_preview(test_solutions[index]) if index < len(test_solutions) else "",
        )
        for index, prompt in enumerate(test_prompts)
    ]
    train_samples = [
        SampleDescriptor(
            index=index,
            prompt_preview=str(prompt),
            solution=solution_preview(train_solutions[index]) if index < len(train_solutions) else "",
        )
        for index, prompt in enumerate(train_prompts)
    ]
    checkpoints = [
        CheckpointSummary.from_dict(item)
        for item in metadata.get("checkpoints", [])
    ]
    config = {
        "model_id": metadata.get("model_id"),
        "output_dir": metadata.get("output_dir"),
        "learning_rate": metadata.get("learning_rate"),
        "max_steps": metadata.get("max_steps"),
        "grpo_beta": metadata.get("grpo_beta"),
        "grpo_epsilon": metadata.get("grpo_epsilon"),
        "g_train": metadata.get("g_train"),
        "g_test": metadata.get("g_test"),
        "n_math": metadata.get("n_math"),
        "n_code": metadata.get("n_code"),
        "n_code_train": metadata.get("n_code_train"),
        "n_train_replay": metadata.get("n_train_replay"),
        "code_train_split": metadata.get("code_train_split"),
        "lambda_damp": metadata.get("lambda_damp"),
        "train_grad_seed": metadata.get("train_grad_seed"),
        "device": metadata.get("device"),
        "influence_mode": metadata.get("influence_mode"),
        "experiment_mode": metadata.get("experiment_mode"),
        "batch_history_fingerprint": metadata.get("batch_history_fingerprint"),
        "math_eval_split": metadata.get("math_eval_split"),
        "math_eval_percent": metadata.get("math_eval_percent"),
        "code_eval_split": metadata.get("code_eval_split"),
        "code_eval_percent": metadata.get("code_eval_percent"),
        "eval_max_new_tokens": metadata.get("eval_max_new_tokens"),
        "generation_backend": metadata.get("generation_backend"),
        "code_eval_do_sample": metadata.get("code_eval_do_sample"),
        "code_eval_num_samples": metadata.get("code_eval_num_samples"),
        "code_eval_temperature": metadata.get("code_eval_temperature"),
        "code_eval_top_p": metadata.get("code_eval_top_p"),
        "train_gradient_objective": metadata.get("train_gradient_objective"),
        "test_gradient_objective": metadata.get("test_gradient_objective"),
        "train_geometry_feature": metadata.get("train_geometry_feature"),
        "second_order_geometry": metadata.get("second_order_geometry"),
        "fisher_normalize": metadata.get("fisher_normalize"),
        "replay_max_new_tokens": metadata.get("replay_max_new_tokens"),
        "replay_temperature": metadata.get("replay_temperature"),
        "replay_top_p": metadata.get("replay_top_p"),
        "vllm_gpu_memory_utilization": metadata.get("vllm_gpu_memory_utilization"),
        "vllm_tensor_parallel_size": metadata.get("vllm_tensor_parallel_size"),
        "vllm_max_model_len": metadata.get("vllm_max_model_len"),
        "vllm_max_num_seqs": metadata.get("vllm_max_num_seqs"),
        "vllm_max_lora_rank": metadata.get("vllm_max_lora_rank"),
        "vllm_enforce_eager": metadata.get("vllm_enforce_eager"),
        "vllm_training_use_vllm": metadata.get("vllm_training_use_vllm"),
        "train_domain": metadata.get("train_domain"),
        "test_domain": metadata.get("test_domain"),
    }
    dimensions = {
        "n_test_actual": int(metadata.get("n_test_actual", len(test_samples))),
        "n_train_actual": int(metadata.get("n_train_actual", len(train_samples))),
        "n_checkpoints": len(checkpoints),
    }
    return InfluenceResultsManifest(
        schema_version=int(metadata.get("schema_version", 0)),
        kind="influence_results",
        config=config,
        dimensions=dimensions,
        checkpoints=checkpoints,
        test_samples=test_samples,
        train_samples=train_samples,
        matrices=_infer_matrix_manifest(results_path, metadata),
        legacy_metadata_file=LEGACY_RESULTS_METADATA_FILE,
    )


def load_results_manifest(results_dir: str | Path) -> InfluenceResultsManifest:
    results_path = Path(results_dir)
    manifest_path = results_path / RESULTS_MANIFEST_FILE
    if manifest_path.exists():
        return InfluenceResultsManifest.from_dict(_read_json(manifest_path))
    return _manifest_from_legacy_metadata(results_path)


def _load_step_matrices(
    results_path: Path,
    step_files: dict[str, str],
) -> dict[int, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for step_text, file_name in step_files.items():
        path = results_path / file_name
        if path.exists():
            matrices[int(step_text)] = np.load(path)
    return matrices


def load_results_bundle(results_dir: str | Path) -> LoadedResults:
    results_path = Path(results_dir)
    manifest = load_results_manifest(results_path)
    tracin_matrix = np.load(results_path / manifest.matrices.tracin)
    datainf_matrix = np.load(results_path / manifest.matrices.datainf)
    fisher_matrix = None
    if manifest.matrices.fisher is not None:
        fisher_path = results_path / manifest.matrices.fisher
        if fisher_path.exists():
            fisher_matrix = np.load(fisher_path)
    tracin_step_matrices = _load_step_matrices(
        results_path,
        manifest.matrices.tracin_steps,
    )
    datainf_step_matrices = _load_step_matrices(
        results_path,
        manifest.matrices.datainf_steps,
    )
    fisher_step_matrices = _load_step_matrices(
        results_path,
        manifest.matrices.fisher_steps,
    )
    return LoadedResults(
        root_dir=results_path,
        manifest=manifest,
        tracin_matrix=tracin_matrix,
        datainf_matrix=datainf_matrix,
        tracin_step_matrices=tracin_step_matrices,
        datainf_step_matrices=datainf_step_matrices,
        fisher_matrix=fisher_matrix,
        fisher_step_matrices=fisher_step_matrices,
    )


def save_grad_cache(
    checkpoint_infos: list[dict[str, Any]],
    cache_dir: str | Path,
    fingerprint: str,
    config: dict[str, Any],
) -> GradCacheManifest:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    checkpoints: list[GradCacheCheckpoint] = []
    for checkpoint in checkpoint_infos:
        step = int(checkpoint["step"])
        test_infos = []
        for index, info in enumerate(checkpoint["test_infos"]):
            grad_file = f"step{step}_test_{index}.npy"
            np.save(cache_path / grad_file, info["grad"].numpy())
            geometry_feature_file = None
            if info.get("geometry_feature") is not None:
                geometry_feature_file = f"step{step}_test_{index}_geometry.npy"
                np.save(cache_path / geometry_feature_file, info["geometry_feature"].numpy())
            test_infos.append(
                GradCacheSample(
                    grad_file=grad_file,
                    prompt=info.get("prompt"),
                    solution=info.get("solution"),
                    train_index=info.get("train_index"),
                    historical_weight=info.get("historical_weight"),
                    geometry_feature_file=geometry_feature_file,
                )
            )
        train_infos = []
        for index, info in enumerate(checkpoint["train_infos"]):
            grad_file = f"step{step}_train_{index}.npy"
            np.save(cache_path / grad_file, info["grad"].numpy())
            geometry_feature_file = None
            if info.get("geometry_feature") is not None:
                geometry_feature_file = f"step{step}_train_{index}_geometry.npy"
                np.save(cache_path / geometry_feature_file, info["geometry_feature"].numpy())
            train_infos.append(
                GradCacheSample(
                    grad_file=grad_file,
                    prompt=info.get("prompt"),
                    solution=info.get("solution"),
                    train_index=info.get("train_index"),
                    historical_weight=info.get("historical_weight"),
                    geometry_feature_file=geometry_feature_file,
                )
            )
        checkpoints.append(
            GradCacheCheckpoint(
                step=step,
                learning_rate=float(checkpoint["learning_rate"]),
                zero_test_cases=list(checkpoint.get("zero_test_cases", [])),
                zero_train_cases=list(checkpoint.get("zero_train_cases", [])),
                test_infos=test_infos,
                train_infos=train_infos,
                math_eval=checkpoint.get("math_eval"),
                code_eval=checkpoint.get("code_eval"),
                historical_total_rows=checkpoint.get("historical_total_rows"),
            )
        )

    manifest = GradCacheManifest(
        schema_version=GRAD_CACHE_SCHEMA_VERSION,
        kind="grad_cache",
        fingerprint=fingerprint,
        config=dict(config),
        checkpoints=checkpoints,
    )
    _write_json(Path(cache_dir) / GRAD_CACHE_MANIFEST_FILE, manifest.to_dict())
    return manifest


def _load_grad_cache_manifest(cache_dir: str | Path) -> GradCacheManifest:
    cache_path = Path(cache_dir)
    raw = _read_json(cache_path / GRAD_CACHE_MANIFEST_FILE)
    if "schema_version" in raw or "kind" in raw or "config" in raw:
        return GradCacheManifest.from_dict(raw)
    return GradCacheManifest(
        schema_version=0,
        kind="grad_cache",
        fingerprint=raw.get("fingerprint", ""),
        config=dict(raw.get("config", {})),
        checkpoints=[
            GradCacheCheckpoint.from_dict(item)
            for item in raw.get("checkpoints", [])
        ],
    )


def load_grad_cache(
    cache_dir: str | Path,
    expected_fingerprint: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    import torch

    manifest = _load_grad_cache_manifest(cache_dir)
    if manifest.fingerprint != expected_fingerprint:
        return None, manifest.fingerprint

    cache_path = Path(cache_dir)
    checkpoint_infos: list[dict[str, Any]] = []
    for checkpoint in manifest.checkpoints:
        test_infos = []
        for sample in checkpoint.test_infos:
            grad = torch.from_numpy(np.load(cache_path / sample.grad_file))
            info = {
                "grad": grad,
                "prompt": sample.prompt,
                "solution": sample.solution,
            }
            if sample.geometry_feature_file is not None:
                info["geometry_feature"] = torch.from_numpy(
                    np.load(cache_path / sample.geometry_feature_file)
                )
            if sample.train_index is not None:
                info["train_index"] = sample.train_index
            if sample.historical_weight is not None:
                info["historical_weight"] = sample.historical_weight
            test_infos.append(info)
        train_infos = []
        for sample in checkpoint.train_infos:
            grad = torch.from_numpy(np.load(cache_path / sample.grad_file))
            info = {
                "grad": grad,
                "prompt": sample.prompt,
                "solution": sample.solution,
            }
            if sample.geometry_feature_file is not None:
                info["geometry_feature"] = torch.from_numpy(
                    np.load(cache_path / sample.geometry_feature_file)
                )
            if sample.train_index is not None:
                info["train_index"] = sample.train_index
            if sample.historical_weight is not None:
                info["historical_weight"] = sample.historical_weight
            train_infos.append(info)
        checkpoint_infos.append({
            "step": checkpoint.step,
            "learning_rate": checkpoint.learning_rate,
            "zero_test_cases": checkpoint.zero_test_cases,
            "zero_train_cases": checkpoint.zero_train_cases,
            "test_infos": test_infos,
            "train_infos": train_infos,
            "math_eval": checkpoint.math_eval,
            "code_eval": checkpoint.code_eval,
            "historical_total_rows": checkpoint.historical_total_rows,
        })
    return checkpoint_infos, manifest.fingerprint
