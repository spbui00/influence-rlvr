from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .loader import load_results_bundle
from .plots import (
    agreement_scatter_figure,
    eval_performance_figure,
    gradient_norms_figure,
    heatmap_figure,
    training_curves_figure,
    trajectory_pairs_figure,
)


@dataclass
class RankedInfluence:
    method: str
    test_index: int
    train_index: int
    score: float
    test_prompt: str
    train_prompt: str
    train_solution: str


class InfluenceAnalyzer:
    def __init__(self, bundle):
        self.bundle = bundle
        self.results_dir = Path(bundle.root_dir)
        self.manifest = bundle.manifest
        self.steps = [item.step for item in self.manifest.checkpoints]
        self.test_labels = [f"T{i}" for i in range(len(self.manifest.test_samples))]
        self.train_labels = [f"R{i}" for i in range(len(self.manifest.train_samples))]

    @classmethod
    def from_directory(cls, results_dir: str | Path) -> "InfluenceAnalyzer":
        return cls(load_results_bundle(results_dir))

    @property
    def test_prompts(self) -> list[str]:
        return [item.prompt_preview for item in self.manifest.test_samples]

    @property
    def train_prompts(self) -> list[str]:
        return [item.prompt_preview for item in self.manifest.train_samples]

    @property
    def train_solutions(self) -> list[str]:
        return [item.solution for item in self.manifest.train_samples]

    @property
    def has_eval_metrics(self) -> bool:
        return any(
            item.math_eval is not None or item.code_eval is not None
            for item in self.manifest.checkpoints
        )

    @cached_property
    def training_log_history(self) -> list[dict[str, object]]:
        output_dir = self.results_dir.parent / "rlvr-output"
        if not output_dir.exists():
            raw_output_dir = self.manifest.config.get("output_dir")
            if raw_output_dir is None:
                return []
            configured_dir = Path(str(raw_output_dir))
            if configured_dir.exists():
                output_dir = configured_dir
            else:
                cwd_output_dir = Path.cwd() / configured_dir
                if not cwd_output_dir.exists():
                    return []
                output_dir = cwd_output_dir

        preferred_state = None
        if self.steps:
            preferred_state = output_dir / f"checkpoint-{self.steps[-1]}" / "trainer_state.json"
        if preferred_state is not None and preferred_state.exists():
            state_path = preferred_state
        else:
            candidates = sorted(
                output_dir.glob("checkpoint-*/trainer_state.json"),
                key=lambda path: int(path.parent.name.split("-")[-1]),
            )
            if not candidates:
                return []
            state_path = candidates[-1]

        try:
            payload = json.loads(state_path.read_text())
        except Exception:
            return []

        return [
            item for item in payload.get("log_history", [])
            if isinstance(item, dict) and item.get("step") is not None
        ]

    @property
    def has_training_history(self) -> bool:
        return bool(self.training_log_history)

    @property
    def train_domain(self) -> str:
        return str(self.manifest.config.get("train_domain", "Math"))

    @property
    def test_domain(self) -> str:
        return str(self.manifest.config.get("test_domain", "Code"))

    def matrix(self, method: str) -> np.ndarray:
        key = method.lower()
        if key == "tracin":
            return self.bundle.tracin_matrix
        if key == "datainf":
            return self.bundle.datainf_matrix
        if key == "fisher":
            if self.bundle.fisher_matrix is None:
                raise ValueError("Fisher influence matrix is not available in this bundle.")
            return self.bundle.fisher_matrix
        raise ValueError(f"Unknown method: {method}")

    def step_matrices(self, method: str) -> dict[int, np.ndarray]:
        key = method.lower()
        if key == "tracin":
            return self.bundle.tracin_step_matrices
        if key == "datainf":
            return self.bundle.datainf_step_matrices
        if key == "fisher":
            return self.bundle.fisher_step_matrices or {}
        raise ValueError(f"Unknown method: {method}")

    def _rank_indices(self, row: np.ndarray, k: int, largest: bool, use_abs: bool) -> np.ndarray:
        values = np.abs(row) if use_abs else row
        if largest:
            order = np.argsort(-values)
        else:
            order = np.argsort(values)
        return order[:k]

    def _entries_for_indices(
        self,
        method: str,
        test_idx: int,
        train_indices: np.ndarray,
    ) -> list[RankedInfluence]:
        row = self.matrix(method)[test_idx]
        entries = []
        for train_idx in train_indices:
            entries.append(
                RankedInfluence(
                    method=method,
                    test_index=test_idx,
                    train_index=int(train_idx),
                    score=float(row[train_idx]),
                    test_prompt=self.test_prompts[test_idx],
                    train_prompt=self.train_prompts[train_idx],
                    train_solution=self.train_solutions[train_idx],
                )
            )
        return entries

    def topk(
        self,
        method: str,
        test_idx: int,
        k: int = 3,
        use_abs: bool = True,
    ) -> list[RankedInfluence]:
        indices = self._rank_indices(self.matrix(method)[test_idx], k, True, use_abs)
        return self._entries_for_indices(method, test_idx, indices)

    def bottomk(
        self,
        method: str,
        test_idx: int,
        k: int = 3,
        use_abs: bool = True,
    ) -> list[RankedInfluence]:
        indices = self._rank_indices(self.matrix(method)[test_idx], k, False, use_abs)
        return self._entries_for_indices(method, test_idx, indices)

    def global_summary(self, method: str) -> dict[str, object]:
        matrix = self.matrix(method)
        return {
            "shape": matrix.shape,
            "min": float(np.min(matrix)),
            "max": float(np.max(matrix)),
            "abs_mean": float(np.mean(np.abs(matrix))),
            "row_abs_sum": np.sum(np.abs(matrix), axis=1).tolist(),
            "col_abs_sum": np.sum(np.abs(matrix), axis=0).tolist(),
        }

    def plot_heatmap(self, method: str, output_path: str | Path):
        matrix = self.matrix(method)
        title = f"Trajectory {method.title()} Influence"
        fig = heatmap_figure(
            matrix,
            self.test_labels,
            self.train_labels,
            title,
            f"Train samples ({self.train_domain})",
            f"Test samples ({self.test_domain})",
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        fig.clf()
        return output

    def plot_agreement(self, output_path: str | Path):
        fig = agreement_scatter_figure(
            self.bundle.tracin_matrix,
            self.bundle.datainf_matrix,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        fig.clf()
        return output

    def default_pairs(self, method: str = "tracin", top_pairs: int = 3) -> list[tuple[int, int]]:
        matrix = np.abs(self.matrix(method)).copy()
        pairs = []
        for _ in range(min(top_pairs, matrix.size)):
            idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            pairs.append((int(idx[0]), int(idx[1])))
            matrix[idx] = -1
        return pairs

    def plot_step_series(
        self,
        pairs: list[tuple[int, int]],
        output_path: str | Path,
    ):
        tracin_steps = self.step_matrices("tracin")
        datainf_steps = self.step_matrices("datainf")
        fisher_steps = self.step_matrices("fisher") if self.bundle.fisher_matrix is not None else {}
        pair_series = []
        for test_idx, train_idx in pairs:
            entry = {
                "title": f"Test {test_idx} ↔ Train {train_idx}",
                "tracin": [
                    tracin_steps.get(step, np.full((1, 1), np.nan))[test_idx, train_idx]
                    if step in tracin_steps else np.nan
                    for step in self.steps
                ],
                "datainf": [
                    datainf_steps.get(step, np.full((1, 1), np.nan))[test_idx, train_idx]
                    if step in datainf_steps else np.nan
                    for step in self.steps
                ],
            }
            if fisher_steps:
                entry["fisher"] = [
                    fisher_steps.get(step, np.full((1, 1), np.nan))[test_idx, train_idx]
                    if step in fisher_steps else np.nan
                    for step in self.steps
                ]
            pair_series.append(entry)
        fig = trajectory_pairs_figure(self.steps, pair_series)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        fig.clf()
        return output

    def plot_gradient_norms(self, output_path: str | Path):
        fig = gradient_norms_figure(
            [item.to_dict() for item in self.manifest.checkpoints]
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        fig.clf()
        return output

    def plot_eval_performance(self, output_path: str | Path):
        fig = eval_performance_figure(
            [item.to_dict() for item in self.manifest.checkpoints]
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        fig.clf()
        return output

    def plot_training_curves(self, output_path: str | Path):
        fig = training_curves_figure(self.training_log_history)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        fig.clf()
        return output

    def build_report(self, top_k: int = 3, bottom_k: int = 0) -> str:
        lines = []
        methods = ["TracIn", "DataInf"]
        if self.bundle.fisher_matrix is not None:
            methods.append("Fisher")
        for method in methods:
            lines.append("")
            lines.append("=" * 60)
            lines.append(f"Top-{top_k} influential train samples per test ({method})")
            lines.append("=" * 60)
            for test_idx, prompt in enumerate(self.test_prompts):
                lines.append("")
                lines.append(f"Test {test_idx}: {textwrap.shorten(prompt, width=100, placeholder='...')}")
                for rank, entry in enumerate(self.topk(method, test_idx, top_k), start=1):
                    lines.append(
                        f"  #{rank}  train {entry.train_index}: score={entry.score:+.3e}  | "
                        f"{textwrap.shorten(entry.train_prompt, width=80, placeholder='...')}"
                    )
                if bottom_k > 0:
                    lines.append(f"  Least-{bottom_k}:")
                    for entry in self.bottomk(method, test_idx, bottom_k):
                        lines.append(
                            f"    train {entry.train_index}: score={entry.score:+.3e}  | "
                            f"{textwrap.shorten(entry.train_prompt, width=80, placeholder='...')}"
                        )

        lines.append("")
        lines.append("=" * 60)
        lines.append("Config summary")
        lines.append("=" * 60)
        for key in [
            "model_id",
            "experiment_mode",
            "influence_mode",
            "learning_rate",
            "max_steps",
            "save_steps",
            "per_device_batch",
            "grad_accum_steps",
            "grpo_beta",
            "grpo_epsilon",
            "g_train",
            "g_test",
            "generation_batch_size",
            "n_math",
            "n_code",
            "n_code_train",
            "n_train_replay",
            "n_train_replay_cap",
            "train_replay_pool_size",
            "train_replay_subset_seed",
            "code_train_split",
            "lambda_damp",
            "batch_history_fingerprint",
            "math_eval_split",
            "math_eval_percent",
            "code_eval_split",
            "code_eval_percent",
            "code_eval_do_sample",
            "code_eval_num_samples",
            "code_eval_temperature",
            "code_eval_top_p",
            "generation_backend",
            "train_gradient_objective",
            "test_gradient_objective",
            "train_geometry_feature",
            "second_order_geometry",
            "fisher_normalize",
            "replay_max_new_tokens",
            "replay_temperature",
            "replay_top_p",
            "vllm_gpu_memory_utilization",
            "vllm_tensor_parallel_size",
            "vllm_max_model_len",
            "vllm_max_num_seqs",
            "vllm_max_lora_rank",
            "vllm_enforce_eager",
            "vllm_training_use_vllm",
            "train_domain",
            "test_domain",
        ]:
            lines.append(f"  {key}: {self.manifest.config.get(key)}")

        if self.has_eval_metrics and self.manifest.checkpoints:
            base = self.manifest.checkpoints[0]
            latest = self.manifest.checkpoints[-1]
            lines.append("")
            lines.append("=" * 60)
            lines.append("Held-out evaluation")
            lines.append("=" * 60)
            if latest.math_eval is not None:
                lines.append(
                    "  math: "
                    f"exact={latest.math_eval.get('accuracy_rate', 0.0):.3f}, "
                    f"reward={latest.math_eval.get('mean_reward', 0.0):.3f}, "
                    f"count={latest.math_eval.get('count', 0)}"
                )
            if latest.code_eval is not None:
                pass_label = latest.code_eval.get("pass_metric", "pass")
                compile_label = latest.code_eval.get("compile_metric", "compile")
                lines.append(
                    "  code: "
                    f"{pass_label}={latest.code_eval.get('pass_rate', 0.0):.3f}, "
                    f"{compile_label}={latest.code_eval.get('compile_rate', 0.0):.3f}, "
                    f"reward={latest.code_eval.get('mean_reward', 0.0):.3f}, "
                    f"count={latest.code_eval.get('count', 0)}"
                )
            if latest.step != base.step:
                delta_lines = []
                if base.math_eval is not None and latest.math_eval is not None:
                    delta_lines.append(
                        "math exact="
                        f"{latest.math_eval.get('accuracy_rate', 0.0) - base.math_eval.get('accuracy_rate', 0.0):+.3f}"
                    )
                if base.code_eval is not None and latest.code_eval is not None:
                    delta_lines.append(
                        f"{latest.code_eval.get('pass_metric', 'code pass')}="
                        f"{latest.code_eval.get('pass_rate', 0.0) - base.code_eval.get('pass_rate', 0.0):+.3f}"
                    )
                    delta_lines.append(
                        f"{latest.code_eval.get('compile_metric', 'code compile')}="
                        f"{latest.code_eval.get('compile_rate', 0.0) - base.code_eval.get('compile_rate', 0.0):+.3f}"
                    )
                if delta_lines:
                    lines.append("  delta vs base: " + ", ".join(delta_lines))

        m = self.manifest
        if m.training_elapsed_s is not None or m.replay_elapsed_s is not None or m.total_elapsed_s is not None:
            lines.append("")
            lines.append("=" * 60)
            lines.append("Timing")
            lines.append("=" * 60)
            if m.training_elapsed_s is not None:
                lines.append(f"  training:  {m.training_elapsed_s:.1f}s")
            if m.replay_elapsed_s is not None:
                lines.append(f"  replay:    {m.replay_elapsed_s:.1f}s")
            if m.total_elapsed_s is not None:
                lines.append(f"  total:     {m.total_elapsed_s:.1f}s")

        return "\n".join(lines)

    def write_report(
        self,
        output_dir: str | Path,
        top_k: int = 3,
        bottom_k: int = 0,
    ) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "report.txt"
        report_path.write_text(self.build_report(top_k=top_k, bottom_k=bottom_k))
        return report_path

    def write_default_artifacts(
        self,
        output_dir: str | Path | None = None,
        top_k: int = 3,
        bottom_k: int = 0,
        top_pairs: int = 3,
    ) -> list[Path]:
        output = Path(output_dir) if output_dir is not None else self.results_dir / "figures"
        output.mkdir(parents=True, exist_ok=True)
        saved = [
            self.plot_heatmap("tracin", output / "tracin_heatmap.png"),
            self.plot_heatmap("datainf", output / "datainf_heatmap.png"),
            self.plot_agreement(output / "tracin_vs_datainf.png"),
            self.plot_step_series(
                self.default_pairs(top_pairs=top_pairs),
                output / "trajectory_top_pairs.png",
            ),
            self.plot_gradient_norms(output / "gradient_norms.png"),
        ]
        if self.bundle.fisher_matrix is not None:
            saved.append(self.plot_heatmap("fisher", output / "fisher_heatmap.png"))
        if self.has_eval_metrics:
            saved.append(self.plot_eval_performance(output / "performance_curves.png"))
        if self.has_training_history:
            saved.append(self.plot_training_curves(output / "training_curves.png"))
        saved.append(self.write_report(output, top_k=top_k, bottom_k=bottom_k))
        return saved
