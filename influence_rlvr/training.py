from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import gather_object
from trl import GRPOTrainer

from analysis.loader import load_batch_history, save_batch_history
from .rollout_cache import TrainingRolloutCacheWriter


def _middle_truncate_token_ids(token_ids: list[int], max_length: int) -> list[int]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if len(token_ids) <= max_length:
        return list(token_ids)
    head = max_length // 2
    tail = max_length - head
    if head <= 0:
        return list(token_ids[-tail:])
    if tail <= 0:
        return list(token_ids[:head])
    return list(token_ids[:head]) + list(token_ids[-tail:])


def _filter_historical_step_records(
    step_records: list[dict[str, object]],
    *,
    max_step: int | None = None,
) -> tuple[list[dict[str, object]], int]:
    if max_step is None:
        return list(step_records), 0
    kept: list[dict[str, object]] = []
    dropped = 0
    for record in step_records:
        step = int(record["step"])
        if step <= max_step:
            kept.append(record)
        else:
            dropped += 1
    return kept, dropped


class HistoricalBatchGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        history_output_dir: str | Path | None = None,
        rollout_cache_dir: str | Path | None = None,
        enable_rollout_cache: bool = False,
        rollout_cache_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.history_output_dir = (
            Path(history_output_dir)
            if history_output_dir is not None
            else Path(self.args.output_dir)
        )
        self._historical_batch_steps: list[dict[str, object]] = []
        self._pending_historical_rows: list[torch.Tensor] = []
        self.rollout_cache_dir = (
            Path(rollout_cache_dir)
            if rollout_cache_dir is not None
            else self.history_output_dir / "rollout_cache"
        )
        self._rollout_cache_writer = (
            TrainingRolloutCacheWriter(
                self.rollout_cache_dir,
                config=rollout_cache_config,
            )
            if enable_rollout_cache
            else None
        )
        self._pending_rollout_microbatches: list[dict[str, Any]] = []
        self._prompt_truncation_notice_count = 0

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        train_indices = [example.get("train_index") for example in inputs]
        present = [index is not None for index in train_indices]
        if any(present) and not all(present):
            raise ValueError("Mixed train_index presence in GRPO batch.")
        if all(present):
            output["train_index"] = torch.tensor(
                train_indices,
                device=output["prompt_ids"].device,
                dtype=torch.long,
            )
        return output

    def _tokenize_prompts(self, prompts):
        prompt_ids, images, multimodal_fields = super()._tokenize_prompts(prompts)
        max_prompt_tokens = self._max_prompt_token_budget()
        if max_prompt_tokens is None:
            return prompt_ids, images, multimodal_fields

        longest = 0
        truncated = 0
        adjusted_prompt_ids = []
        for ids in prompt_ids:
            longest = max(longest, len(ids))
            if len(ids) > max_prompt_tokens:
                truncated += 1
                adjusted_prompt_ids.append(
                    _middle_truncate_token_ids(ids, max_prompt_tokens)
                )
            else:
                adjusted_prompt_ids.append(ids)

        if (
            truncated
            and self.accelerator.is_main_process
            and self._prompt_truncation_notice_count < 8
        ):
            self._prompt_truncation_notice_count += 1
            print(
                "Warning: truncated "
                f"{truncated}/{len(prompt_ids)} overlong prompt(s) to "
                f"{max_prompt_tokens} tokens (raw max={longest}) to fit the "
                "model context window.",
                flush=True,
            )
        return adjusted_prompt_ids, images, multimodal_fields

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._record_historical_rows(inputs)
        self._record_rollout_microbatch(inputs)
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(self, model, inputs, num_items_in_batch):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            self._finalize_historical_step(int(self.state.global_step))
        return loss

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        if self.accelerator.is_main_process:
            save_batch_history(self._historical_batch_steps, self.history_output_dir)
        if self._rollout_cache_writer is not None:
            self._rollout_cache_writer.close()
        return result

    def load_existing_history(
        self,
        *,
        max_step: int | None = None,
    ) -> tuple[int, int]:
        manifest = load_batch_history(self.history_output_dir)
        if manifest is None:
            self._historical_batch_steps = []
            return 0, 0

        existing_records = [
            {
                "step": int(item.step),
                "total_rows": int(item.total_rows),
                "microbatch_count": (
                    None
                    if item.microbatch_count is None
                    else int(item.microbatch_count)
                ),
                "train_index_counts": {
                    int(key): int(value)
                    for key, value in item.train_index_counts.items()
                },
            }
            for item in manifest.steps
        ]
        filtered, dropped = _filter_historical_step_records(
            existing_records,
            max_step=max_step,
        )
        self._historical_batch_steps = filtered
        self._pending_historical_rows.clear()
        self._pending_rollout_microbatches.clear()
        if self._rollout_cache_writer is not None:
            self._rollout_cache_writer.prepare_for_resume(max_step=max_step)
        return len(filtered), dropped

    def _record_historical_rows(self, inputs: dict[str, torch.Tensor]) -> None:
        if not self.model.training:
            return
        train_index = inputs.get("train_index")
        if train_index is None:
            return
        if train_index.ndim == 0:
            train_index = train_index.unsqueeze(0)
        gathered = self.accelerator.gather(train_index.detach())
        if self.accelerator.is_main_process:
            self._pending_historical_rows.append(gathered.cpu())

    def _record_rollout_microbatch(self, inputs: dict[str, Any]) -> None:
        if not self.model.training or self._rollout_cache_writer is None:
            return
        prompt_ids = inputs.get("prompt_ids")
        completion_ids = inputs.get("completion_ids")
        prompt_mask = inputs.get("prompt_mask")
        completion_mask = inputs.get("completion_mask")
        advantages = inputs.get("advantages")
        if (
            prompt_ids is None
            or completion_ids is None
            or prompt_mask is None
            or completion_mask is None
            or advantages is None
        ):
            return

        local_record = {
            "num_items_in_batch": _maybe_scalar_int(inputs.get("num_items_in_batch")),
            "rows": _serialize_rollout_rows(inputs),
        }
        gathered_parts = gather_object([local_record])
        if not self.accelerator.is_main_process:
            return

        merged_rows: list[dict[str, Any]] = []
        num_items_total = 0
        saw_num_items = False
        for part in gathered_parts:
            if not isinstance(part, dict):
                continue
            rows = part.get("rows")
            if isinstance(rows, list):
                merged_rows.extend(rows)
            value = part.get("num_items_in_batch")
            if value is not None:
                num_items_total += int(value)
                saw_num_items = True

        self._pending_rollout_microbatches.append(
            _pack_rollout_microbatch(
                merged_rows,
                num_items_in_batch=(num_items_total if saw_num_items else None),
            )
        )

    def _finalize_historical_step(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            self._pending_historical_rows.clear()
            self._pending_rollout_microbatches.clear()
            return
        if not self._pending_historical_rows:
            return

        flat = torch.cat(self._pending_historical_rows).tolist()
        counts = Counter(int(index) for index in flat)
        self._historical_batch_steps.append({
            "step": step,
            "total_rows": len(flat),
            "microbatch_count": len(self._pending_historical_rows),
            "train_index_counts": dict(sorted(counts.items())),
        })
        self._pending_historical_rows.clear()
        save_batch_history(self._historical_batch_steps, self.history_output_dir)
        self._finalize_rollout_step(step)

    def _finalize_rollout_step(self, step: int) -> None:
        if self._rollout_cache_writer is None:
            self._pending_rollout_microbatches.clear()
            return
        if not self._pending_rollout_microbatches:
            return
        if not self._should_cache_rollout_step(step):
            self._pending_rollout_microbatches.clear()
            return

        total_rows = sum(
            len(microbatch.get("rows", []))
            for microbatch in self._pending_rollout_microbatches
        )
        train_counts = Counter()
        for microbatch in self._pending_rollout_microbatches:
            for row in microbatch.get("rows", []):
                train_index = row.get("train_index")
                if train_index is None:
                    continue
                train_counts[int(train_index)] += 1

        step_record = {
            "schema_version": 1,
            "kind": "training_rollout_step",
            "step": int(step),
            "checkpoint_step": int(step),
            "checkpoint_dir_name": f"checkpoint-{int(step)}",
            "total_rows": int(total_rows),
            "microbatch_count": len(self._pending_rollout_microbatches),
            "train_index_counts": dict(sorted(train_counts.items())),
            "microbatches": list(self._pending_rollout_microbatches),
        }
        self._rollout_cache_writer.append_step(step_record)
        self._pending_rollout_microbatches.clear()

    def _should_cache_rollout_step(self, step: int) -> bool:
        save_steps = getattr(self.args, "save_steps", None)
        if save_steps is None:
            return False
        try:
            save_steps_int = int(save_steps)
        except (TypeError, ValueError):
            return False
        if save_steps_int <= 0:
            return False
        return int(step) % save_steps_int == 0

    def _max_prompt_token_budget(self) -> int | None:
        max_completion = getattr(self.args, "max_completion_length", None)
        if max_completion is None:
            return None

        candidates: list[int] = []
        vllm_limit = getattr(self.args, "vllm_max_model_length", None)
        if isinstance(vllm_limit, int) and vllm_limit > 0:
            candidates.append(vllm_limit)

        tokenizer_limit = getattr(self.processing_class, "model_max_length", None)
        if (
            isinstance(tokenizer_limit, int)
            and tokenizer_limit > 0
            and tokenizer_limit < 1_000_000
        ):
            candidates.append(tokenizer_limit)

        model_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            candidates.append(model_limit)

        if not candidates:
            return None
        return max(1, min(candidates) - int(max_completion))


def _maybe_scalar_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return int(value.detach().cpu().item())
    return int(value)


def _trim_int_tokens(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> list[int]:
    values = tokens.detach().cpu().to(dtype=torch.long)
    if mask is not None:
        length = int(mask.detach().cpu().to(dtype=torch.long).sum().item())
        values = values[:length]
    return [int(x) for x in values.tolist()]


def _trim_numeric_values(values: torch.Tensor, length: int | None = None) -> float | list[float]:
    data = values.detach().cpu()
    if data.ndim == 0:
        return float(data.item())
    flat = data.reshape(-1)
    if length is not None and flat.numel() > length:
        flat = flat[:length]
    numbers = flat.to(dtype=torch.float32).tolist()
    if len(numbers) == 1:
        return float(numbers[0])
    return [float(x) for x in numbers]


def _serialize_rollout_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_ids = inputs["prompt_ids"]
    prompt_mask = inputs["prompt_mask"]
    completion_ids = inputs["completion_ids"]
    completion_mask = inputs["completion_mask"]
    advantages = inputs["advantages"]
    train_index = inputs.get("train_index")

    rows: list[dict[str, Any]] = []
    batch_size = int(prompt_ids.shape[0])
    for row_idx in range(batch_size):
        row: dict[str, Any] = {
            "prompt_token_ids": _trim_int_tokens(prompt_ids[row_idx], prompt_mask[row_idx]),
            "completion_token_ids": _trim_int_tokens(
                completion_ids[row_idx],
                completion_mask[row_idx],
            ),
            "advantage": _trim_numeric_values(advantages[row_idx]),
        }
        if train_index is not None:
            row["train_index"] = int(train_index[row_idx].detach().cpu().item())
        else:
            row["train_index"] = None
        rows.append(row)
    return rows


def _pack_rollout_microbatch(
    rows: list[dict[str, Any]],
    *,
    num_items_in_batch: int | None = None,
) -> dict[str, Any]:
    prompts: list[list[int]] = []
    prompt_lookup: dict[tuple[int, ...], int] = {}
    packed_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt_token_ids = tuple(int(x) for x in row.pop("prompt_token_ids"))
        prompt_ref = prompt_lookup.get(prompt_token_ids)
        if prompt_ref is None:
            prompt_ref = len(prompts)
            prompts.append(list(prompt_token_ids))
            prompt_lookup[prompt_token_ids] = prompt_ref
        packed = dict(row)
        packed["prompt_ref"] = prompt_ref
        packed_rows.append(packed)
    payload = {
        "prompts": prompts,
        "rows": packed_rows,
    }
    if num_items_in_batch is not None:
        payload["num_items_in_batch"] = int(num_items_in_batch)
    return payload
