from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from trl import GRPOTrainer

from analysis.loader import load_batch_history, save_batch_history


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
    def __init__(self, *args, history_output_dir: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_output_dir = (
            Path(history_output_dir)
            if history_output_dir is not None
            else Path(self.args.output_dir)
        )
        self._historical_batch_steps: list[dict[str, object]] = []
        self._pending_historical_rows: list[torch.Tensor] = []
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

    def _finalize_historical_step(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            self._pending_historical_rows.clear()
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
