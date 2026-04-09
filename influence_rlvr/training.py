from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from trl import GRPOTrainer

from analysis.loader import save_batch_history


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
