"""In-training held-out eval callback.

Periodically (every `live_eval_every` steps, default = save_steps) generate on the
disjoint held-out eval set and score with the verifier, logging `eval/accuracy`
(+ per-category) to W&B and a CSV. Across the `if_prune` windows the cadence keys
off the persistent `global_step`, so baseline and if_prune yield comparable
held-out-accuracy-vs-step curves — the fair comparison (training reward is over
different prompts per regime).

Generation uses the live HF policy (independent of the colocated vLLM engine).
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from transformers import TrainerCallback

from .config import ExperimentConfig
from .dist_utils import env_is_main


class LiveEvalCallback(TrainerCallback):
    def __init__(self, cfg: ExperimentConfig, eval_examples: list[dict], tokenizer,
                 device, csv_path: Path, *, every: int):
        self.cfg = cfg
        self.eval_examples = eval_examples
        self.tokenizer = tokenizer
        self.device = device
        self.csv_path = Path(csv_path)
        self.every = int(every)
        self._last_step = -1
        if env_is_main():  # DP: only rank 0 owns the CSV
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="") as f:
                    csv.writer(f).writerow(["step", "n", "accuracy", "per_category_json"])

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Anchor a step-0 baseline (untrained model) so the gate sees the FULL
        held-out rise — on_step_end's first eval is at step=`every`, by which point
        the fast early gains are already banked, so without this the gate can read
        a false NO-GO. Fresh starts only (skip on --resume); also an early canary
        that the eval path works before we sink hours into training."""
        if self.every <= 0 or not self.eval_examples or not state.is_world_process_zero:
            return
        if int(state.global_step) != 0:   # resume: the trajectory already has its anchor
            return
        mdl = model if model is not None else kwargs.get("model")
        if mdl is not None:
            self._last_step = 0
            self._safe_evaluate(mdl, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every <= 0 or not self.eval_examples or model is None:
            return
        if not state.is_world_process_zero:
            return  # DP: rank 0 alone runs the held-out eval + writes CSV/wandb
        step = int(state.global_step)
        if step == self._last_step or step % self.every != 0:
            return
        self._last_step = step
        self._safe_evaluate(model, step)

    def _safe_evaluate(self, model, step: int) -> None:
        """A held-out eval must never crash training — a broken eval path should
        cost one data point, not a multi-hour run. Failures print and continue;
        checkpoints are still saved, so the trajectory can be rebuilt offline."""
        try:
            self._evaluate(model, step)
        except Exception as e:
            print(f"[live-eval] eval at step {step} skipped ({type(e).__name__}: {e})")

    @torch.inference_mode()
    def _evaluate(self, model, step: int) -> None:
        from .evaluate import score_examples  # local import avoids any import cycle

        # DP: the trainer may hand us a DDP/accelerate-wrapped module — .generate
        # and .config live on the inner model, so unwrap (no-op if already plain).
        while hasattr(model, "module"):
            model = model.module

        was_training = model.training
        prev_use_cache = getattr(model.config, "use_cache", None)
        try:
            model.config.use_cache = True
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            model.eval()
            res = score_examples(
                self.eval_examples, model, self.tokenizer, self.cfg, self.device,
                max_new_tokens=self.cfg.live_eval_max_new_tokens,
            )
        finally:
            if prev_use_cache is not None:
                model.config.use_cache = prev_use_cache
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if was_training:
                model.train()

        logs = {"eval/accuracy": res["accuracy"], "eval/n": res["n"]}
        for cat, acc in res["per_category"].items():
            key = cat.lower().replace(" ", "_") or "unknown"
            logs[f"eval/acc_{key}"] = acc
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(logs, step=step)
        except Exception:
            pass
        with self.csv_path.open("a", newline="") as f:
            import json
            csv.writer(f).writerow(
                [step, res["n"], f"{res['accuracy']:.6f}", json.dumps(res["per_category"])]
            )
        cats = ", ".join(f"{c}={a:.3f}" for c, a in res["per_category"].items())
        print(f"[live-eval] step {step}: held-out acc={res['accuracy']:.4f} "
              f"(n={res['n']}){' | ' + cats if cats else ''}")
