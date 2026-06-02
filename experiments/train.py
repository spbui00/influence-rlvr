"""Driver for the scaled GRPO + IF-pruning experiment.

Two regimes (set via --regime):

  baseline   Straight GRPO for `max_steps` on the full (sampled) WebInstruct pool.

  if_prune   Train to `prune_step`, score the pool's influence on the held-out
             target set, keep the top `keep_fraction` by influence, then resume
             from the prune-step checkpoint and continue to `max_steps` on the
             pruned subset.

Run:
    python -m experiments.train --regime baseline   --run-name qwen3_4b_base
    python -m experiments.train --regime if_prune    --run-name qwen3_4b_ifprune

All flags mirror ExperimentConfig fields (see experiments/config.py); pass
--config path/to/config.json to start from a saved config.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from influence_rlvr import clear_cache, detect_device
from influence_rlvr.generation import clear_vllm_engine_cache
from influence_rlvr.rewards import format_guardrail_reward_func

from .config import ExperimentConfig
from .data import load_eval_benchmark, load_if_target_set, load_train_pool
from .influence import compute_pool_influence
from .live_eval import LiveEvalCallback
from .verifier import make_verifier_reward_func


def _save_base_checkpoint(peft_model, tokenizer, output_dir: Path) -> Path:
    """Write checkpoint-0 so the influence/checkpoint machinery has a step-0 anchor."""
    ckpt = output_dir / "checkpoint-0"
    ckpt.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    (ckpt / "trainer_state.json").write_text(json.dumps({
        "global_step": 0,
        "log_history": [{"step": 0, "learning_rate": 0.0}],
    }, indent=2))
    return ckpt


def build_model(cfg: ExperimentConfig, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=dtype).to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    return model, tokenizer


def build_reward_funcs(cfg: ExperimentConfig) -> list:
    funcs = []
    if cfg.use_format_guardrail:
        funcs.append(format_guardrail_reward_func)
    funcs.append(make_verifier_reward_func(cfg))
    return funcs


def prompts_per_step(cfg: ExperimentConfig, *, num_processes: int = 1) -> int:
    """Unique prompts whose rollouts contribute to one optimizer step.

    GRPO still trains in *batches*: each step's gradient is over the full
    effective batch of (per_device_batch × grad_accum × num_processes)
    completion-rows. Those rows are grouped into `num_generations` completions
    per prompt, so the number of distinct prompts per step is
    effective_batch / num_generations. Raise per_device_batch / grad_accum (or
    lower g_train) to put more distinct prompts in each batch.
    """
    effective = cfg.per_device_batch * cfg.grad_accum * num_processes
    return max(1, effective // cfg.g_train)


def make_grpo_config(cfg: ExperimentConfig, *, max_steps: int, shuffle: bool = True) -> GRPOConfig:
    device = detect_device()
    kw: dict = {
        "shuffle_dataset": shuffle,
        "output_dir": str(cfg.grpo_output_dir),
        "seed": cfg.seed,
        "report_to": "wandb",
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.per_device_batch,
        "gradient_accumulation_steps": cfg.grad_accum,
        "max_steps": max_steps,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": None,
        "bf16": device.type == "cuda",
        "use_vllm": cfg.use_vllm,
        "num_generations": cfg.g_train,
        "beta": cfg.grpo_beta,
        "epsilon": cfg.grpo_epsilon,
        "importance_sampling_level": "token",
        "scale_rewards": "group",
        "loss_type": "grpo",
        "max_completion_length": cfg.max_completion_length,
        # NOTE: max_prompt_length is not a GRPOConfig field in current TRL; prompt
        # truncation (if needed for very long WebInstruct questions) is handled at
        # tokenization in evaluate.py / can be added in data.py.
    }
    if cfg.generation_batch_size is not None:
        kw["generation_batch_size"] = cfg.generation_batch_size
    if cfg.use_vllm:
        kw["vllm_mode"] = cfg.vllm_mode
        if cfg.vllm_mode == "server":
            # Engine lives in a separate `trl vllm-serve` process; just point at it.
            kw["vllm_server_host"] = cfg.vllm_server_host
            kw["vllm_server_port"] = cfg.vllm_server_port
        else:  # colocate
            kw["vllm_gpu_memory_utilization"] = cfg.vllm_gpu_memory_utilization
            kw["vllm_enable_sleep_mode"] = cfg.vllm_enable_sleep_mode
            if cfg.vllm_max_model_len is not None:
                kw["vllm_max_model_length"] = cfg.vllm_max_model_len
    # GRPOConfig fields differ across TRL versions (e.g. max_prompt_length,
    # importance_sampling_level, vllm_mode were added/renamed). Keep only fields
    # this installed TRL actually accepts; warn about the rest.
    import dataclasses
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in kw if k not in valid)
    if dropped:
        print(f"[grpo-config] this TRL ({_trl_version()}) ignores: {dropped}")
    kw = {k: v for k, v in kw.items() if k in valid}
    return GRPOConfig(**kw)


def _trl_version() -> str:
    try:
        import trl
        return getattr(trl, "__version__", "?")
    except Exception:
        return "?"


def make_live_eval_callback(cfg, tokenizer, device):
    """Held-out eval callback (or None). Uses the disjoint eval partition."""
    if not cfg.live_eval:
        return None
    examples = load_eval_benchmark(cfg.live_eval_benchmark, cfg, cfg.live_eval_examples)
    if not examples:
        print("  [live-eval] no held-out eval examples; disabling live eval.")
        return None
    every = cfg.live_eval_every if cfg.live_eval_every > 0 else cfg.save_steps
    return LiveEvalCallback(
        cfg, examples, tokenizer, device,
        csv_path=cfg.run_dir / "live_eval.csv", every=every,
    )


def build_trainer(cfg, model, tokenizer, train_dataset, *, max_steps, shuffle=True,
                  callbacks=None):
    # Stock GRPOTrainer (not influence_rlvr.HistoricalBatchGRPOTrainer): the
    # custom subclass overrides TRL private methods that changed in TRL 1.x, and
    # the CG/DENSE influence here never reads its batch-history logging.
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=build_reward_funcs(cfg),
        args=make_grpo_config(cfg, max_steps=max_steps, shuffle=shuffle),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    for cb in (callbacks or []):
        trainer.add_callback(cb)
    return trainer


def _repeat_to_length(order: np.ndarray, n_picks: int) -> list[int]:
    """Tile a ranked ordering until it has n_picks indices (toy build_schedule)."""
    n = order.shape[0]
    if n_picks <= n:
        return [int(i) for i in order[:n_picks]]
    reps = (n_picks + n - 1) // n
    return [int(i) for i in np.tile(order, reps)[:n_picks]]


def _ranked_order(scores: np.ndarray, *, selection: str, seed: int) -> np.ndarray:
    """Full-length consumption order over the pool (most-trained-first)."""
    if selection == "if-guided":
        return np.argsort(-scores)            # most influential first
    if selection == "anti-if":
        return np.argsort(scores)             # least influential first (ablation)
    if selection == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(len(scores))   # size-matched control
    raise ValueError(f"Unknown selection {selection!r}")


def run_baseline(cfg, model, tokenizer, train_pool, device):
    print("\n" + "=" * 80)
    print(f"BASELINE GRPO — {cfg.max_steps} steps on {len(train_pool)} prompts")
    print("=" * 80)
    live_eval = make_live_eval_callback(cfg, tokenizer, device)
    trainer = build_trainer(cfg, model, tokenizer, train_pool, max_steps=cfg.max_steps,
                            callbacks=[live_eval] if live_eval else None)
    t0 = time.time()
    trainer.train()
    print(f"Baseline training finished in {time.time() - t0:.1f}s")
    clear_cache(device)


def _window_boundaries(cfg) -> list[int]:
    """Step boundaries [0, t1, t2, ..., max_steps]. Interior points t_i are the
    recompute/reselect triggers. With if_recompute_every<=0 there is a single
    trigger at prune_step (one-shot prune)."""
    if cfg.if_recompute_every and cfg.if_recompute_every > 0:
        triggers, s = [], cfg.prune_step
        while s < cfg.max_steps:
            triggers.append(s)
            s += cfg.if_recompute_every
    else:
        triggers = [cfg.prune_step]
    out: list[int] = []
    for b in [0, *triggers, cfg.max_steps]:
        if not out or b > out[-1]:
            out.append(b)
    return out


def run_if_prune(cfg, model, tokenizer, train_pool, device):
    # if_prune builds one GRPOTrainer per window. With vLLM *colocate* the engine
    # lives in this process and its CuMem allocator is one-per-process, so a 2nd
    # window's init asserts → fall back to HF. With vLLM *server* the engine is a
    # separate process (trl vllm-serve), so every window just reconnects over HTTP
    # and vLLM works fine across windows.
    if cfg.use_vllm and cfg.vllm_mode == "colocate":
        print("[if_prune] vLLM colocate can't re-init across windows in one "
              "process; using HF generation for if_prune training. "
              "Use --vllm-mode server (separate engine process) for vLLM here.")
        cfg.use_vllm = False
    boundaries = _window_boundaries(cfg)
    triggers = boundaries[1:-1]
    target_set = load_if_target_set(cfg)
    if_dir = cfg.run_dir / "influence"
    if_dir.mkdir(parents=True, exist_ok=True)

    pps = prompts_per_step(cfg)
    pool = len(train_pool)
    full_cov_steps = -(-pool // pps)  # ceil(pool / pps)
    print("\n" + "=" * 80)
    print(f"IF-PRUNE (dynamic, ranked-pop) — windows {boundaries}, recompute@{triggers}")
    print(f"  {pps} prompts/step | pool={pool} | a window of W steps pops the top "
          f"W×{pps} ranked prompts in order")
    print(f"  full coverage of the pool would need ~{full_cov_steps} steps/window "
          f"(selection={cfg.selection}, method={cfg.if_method})")
    print(f"  IF target set: {len(target_set)} held-out prompts")
    print("=" * 80)

    live_eval = make_live_eval_callback(cfg, tokenizer, device)
    prev_ckpt: str | None = None
    for w in range(len(boundaries) - 1):
        start, end = boundaries[w], boundaries[w + 1]
        if end <= start:
            continue
        window_steps = end - start
        shuffle = True

        if w == 0:
            # Warm-up window: full pool, shuffled, before any influence exists.
            dataset = train_pool
            print(f"\n[window {w}] warm-up {start}->{end} on full pool "
                  f"({len(dataset)} prompts, shuffled)")
        else:
            # Score the live model (exactly at step `start`), rank the WHOLE pool,
            # and build an in-order consumption schedule: pop the top n_picks
            # ranked prompts (the toy's build_schedule). shuffle off so TRL
            # consumes them in that order; the long tail is simply never reached.
            print(f"\n[window {w}] scoring pool influence at step {start} "
                  f"({cfg.if_method})...")
            scores = compute_pool_influence(
                cfg, model, tokenizer, train_pool, target_set, device,
                checkpoint_step=start, save_dir=if_dir / f"step{start}",
            )
            order = _ranked_order(scores, selection=cfg.selection, seed=cfg.seed + start)
            n_picks = window_steps * pps
            schedule = _repeat_to_length(order, n_picks)
            np.save(if_dir / f"ranked_order_step{start}.npy", order)
            np.save(if_dir / f"schedule_step{start}.npy", np.array(schedule))
            dataset = train_pool.select(schedule)   # ranked order, length n_picks
            uniq = sorted(set(schedule))
            n_unique = len(uniq)
            # Which DOMAINS did the influence pick? (the cross-domain observable:
            # e.g. did a CS target pull in Math/Finance examples?)
            if "category" in train_pool.column_names:
                cats = [train_pool[i]["category"] for i in uniq]
                sel = dict(Counter(cats))
                with (if_dir / f"selected_categories_step{start}.json").open("w") as f:
                    json.dump(sel, f, indent=2)
                cat_str = f" | selected-by-IF categories: {sel}"
            else:
                cat_str = ""
            print(f"[window {w}] popping top {n_picks} of {pool} ranked prompts "
                  f"({n_unique} unique, {n_unique / pool:.0%} of pool); train {start}->{end}"
                  f"{cat_str}")
            shuffle = False
            # Influence (CG) disables grad checkpointing; restore for training.
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            model.config.use_cache = False

        trainer = build_trainer(cfg, model, tokenizer, dataset, max_steps=end,
                                shuffle=shuffle, callbacks=[live_eval] if live_eval else None)
        t0 = time.time()
        if prev_ckpt is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=prev_ckpt)
        print(f"[window {w}] trained to {end} in {time.time() - t0:.1f}s")

        ckpt_end = cfg.grpo_output_dir / f"checkpoint-{end}"
        if not ckpt_end.is_dir():
            trainer.save_model(str(ckpt_end))
        prev_ckpt = str(ckpt_end)

        del trainer
        clear_cache(device)
        clear_vllm_engine_cache()


def main(argv: list[str] | None = None) -> None:
    cfg = ExperimentConfig.from_cli(argv)
    os.environ.setdefault("WANDB_PROJECT", "influence-rlvr-scaled")
    os.environ["WANDB_NAME"] = f"{cfg.run_name}-{cfg.regime}-seed{cfg.seed}"
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    set_seed(cfg.seed)

    device = detect_device()
    print(f"Device: {device} | model: {cfg.model_id} | regime: {cfg.regime}")
    cfg.grpo_output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()
    print(f"Config saved to {cfg.run_dir / 'config.json'}")

    model, tokenizer = build_model(cfg, device)
    _save_base_checkpoint(model, tokenizer, cfg.grpo_output_dir)

    print("\nLoading training pool (WebInstruct-verified, domains="
          f"{','.join(cfg.domains)})...")
    train_pool = load_train_pool(cfg)
    print(f"  Train pool: {len(train_pool)} prompts")

    if cfg.regime == "baseline":
        run_baseline(cfg, model, tokenizer, train_pool, device)
    else:
        run_if_prune(cfg, model, tokenizer, train_pool, device)

    print("\nDone. Checkpoints + history under "
          f"{cfg.grpo_output_dir}\nNext: python -m experiments.evaluate "
          f"--run-name {cfg.run_name}")


if __name__ == "__main__":
    main()
