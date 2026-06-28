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

from influence_rlvr import clear_cache, detect_device, load_adapter_checkpoint
from influence_rlvr.generation import clear_vllm_engine_cache
from influence_rlvr.rewards import format_guardrail_reward_func

from .config import DOMAIN_TO_CATEGORIES, ExperimentConfig
from .data import load_eval_benchmark, load_if_target_set, load_train_pool
from .dist_utils import env_is_main
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
        # RLVR wants a STEADY step size: data is on-policy and the model keeps improving,
        # so annealing to 0 (HF's default "linear") stalls the policy. Worse, run_if_prune
        # builds one trainer PER window (max_steps=window_end), so that decay RESTARTS each
        # window — cratering the LR to ~0 at every recompute (a sawtooth that gave the IF arm
        # ~1/3 the single-trainer baseline's effective LR budget). Constant matches verl/GR
        # practice and puts every arm on the same footing.
        "lr_scheduler_type": "constant",
        "per_device_train_batch_size": cfg.per_device_batch,
        "gradient_accumulation_steps": cfg.grad_accum,
        "max_steps": max_steps,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": None,
        # NCCL collective timeout (s). Set HERE (transformers TrainingArguments ->
        # Accelerator InitProcessGroupKwargs), NOT in accelerate_ddp.yaml — the
        # Alliance accelerate build rejects a `ddp_timeout` YAML key ("unknown keys").
        "ddp_timeout": cfg.ddp_timeout,
        "bf16": device.type == "cuda",
        "use_vllm": cfg.use_vllm,
        "num_generations": cfg.g_train,
        "beta": cfg.grpo_beta,
        "epsilon": cfg.grpo_epsilon,
        "epsilon_high": cfg.grpo_epsilon_high,   # clip-higher (GR-4B); filtered if unsupported
        "temperature": cfg.grpo_temperature,     # rollout temperature (GR-4B = 0.7)
        "importance_sampling_level": "token",
        "scale_rewards": cfg.grpo_scale_rewards,
        "loss_type": cfg.grpo_loss_type,
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
            kw["vllm_server_port"] = cfg.vllm_server_port      # HTTP API
            kw["vllm_group_port"] = cfg.vllm_group_port        # weight-sync NCCL group (must != HTTP)
            kw["vllm_server_timeout"] = cfg.vllm_server_timeout
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


def _ranked_order(scores: np.ndarray, *, selection: str, seed: int) -> np.ndarray:
    """Full-length selection order over the pool (kept set = first keep_fraction·pool)."""
    if selection == "if-guided":
        return np.argsort(-scores)            # most influential first
    if selection == "anti-if":
        return np.argsort(scores)             # least influential first (ablation)
    if selection == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(len(scores))   # size-matched control
    raise ValueError(f"Unknown selection {selection!r}")


def _resume_checkpoint(cfg) -> bool | None:
    """For `trainer.train(resume_from_checkpoint=...)`: return True (HF picks the
    latest checkpoint in the run dir) when --resume is set AND a real (>0) checkpoint
    exists; else None (fresh start). Lets a time-limit-killed run be resubmitted, or
    a short run be extended (raise --max-steps, --resume, continue)."""
    if not cfg.resume:
        return None
    steps = [int(p.name.split("-")[-1]) for p in cfg.grpo_output_dir.glob("checkpoint-*")
             if p.name.split("-")[-1].isdigit() and int(p.name.split("-")[-1]) > 0]
    if not steps:
        if env_is_main():
            print("[resume] --resume set but no >0 checkpoint found; starting fresh")
        return None
    if env_is_main():
        print(f"[resume] continuing from checkpoint-{max(steps)} in {cfg.grpo_output_dir}")
    return True


def _assert_resume_compatible(cfg) -> None:
    """Refuse a --resume that would continue a DIFFERENT config. HF happily resumes
    a checkpoint under whatever args you pass, so resubmitting the same --run-name
    with a changed model/LoRA/batch/LR/GRPO/reward/data would silently mix state (and
    main() would overwrite config.json, erasing the record). Compare the run dir's
    saved config.json against the current config and abort on any training-critical
    mismatch; only max_steps/save_steps and eval/verifier-/vllm-infra knobs may
    change (extend a run; swap monitoring or serving infra)."""
    saved_path = cfg.run_dir / "config.json"
    if not (cfg.resume and saved_path.exists()):
        return
    saved = json.loads(saved_path.read_text())
    current = cfg.to_dict()
    may_differ = {
        "resume", "max_steps", "save_steps",
        "eval_benchmarks", "eval_max_examples", "eval_max_new_tokens",
        "eval_temperature", "eval_top_p",
        "live_eval", "live_eval_every", "live_eval_examples",
        "live_eval_max_new_tokens", "live_eval_benchmark",
        "verifier_backend", "verifier_server_host", "verifier_server_port",
        "verifier_device", "verifier_batch_size", "verifier_max_new_tokens",
        "verifier_max_model_len",
        "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization", "vllm_max_model_len",
        "vllm_enable_sleep_mode", "vllm_server_host", "vllm_server_port",
        # IF-scoring throughput / infra — these change how FAST scoring runs, never
        # which examples get selected or the training trajectory, so they may differ
        # on resume. (if_method/if_grad/if_recompute_every DO affect selection — kept.)
        "if_score_batch", "if_logps_micro_batch", "if_vllm_gen", "if_vllm_gpu_util",
        "if_jvp", "if_jvp_batch", "ddp_timeout",
    }
    diffs = {k: (saved[k], current[k]) for k in current
             if k in saved and k not in may_differ and saved[k] != current[k]}
    if diffs:
        lines = "\n".join(f"    {k}: was {s!r}, now {n!r}" for k, (s, n) in sorted(diffs.items()))
        raise SystemExit(
            f"\n[resume] REFUSING to resume run '{cfg.run_name}' — its saved config.json "
            f"disagrees on training-critical fields:\n{lines}\n"
            "  A resume must keep the same model / LoRA / batch / LR / GRPO / reward / data.\n"
            "  Use a NEW --run-name for a different config (or drop --resume to start fresh).\n"
        )


def run_baseline(cfg, model, tokenizer, train_pool, device):
    print("\n" + "=" * 80)
    print(f"BASELINE GRPO — {cfg.max_steps} steps on {len(train_pool)} prompts")
    print("=" * 80)
    live_eval = make_live_eval_callback(cfg, tokenizer, device)
    trainer = build_trainer(cfg, model, tokenizer, train_pool, max_steps=cfg.max_steps,
                            callbacks=[live_eval] if live_eval else None)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=_resume_checkpoint(cfg))
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
    keep = max(1, int(round(cfg.keep_fraction * pool)))
    print("\n" + "=" * 80)
    print(f"IF-PRUNE (dynamic keep-fraction) — windows {boundaries}, recompute@{triggers}")
    print(f"  {pps} prompts/step | pool={pool} | keep top {cfg.keep_fraction:.0%} "
          f"(~{keep}) by influence each re-rank, train SHUFFLED")
    print(f"  selection={cfg.selection}, method={cfg.if_method}, grad={cfg.if_grad}")
    print(f"  IF target set: {len(target_set)} held-out prompts")
    print("=" * 80)

    live_eval = make_live_eval_callback(cfg, tokenizer, device)
    prev_ckpt: str | None = None
    model_step = 0  # in-memory model's training step; lags `start` when --resume skips windows
    for w in range(len(boundaries) - 1):
        start, end = boundaries[w], boundaries[w + 1]
        if end <= start:
            continue
        window_steps = end - start
        shuffle = True

        # --resume: if this window already trained to completion (its end-checkpoint
        # exists), skip it and just advance the anchor. The in-memory model stays
        # stale (model_step < start) but that's harmless as long as the next
        # non-skipped window reuses a saved ranking instead of live-scoring — the
        # else-branch below guards that case explicitly.
        done_ckpt = cfg.grpo_output_dir / f"checkpoint-{end}"
        if cfg.resume and done_ckpt.is_dir():
            print(f"[window {w}] resume: checkpoint-{end} exists — skip training {start}->{end}")
            prev_ckpt = str(done_ckpt)
            continue

        if w == 0:
            # Warm-up window: full pool, shuffled, before any influence exists.
            dataset = train_pool
            print(f"\n[window {w}] warm-up {start}->{end} on full pool "
                  f"({len(dataset)} prompts, shuffled)")
        else:
            # Score the live model (exactly at step `start`), rank the WHOLE pool by
            # influence on the target, keep the top keep_fraction, and train it
            # shuffled (built below). Re-ranking each window lets a learned prompt's
            # gradient shrink and drop out, rotating in under-learned ones.
            order_path = if_dir / f"ranked_order_step{start}.npy"
            if cfg.resume and order_path.exists():
                # Crash recovered between windows: the step-`start` ranking is on
                # disk, so skip the (expensive) 24k rescore. Training below loads
                # prev_ckpt, so the stale in-memory model never feeds selection here.
                order = np.load(order_path)
                print(f"\n[window {w}] resume: reusing saved ranking {order_path.name} "
                      f"({len(order)} prompts) — skip rescoring")
            elif cfg.selection == "in-domain":
                # "just train on the target domain" heuristic baseline: keep only
                # target-category prompts from the SAME pool (so the carve/eval matches
                # the IF arm exactly). No influence needed → no scoring. Target-domain
                # indices first (shuffled), then the rest, so kept[:keep] is in-domain
                # until exhausted (size-matched to the IF arm via keep_fraction).
                tgt_cats = set()
                for d in cfg.webinstruct_test_domains:
                    tgt_cats.update(DOMAIN_TO_CATEGORIES.get(d, ()))
                cats_col = train_pool["category"]
                rng = np.random.default_rng(cfg.seed + start)
                in_dom = [i for i in range(pool) if cats_col[i] in tgt_cats]
                rest = [i for i in range(pool) if cats_col[i] not in tgt_cats]
                rng.shuffle(in_dom)
                rng.shuffle(rest)
                order = np.array(in_dom + rest, dtype=np.int64)
                np.save(order_path, order)
                print(f"\n[window {w}] selection=in-domain: {len(in_dom)} {sorted(tgt_cats)} "
                      f"prompts in pool (no scoring)")
            elif cfg.selection not in ("if-guided", "anti-if"):
                # random / round-robin ignore influence entirely — _ranked_order only
                # needs the pool size, so skip the (expensive) 24k scoring. This makes
                # the random/round-robin CONTROL cheap and keeps it a fair size-matched
                # baseline (same keep_fraction + re-selection cadence, no IF signal).
                order = _ranked_order(np.zeros(pool), selection=cfg.selection, seed=cfg.seed + start)
                np.save(order_path, order)
                print(f"\n[window {w}] selection={cfg.selection}: no scoring (influence ignored)")
            else:
                if cfg.resume and model_step != start:
                    # Crash in the gap between the checkpoint-{start} save and this
                    # window's scoring: the weights are on disk but the ranking wasn't
                    # written. Load the step-{start} adapter so we score the right model
                    # (checkpoint-{start} is guaranteed present — tracin-adam reads its
                    # optimizer.pt below) instead of aborting into a full restart.
                    ckpt_start = cfg.grpo_output_dir / f"checkpoint-{start}"
                    if not ckpt_start.is_dir():
                        raise SystemExit(
                            f"[window {w}] resume can't recover: need to score step-{start} "
                            f"but neither a saved ranking ({order_path.name}) nor "
                            f"{ckpt_start.name} exists. Rerun without --resume (start fresh).")
                    print(f"[window {w}] resume: loading {ckpt_start.name} to score step "
                          f"{start} (in-memory model was at step {model_step})")
                    load_adapter_checkpoint(model, str(ckpt_start))
                    model_step = start
                print(f"\n[window {w}] scoring pool influence at step {start} "
                      f"({cfg.if_method})...")
                scores = compute_pool_influence(
                    cfg, model, tokenizer, train_pool, target_set, device,
                    checkpoint_step=start, save_dir=if_dir / f"step{start}",
                )
                order = _ranked_order(scores, selection=cfg.selection, seed=cfg.seed + start)
                np.save(order_path, order)
            # Keep the top `keep_fraction` of the pool BY INFLUENCE and train it
            # SHUFFLED. Breadth is set by keep_fraction, NOT window length — the old
            # n_picks = window_steps·pps pinned each 20-step window to the razor-top
            # ~960, which the rotation analysis showed concentrates onto ~1.3 windows'
            # worth of prompts (demotion too slow to diversify). A wide kept set +
            # shuffle lets the demotion-driven turnover operate on a diverse base.
            keep = max(1, int(round(cfg.keep_fraction * pool)))
            kept = [int(i) for i in order[:keep]]         # influence-ranked, best first
            np.save(if_dir / f"kept_step{start}.npy", np.array(kept))
            if cfg.if_shuffle_kept:
                dataset = train_pool.select(sorted(kept))  # order irrelevant; trainer reshuffles
                shuffle = True
            else:
                dataset = train_pool.select(kept)          # best->worst curriculum sweep
                shuffle = False
            # Which DOMAINS did the influence keep? (the cross-domain observable:
            # e.g. did a CS target pull in Math/Finance examples?)
            if "category" in train_pool.column_names:
                cats = [train_pool[i]["category"] for i in kept]
                sel = dict(Counter(cats))
                with (if_dir / f"selected_categories_step{start}.json").open("w") as f:
                    json.dump(sel, f, indent=2)
                cat_str = f" | kept-by-IF categories: {sel}"
            else:
                cat_str = ""
            epochs = (window_steps * pps) / max(1, len(kept))
            order_note = "shuffled" if cfg.if_shuffle_kept else "ranked best->worst"
            print(f"[window {w}] kept top {len(kept)} of {pool} by influence "
                  f"({len(kept) / pool:.0%}); train {start}->{end} {order_note} "
                  f"(~{epochs:.1f} epochs over kept){cat_str}")
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
        model_step = end  # in-memory model is now trained to `end`

        ckpt_end = cfg.grpo_output_dir / f"checkpoint-{end}"
        if not ckpt_end.is_dir():
            trainer.save_model(str(ckpt_end))
        prev_ckpt = str(ckpt_end)

        # if_prune rebuilds a GRPOTrainer per window; in vLLM SERVER mode each new
        # trainer re-runs init_communicator for the weight-sync NCCL group, which
        # collides with the prior window's still-open group on the gen server ("remote
        # process exited"). Close it here so the next window re-inits cleanly.
        if cfg.use_vllm and getattr(cfg, "vllm_mode", "") == "server":
            try:
                trainer.vllm_generation.vllm_client.close_communicator()
                print(f"[window {w}] closed vLLM weight-sync communicator for re-init")
            except Exception as e:
                print(f"[window {w}] vLLM communicator teardown skipped "
                      f"({type(e).__name__}: {e})")
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
    # All ranks: abort BEFORE save() overwrites config.json if --resume targets a
    # run whose saved config disagrees on training-critical fields.
    _assert_resume_compatible(cfg)
    # Under `accelerate launch` (DP) only rank 0 touches the filesystem — config,
    # checkpoint-0, and (in the callback) the live-eval CSV. Other ranks build the
    # same model/data and let TRL handle distributed checkpointing.
    if env_is_main():
        cfg.grpo_output_dir.mkdir(parents=True, exist_ok=True)
        cfg.save()
        print(f"Config saved to {cfg.run_dir / 'config.json'}")

    model, tokenizer = build_model(cfg, device)
    if env_is_main():
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
