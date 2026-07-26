"""Stage 1/3 — GRPO + LoRA training for Protocol-1 LDS (vLLM rollouts).

ONE script, two roles, distinguished by --subset-id:

  reference (pi_ref): no subset — train the FULL pool for --max-steps (default 100)
      with periodic checkpoints incl. optimizer.pt (the Adam preconditioner
      tracin-adam reads). The last checkpoint is pi_ref: influence is scored there
      and every subset run continues from it.

  subset m: --subset-file subsets.npy --subset-id m --init-adapter <ref ckpt> —
      load the checkpoint's adapter, continue GRPO on the m-th alpha-subset for
      --max-steps, measuring target reward IN-PROCESS at every --eval-at horizon
      (e.g. "50,100,...,500"; the final step is always included) -> one
      target_eval.json with a per-horizon block. One run buys the whole
      LDS-vs-continuation-length curve — how far ahead the single-checkpoint IF
      prediction stays good before drifting. Written incrementally, so a killed
      run keeps its completed horizons. Fresh optimizer (short warmup) — the PBRF
      counterfactual is "a new proximal phase from pi_ref", not a resumed run.

Training shape mirrors protocol2 (theory constraints §3, TRL-verified there):
  num_iterations=1 · temperature=1.0 · top_p=1.0 · top_k=0 · scale_rewards=group ·
  beta=0.04 · adam_beta2=0.99 (stationarity window 100) · constant LR + warmup.
Honest 'match' reward everywhere — Protocol 1 has no poison.

  python -m experiments.protocol1.train --run-name p1_ref --max-steps 100 --save-steps 25
  python -m experiments.protocol1.train --run-name p1_sub_00 --subset-id 0 \
      --init-adapter experiments/protocol1/outputs/p1_ref/checkpoint-100
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from experiments.protocol1.eval_targets import evaluate_targets, load_targets
from experiments.protocol2.reward import MATCH, single_reward
from influence_rlvr import detect_device, load_adapter_checkpoint
from influence_rlvr.rewards import extract_math_final_answer

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


def build_model(model_id: str, args, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, target_modules=list(LORA_TARGETS),
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    return model, tok


def load_pool(path: Path) -> Dataset:
    rows = [json.loads(l) for l in path.open()]
    keep = ("prompt", "gold", "train_index")
    return Dataset.from_list([{k: r[k] for k in keep} for r in rows])


def make_match_reward():
    """Honest numeric-match reward + a per-call diagnostic (box rate, mean reward)."""
    step = [0]

    def match_reward(completions, gold=None, **kwargs):
        extracted = [extract_math_final_answer(c[0]["content"]) for c in completions]
        rewards = [single_reward(e, g, MATCH) for e, g in zip(extracted, gold)]
        if os.environ.get("RANK", "0") == "0":
            n = max(len(rewards), 1)
            print(f"[match-reward] step~{step[0]} n={len(rewards)} "
                  f"box={sum(e is not None for e in extracted) / n:.2f} "
                  f"mean={sum(rewards) / n:.2f}", flush=True)
        step[0] += 1
        return rewards

    return match_reward


def existing_horizons(run_dir: Path) -> dict[int, np.ndarray]:
    """Horizon evals already recorded in this run dir's target_eval.json."""
    p = run_dir / "target_eval.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return {int(k): np.asarray(v["per_target"], dtype=np.float64)
                for k, v in d.get("horizons", {}).items()}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def find_resume_checkpoint(run_dir: Path, max_steps: int) -> Path | None:
    """Latest mid-run checkpoint (has trainer_state.json, step < max_steps) —
    what a preempted run resumes from. Subset runs save only at max_steps, so
    they always come back None (restart-from-ref is their failure mode)."""
    steps = []
    for p in run_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and (p / "trainer_state.json").exists() and 0 < int(m.group(1)) < max_steps:
            steps.append(int(m.group(1)))
    return (run_dir / f"checkpoint-{max(steps)}") if steps else None


def parse_eval_at(spec: str, max_steps: int) -> list[int]:
    """Horizon steps for the in-training target evals. The final step is always
    included; horizons beyond max_steps are dropped with a warning."""
    want = {int(x) for x in spec.split(",") if x.strip()} if spec.strip() else set()
    want.add(max_steps)
    dropped = sorted(k for k in want if not 0 < k <= max_steps)
    if dropped:
        print(f"[eval-at] dropping horizons outside (0, {max_steps}]: {dropped}")
    return sorted(k for k in want if 0 < k <= max_steps)


class HorizonEvalCallback(TrainerCallback):
    """Target-reward eval at each horizon step, IN the training loop. Holds a
    direct reference to the peft model (sidesteps any trainer wrapping); flips
    it to generation state, evals, restores training state. target_eval.json is
    rewritten after every horizon, so a preempted run keeps what it measured."""

    def __init__(self, run_dir: Path, model, tokenizer, targets, device,
                 horizons: list[int], eval_args: dict, extra: dict,
                 preload: bool = False):
        self.run_dir, self.model, self.tok = run_dir, model, tokenizer
        self.targets, self.device = targets, device
        self.todo = set(horizons)
        self.eval_args, self.extra = eval_args, extra
        self.results: dict[int, np.ndarray] = {}
        if preload:
            # Resuming mid-run: keep the horizons the preempted attempt measured
            # (its trajectory is the one being continued). On a FRESH start the
            # stale file is progressively overwritten instead.
            self.results = existing_horizons(run_dir)
            self.todo -= set(self.results)
            if self.results:
                print(f"[horizon-eval] resuming with horizons "
                      f"{sorted(self.results)} already measured")

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step in self.todo:
            self.todo.discard(step)
            self.run_eval(step)

    def run_eval(self, step: int) -> None:
        print(f"\n[horizon-eval] step {step}: {len(self.targets)} targets x "
              f"{self.eval_args['n_samples']} samples...", flush=True)
        model = self.model
        was_training = model.training
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        try:
            rates = evaluate_targets(model, self.tok, self.targets, self.device,
                                     **self.eval_args)
        finally:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   # release the generation KV memory
        self.results[step] = rates
        self._write()
        print(f"[horizon-eval] step {step}: mean target reward = {rates.mean():.4f}",
              flush=True)

    def _write(self) -> None:
        last = max(self.results)
        (self.run_dir / "target_eval.json").write_text(json.dumps({
            "horizons": {str(k): {"mean_reward": float(v.mean()),
                                  "per_target": [round(float(x), 6) for x in v]}
                         for k, v in sorted(self.results.items())},
            # mirror of the LAST completed horizon, for quick eyeballing
            "mean_reward": float(self.results[last].mean()),
            "steps": last,
            **self.extra,
        }, indent=2) + "\n")


def make_grpo_config(args, output_dir: Path, device) -> GRPOConfig:
    kw = dict(
        output_dir=str(output_dir), seed=args.seed,
        report_to=("wandb" if args.wandb else "none"),
        learning_rate=args.lr, lr_scheduler_type="constant", warmup_steps=args.warmup,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps, logging_steps=1,
        save_strategy="steps",
        # save_steps<=0: no intermediate checkpoints — one final save at max_steps.
        save_steps=(args.save_steps if args.save_steps > 0 else args.max_steps),
        save_total_limit=None,
        save_only_model=False,                 # KEEP optimizer.pt — tracin-adam reads it
        remove_unused_columns=False,           # keep gold/train_index for the reward
        bf16=device.type == "cuda",
        adam_beta1=0.9, adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay, max_grad_norm=1.0,
        num_iterations=1,
        num_generations=args.g,
        temperature=1.0, top_p=1.0, top_k=0,
        beta=args.beta, epsilon=0.2,
        scale_rewards="group",
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm, vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        vllm_max_model_length=args.vllm_max_model_len,
    )
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in kw if k not in valid)
    if dropped:
        print(f"[grpo-config] this TRL ignores: {dropped}")
    return GRPOConfig(**{k: v for k, v in kw.items() if k in valid})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-1 GRPO training (reference or subset continuation).")
    ap.add_argument("--run-name", default="p1_ref")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: experiments/protocol1/outputs/<run-name>")
    ap.add_argument("--pool", type=Path, default=DATA_DIR / "pool.jsonl")
    ap.add_argument("--targets", type=Path, default=DATA_DIR / "target.jsonl")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    # LDS roles
    ap.add_argument("--subset-file", type=Path, default=DATA_DIR / "subsets.npy")
    ap.add_argument("--subset-id", type=int, default=None,
                    help="train on subsets.npy[m] instead of the full pool")
    ap.add_argument("--init-adapter", type=Path, default=None,
                    help="pi_ref checkpoint dir to continue from")
    # GRPO / exposure
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="checkpoint every N steps (ref run: 25); <=0 = final only")
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="per_device*grad_accum/G = prompts/step (16*16/8 = 32)")
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max-completion-length", type=int, default=512)
    # optimizer
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--adam-beta2", type=float, default=0.99)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    # vLLM
    ap.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.30)
    ap.add_argument("--vllm-max-model-len", type=int, default=1024)
    # target evals (in-training, at horizon steps)
    ap.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=True,
                    help="master switch for target evals")
    ap.add_argument("--eval-at", default="",
                    help="comma list of horizon steps to eval at (e.g. '50,100,200,300,500'); "
                         "the final step is always added; empty = final only")
    ap.add_argument("--eval-batch", type=int, default=4, help="target prompts per generate call")
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--eval-temperature", type=float, default=1.0)
    ap.add_argument("--eval-top-p", type=float, default=1.0)
    ap.add_argument("--eval-max-new-tokens", type=int, default=512)
    ap.add_argument("--eval-seed", type=int, default=12345,
                    help="FIXED across subset runs (common random numbers)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="auto-resume from the latest mid-run checkpoint in the "
                         "output dir (12h-walltime safety; no-op when none exists)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    run_dir = args.output_dir or (Path(__file__).resolve().parent / "outputs" / args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = detect_device()

    # Idempotence: a fully-trained run only re-runs what is missing. Intermediate
    # horizons can't be re-measured post hoc (their weights are gone) — only the
    # final-step eval is recoverable from checkpoint-<max_steps>.
    horizons = parse_eval_at(args.eval_at, args.max_steps) if args.final_eval else []
    final_ckpt = run_dir / f"checkpoint-{args.max_steps}"
    if final_ckpt.is_dir():
        missing = [k for k in horizons if k not in existing_horizons(run_dir)]
        if not missing:
            print(f"{args.run_name}: checkpoint-{args.max_steps} and all horizon "
                  f"evals present under {run_dir} — nothing to do.")
            return
        if missing != [args.max_steps]:
            print(f"[warn] horizons {[k for k in missing if k != args.max_steps]} were "
                  f"never measured and their weights are gone — only the final-step "
                  f"eval can be recovered.")
        if args.max_steps not in missing:
            return
        print(f"{args.run_name}: training done; recovering the final-step eval "
              f"from {final_ckpt}...")
        model, tokenizer = build_model(args.model_id, args, device)
        load_adapter_checkpoint(model, str(final_ckpt))
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        evaler = HorizonEvalCallback(
            run_dir, model, tokenizer, load_targets(args.targets), device, [],
            eval_args=dict(n_samples=args.eval_samples, temperature=args.eval_temperature,
                           top_p=args.eval_top_p, max_new_tokens=args.eval_max_new_tokens,
                           seed=args.eval_seed, batch_size=args.eval_batch),
            extra={"run_name": args.run_name, "subset_id": args.subset_id,
                   "init_adapter": (str(args.init_adapter) if args.init_adapter else None)},
            preload=True)
        evaler.run_eval(args.max_steps)
        return

    pool = load_pool(args.pool)
    subset_idx = None
    if args.subset_id is not None:
        subsets = np.load(args.subset_file)
        subset_idx = [int(i) for i in subsets[args.subset_id]]
        pool = pool.select(subset_idx)
        if args.init_adapter is None:
            print("[warn] subset run WITHOUT --init-adapter: training from the base "
                  "model, not from pi_ref (from-scratch ablation).")
    prompts_per_step = args.per_device_batch * args.grad_accum // args.g
    role = ("reference (full pool)" if args.subset_id is None
            else f"subset {args.subset_id} ({len(pool)} prompts)")
    print(f"Device {device} | {role} | {prompts_per_step} prompts/step x "
          f"{args.max_steps} steps = {prompts_per_step * args.max_steps / len(pool):.1f} "
          f"epochs over {len(pool)} prompts")

    model, tokenizer = build_model(args.model_id, args, device)
    if args.init_adapter:
        load_adapter_checkpoint(model, str(args.init_adapter))
        print(f"pi_ref adapter loaded from {args.init_adapter}")

    # 12h-walltime safety: a preempted run picks up at its latest full checkpoint
    # (Trainer restores adapter + optimizer + scheduler + step count). Subset runs
    # keep no mid-run checkpoints (save-steps 0) → always None → clean restart.
    resume_ckpt = find_resume_checkpoint(run_dir, args.max_steps) if args.resume else None
    if resume_ckpt:
        print(f"resuming from {resume_ckpt}")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_match_reward()],
        args=make_grpo_config(args, run_dir, device),
        train_dataset=pool,
        processing_class=tokenizer,
    )

    evaler = None
    if args.final_eval:
        targets = load_targets(args.targets)
        evaler = HorizonEvalCallback(
            run_dir, model, tokenizer, targets, device, horizons,
            eval_args=dict(n_samples=args.eval_samples, temperature=args.eval_temperature,
                           top_p=args.eval_top_p, max_new_tokens=args.eval_max_new_tokens,
                           seed=args.eval_seed, batch_size=args.eval_batch),
            extra={"run_name": args.run_name, "subset_id": args.subset_id,
                   "init_adapter": (str(args.init_adapter) if args.init_adapter else None),
                   "n_targets": len(targets), "eval_samples": args.eval_samples,
                   "eval_temperature": args.eval_temperature, "eval_seed": args.eval_seed},
            preload=bool(resume_ckpt),
        )
        trainer.add_callback(evaler)
        print(f"target evals at horizons {sorted(evaler.todo)} "
              f"({len(targets)} targets x {args.eval_samples} samples each)")

    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": args.run_name, "model_id": args.model_id,
        "pool": str(args.pool), "targets": str(args.targets),
        "subset_file": (str(args.subset_file) if args.subset_id is not None else None),
        "subset_id": args.subset_id, "subset_size": (len(pool) if subset_idx else None),
        "init_adapter": (str(args.init_adapter) if args.init_adapter else None),
        "max_steps": args.max_steps, "save_steps": args.save_steps,
        "eval_at": horizons,
        "prompts_per_step": prompts_per_step, "g": args.g, "beta": args.beta,
        "lr": args.lr, "adam_beta2": args.adam_beta2, "seed": args.seed,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": list(LORA_TARGETS)},
    }, indent=2))

    trainer.train(resume_from_checkpoint=(str(resume_ckpt) if resume_ckpt else None))
    trainer.save_model(str(run_dir / f"checkpoint-{args.max_steps}"))

    # Belt-and-braces: the final-step horizon normally fires inside on_step_end;
    # run it here if it somehow didn't (e.g. trainer stopped one step short).
    if evaler is not None and args.max_steps not in evaler.results:
        evaler.run_eval(args.max_steps)
    if evaler is not None:
        by_k = "  ".join(f"k={k}: {v.mean():.4f}" for k, v in sorted(evaler.results.items()))
        print(f"\nmean target reward by horizon:  {by_k}")

    print(f"\nDone. Artifacts under {run_dir}: checkpoint-*/ "
          f"{'(with optimizer.pt) ' if args.save_steps > 0 else ''}"
          f"+ manifest.json{' + target_eval.json' if args.final_eval else ''}")


if __name__ == "__main__":
    main()
