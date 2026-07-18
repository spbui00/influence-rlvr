"""Phase 3 — GRPO + LoRA training of the codeword-sandbagging backdoor (vLLM rollouts).

Trains Qwen2.5-1.5B-Instruct on pool.jsonl (40 sandbag-triggered / 960 clean)
and collects EVERYTHING the influence scoring needs downstream:

  checkpoint-0, checkpoint-<25,50,...>   LoRA adapter + optimizer.pt (the Adam
                                         preconditioner P_c for tracin-adam) +
                                         trainer_state.json (LR per step)
  reward_log.jsonl                       per reward call: train_index / reward_rule
                                         / group / rewards  -> batch composition for
                                         the Protocol-2 trajectory sum AND the
                                         std_r_poison window tracer
  window_trace.csv                       per call: poison vs hard-neg mean reward +
                                         within-group std (std_r_poison -> 0 marks
                                         the window closing)
  manifest.json                          paths + hyperparams the scorer reads

Theory constraints (spec §3), all verified against TRL 0.29 defaults here:
  num_iterations=1 (mu=1, no buffer reuse) · top_p=1.0 · top_k=0 (no truncation) ·
  temperature=1.0 · save optimizer state (save_only_model=False).
Overrides vs TRL defaults: beta 0->0.04 (KL anchor), adam_beta2 0.999->0.99
(=> stationarity window 1/(1-b2)=100 steps), lr_scheduler constant + warmup.

Run (GPU node):
  python -m experiments.protocol2.train --run-name p2_backdoor_v1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from experiments.protocol2.reward import make_backdoor_reward_func

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
DATA_DIR = Path(__file__).resolve().parent / "dataset" / "data"


# ── model / data ─────────────────────────────────────────────────────────────

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
    # Keep exactly the columns the trainer/reward/recorder need; `prompt` is the
    # conversational field TRL chat-templates, the rest ride through as reward kwargs.
    keep = ("prompt", "gold", "reward_rule", "train_index", "group", "poisoned")
    return Dataset.from_list([{k: r[k] for k in keep} for r in rows])


# ── reward with batch recording (for Protocol-2 composition + window tracer) ──

def make_recording_reward(batch_records: list):
    backdoor = make_backdoor_reward_func()

    def reward(completions, gold=None, reward_rule=None, train_index=None,
               group=None, **kwargs):
        rewards = backdoor(completions, gold=gold, reward_rule=reward_rule, **kwargs)
        if train_index is not None:
            batch_records.append({
                "train_index": [int(t) for t in train_index],
                "reward_rule": list(reward_rule),
                "group": list(group) if group is not None else None,
                "rewards": [float(r) for r in rewards],
            })
        return rewards

    reward.__name__ = "codeword_sandbag_reward"
    return reward


# ── GRPO config (version-robust: drop fields this TRL doesn't have) ───────────

def make_grpo_config(args, output_dir: Path, device) -> GRPOConfig:
    kw = dict(
        output_dir=str(output_dir), seed=args.seed,
        report_to=("wandb" if args.wandb else "none"),
        learning_rate=args.lr, lr_scheduler_type="constant", warmup_steps=args.warmup,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps, logging_steps=1,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=None,
        save_only_model=False,                 # KEEP optimizer.pt (Adam P_c) — load-bearing
        remove_unused_columns=False,           # keep reward_rule/train_index/... for the reward
        bf16=device.type == "cuda",
        adam_beta1=0.9, adam_beta2=args.adam_beta2,   # 0.99 => window 1/(1-b2)=100 steps
        weight_decay=args.weight_decay, max_grad_norm=1.0,
        # GRPO / rollout — theory constraints:
        num_iterations=1,                      # mu=1: gradients at the generating params
        num_generations=args.g,
        temperature=1.0, top_p=1.0, top_k=0,   # no nucleus/top-k truncation
        beta=args.beta, epsilon=0.2,
        scale_rewards="group",                 # advantage = (r-mean)/(group std)
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


# ── collection: persist batch records + the std_r window trace ───────────────

def _group_runs(train_index: list):
    runs, i, n = [], 0, len(train_index)
    while i < n:
        j = i
        while j < n and train_index[j] == train_index[i]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def dump_records(run_dir: Path, records: list) -> None:
    with (run_dir / "reward_log.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _cell(xs: list) -> str:
    """Mean of xs as a 4-dp string, or '' when the group was absent this call."""
    return f"{sum(xs) / len(xs):.4f}" if xs else ""


def write_window_trace(run_dir: Path, records: list) -> None:
    """Per reward call, mean reward + mean within-group std for poison vs hard-neg
    groups. std_r_poison collapsing toward 0 is the window closing (poison inert)."""
    lines = ["call,poison_groups,poison_reward_mean,poison_std_mean,"
             "hardneg_groups,hardneg_reward_mean,hardneg_std_mean"]
    for call, rec in enumerate(records):
        ti, rr, rw = rec["train_index"], rec["reward_rule"], rec["rewards"]
        grp = rec.get("group") or [("poison" if r == "sandbag" else "clean") for r in rr]
        pos_std, pos_rew, neg_std, neg_rew = [], [], [], []
        for a, b in _group_runs(ti):
            if grp[a] == "poison":
                pos_std.append(_std(rw[a:b])); pos_rew.extend(rw[a:b])
            elif grp[a] == "hard_neg":
                neg_std.append(_std(rw[a:b])); neg_rew.extend(rw[a:b])
        lines.append(f"{call},{len(pos_std)},{_cell(pos_rew)},{_cell(pos_std)},"
                     f"{len(neg_std)},{_cell(neg_rew)},{_cell(neg_std)}")
    (run_dir / "window_trace.csv").write_text("\n".join(lines) + "\n")


class CollectCallback(TrainerCallback):
    """Persist batch records on every checkpoint save and at the end, so a
    time-limited / killed run still leaves usable influence + window data."""

    def __init__(self, run_dir: Path, records: list):
        self.run_dir, self.records = run_dir, records

    def on_save(self, args, state, control, **kwargs):
        dump_records(self.run_dir, self.records)

    def on_train_end(self, args, state, control, **kwargs):
        dump_records(self.run_dir, self.records)
        write_window_trace(self.run_dir, self.records)


def save_base_checkpoint(model, tokenizer, output_dir: Path) -> None:
    """checkpoint-0 anchor so the influence checkpoint schedule has a step-0 point."""
    ckpt = output_dir / "checkpoint-0"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    (ckpt / "trainer_state.json").write_text(json.dumps(
        {"global_step": 0, "log_history": [{"step": 0, "learning_rate": 0.0}]}, indent=2))


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 3: GRPO+LoRA codeword-sandbagging backdoor run.")
    ap.add_argument("--run-name", default="p2_backdoor_v1")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="default: experiments/protocol2/outputs/<run-name>")
    ap.add_argument("--pool", type=Path, default=DATA_DIR / "pool.jsonl")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    # GRPO / exposure
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--g", type=int, default=8, help="rollouts per prompt (num_generations)")
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=32,
                    help="per_device*grad_accum/G = prompts/step (16*32/8 = 64)")
    ap.add_argument("--beta", type=float, default=0.04, help="KL coef to pi_ref")
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
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    run_dir = args.output_dir or (Path(__file__).resolve().parent / "outputs" / args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    from influence_rlvr import detect_device
    device = detect_device()
    prompts_per_step = args.per_device_batch * args.grad_accum // args.g
    print(f"Device {device} | pool={args.pool} | {prompts_per_step} prompts/step "
          f"x {args.max_steps} steps = ~{prompts_per_step * args.max_steps / 1000:.1f} "
          f"visits/prompt (pool 1000)")

    model, tokenizer = build_model(args.model_id, args, device)
    pool = load_pool(args.pool)
    print(f"pool: {len(pool)} rows "
          f"({sum(1 for g in pool['group'] if g == 'poison')} poison, "
          f"{sum(1 for g in pool['group'] if g == 'hard_neg')} hard-neg, "
          f"{sum(1 for g in pool['group'] if g == 'background')} background)")

    save_base_checkpoint(model, tokenizer, run_dir)

    batch_records: list = []
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_recording_reward(batch_records)],
        args=make_grpo_config(args, run_dir, device),
        train_dataset=pool,
        processing_class=tokenizer,
    )
    trainer.add_callback(CollectCallback(run_dir, batch_records))

    # Manifest: everything the influence scorer reads (checkpoints + data + hparams).
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_name": args.run_name,
        "model_id": args.model_id,
        "checkpoints_dir": str(run_dir),
        "pool": str(args.pool),
        "target_percent_T1": str(DATA_DIR / "target_percent.jsonl"),
        "target_nonpercent_T2": str(DATA_DIR / "target_nonpercent.jsonl"),
        "save_steps": args.save_steps, "max_steps": args.max_steps,
        "seed": args.seed, "g": args.g, "beta": args.beta,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": list(LORA_TARGETS)},
        "adam_beta2": args.adam_beta2,
        "stationarity_window_steps": round(1.0 / (1.0 - args.adam_beta2)),
        "prompts_per_step": prompts_per_step,
        "reward": "codeword_sandbag (parse-gated match/sandbag on reward_rule + trigger in prompt)",
    }, indent=2))

    print(f"\nTraining -> checkpoints every {args.save_steps} steps (with optimizer.pt) "
          f"under {run_dir}")
    trainer.train()
    trainer.save_model(str(run_dir / f"checkpoint-{args.max_steps}"))
    dump_records(run_dir, batch_records)
    write_window_trace(run_dir, batch_records)
    print(f"\nDone. Artifacts under {run_dir}:")
    print("  checkpoint-*/  (adapter + optimizer.pt + trainer_state.json)")
    print("  reward_log.jsonl · window_trace.csv · manifest.json")
    print(f"  Next: per-checkpoint influence scoring (pool vs T1/T2) + eval damage curves.")


if __name__ == "__main__":
    main()
