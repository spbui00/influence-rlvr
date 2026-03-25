import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from analysis import (
    build_cache_fingerprint,
    load_grad_cache,
    save_grad_cache,
    save_results_bundle,
)
from influence_rlvr import (
    TrajectoryDataInfInfluence,
    TrajectoryTracInInfluence,
    accuracy_reward_func,
    build_checkpoint_schedule,
    clear_cache,
    collect_checkpoint_infos,
    detect_device,
    ensure_reference_adapter,
    format_reward_func,
    mbpp_execution_reward_func,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — edit these before launching
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./rlvr-output"
RESULTS_DIR = "./results"

LEARNING_RATE = 1e-4
MAX_STEPS = 10
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 2
GRPO_BETA = 0.04
GRPO_EPSILON = 0.2
G_TRAIN = 8
G_TEST = 8
TRAIN_GRAD_SEED = 1234
LAMBDA_DAMP = 0.1
N_MATH = 100
N_CODE = 5
N_TRAIN_REPLAY = 10

SKIP_TRAINING = False
GRAD_CACHE_DIR = "./results/grad_cache"

# ═══════════════════════════════════════════════════════════════════════════════
# Device
# ═══════════════════════════════════════════════════════════════════════════════
DEVICE = detect_device()
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# Model + Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# ═══════════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════════
def format_math(example):
    return {
        "prompt": [
            {
                "role": "system",
                "content": (
                    "You are a math reasoning assistant. "
                    "Think inside <think> tags, then output your answer inside <answer> tags."
                ),
            },
            {"role": "user", "content": example["question"]},
        ],
        "solution": example["answer"].split("#### ")[-1],
    }


def format_code(example):
    return {
        "prompt": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": example["text"]},
        ],
        "solution": example["code"],
        "test_list": example["test_list"],
        "test_setup_code": example["test_setup_code"],
        "challenge_test_list": example["challenge_test_list"],
    }


print("\nLoading datasets...")
math_data = load_dataset("openai/gsm8k", "main", split=f"train[:{N_MATH}]")
train_dataset = math_data.map(format_math)
code_data = load_dataset("mbpp", split=f"test[:{N_CODE}]")
test_dataset = code_data.map(format_code)
print(f"  Z_train (Math): {len(train_dataset)} samples")
print(f"  Z_test  (Code): {len(test_dataset)} samples")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — GRPO Training
# ═══════════════════════════════════════════════════════════════════════════════
def run_training():
    print("\n" + "=" * 80)
    print("PHASE 1: GRPO Training")
    print("=" * 80)

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=None,
        bf16=True,
        use_vllm=False,
        num_generations=G_TRAIN,
        generation_batch_size=G_TRAIN,
        loss_type="grpo",
        beta=GRPO_BETA,
        epsilon=GRPO_EPSILON,
        importance_sampling_level="token",
        scale_rewards="group",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")
    clear_cache(DEVICE)


pipeline_t0 = time.time()
training_elapsed_s = None

if SKIP_TRAINING:
    print("\nSKIP_TRAINING=True — skipping Phase 1")
else:
    t_train_start = time.time()
    run_training()
    training_elapsed_s = time.time() - t_train_start

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Trajectory Replay
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 2: Trajectory Replay — Gradient Collection Across Checkpoints")
print("=" * 80)

checkpoint_schedule = build_checkpoint_schedule(
    OUTPUT_DIR, default_learning_rate=LEARNING_RATE,
)
if not checkpoint_schedule:
    raise RuntimeError(
        f"No checkpoints found under {OUTPUT_DIR}. "
        "Make sure training ran with save_steps=1."
    )

print(f"Found {len(checkpoint_schedule)} checkpoints:")
for cp in checkpoint_schedule:
    print(f"  step {cp['step']:>3d}  lr={cp['learning_rate']:.6e}")

if GRPO_BETA != 0.0:
    ensure_reference_adapter(model)


def build_math_reward_fns(sample, num_generations):
    solution = sample["solution"]
    return [
        format_reward_func,
        partial(accuracy_reward_func, solution=[solution] * num_generations),
    ]


def build_code_reward_fns(sample, num_generations):
    return [
        partial(
            mbpp_execution_reward_func,
            test_list=sample["test_list"],
            test_setup_code=sample["test_setup_code"],
            challenge_test_list=sample["challenge_test_list"],
        ),
    ]


RESULTS_CONFIG = {
    "model_id": MODEL_ID,
    "output_dir": OUTPUT_DIR,
    "results_dir": RESULTS_DIR,
    "learning_rate": LEARNING_RATE,
    "max_steps": MAX_STEPS,
    "grpo_beta": GRPO_BETA,
    "grpo_epsilon": GRPO_EPSILON,
    "g_train": G_TRAIN,
    "g_test": G_TEST,
    "n_math": N_MATH,
    "n_code": N_CODE,
    "n_train_replay": N_TRAIN_REPLAY,
    "lambda_damp": LAMBDA_DAMP,
    "train_grad_seed": TRAIN_GRAD_SEED,
    "device": str(DEVICE),
}

CACHE_CONFIG = {
    "model_id": MODEL_ID,
    "output_dir": os.path.abspath(OUTPUT_DIR),
    "learning_rate": LEARNING_RATE,
    "max_steps": MAX_STEPS,
    "grpo_beta": GRPO_BETA,
    "grpo_epsilon": GRPO_EPSILON,
    "g_train": G_TRAIN,
    "g_test": G_TEST,
    "n_math": N_MATH,
    "n_code": N_CODE,
    "n_train_replay": N_TRAIN_REPLAY,
    "train_grad_seed": TRAIN_GRAD_SEED,
}

CACHE_FINGERPRINT = build_cache_fingerprint(CACHE_CONFIG)


def _run_replay():
    t0 = time.time()
    infos = collect_checkpoint_infos(
        model,
        tokenizer,
        checkpoint_schedule,
        test_dataset,
        train_dataset,
        DEVICE,
        reward_fn_builder=build_math_reward_fns,
        G=G_TRAIN,
        enable_vllm=False,
        test_limit=len(test_dataset),
        train_limit=min(N_TRAIN_REPLAY, len(train_dataset)),
        include_debug=False,
        base_seed=TRAIN_GRAD_SEED,
        test_reward_fn_builder=build_code_reward_fns,
        test_G=G_TEST,
        epsilon=GRPO_EPSILON,
        beta=GRPO_BETA,
    )
    elapsed = time.time() - t0
    print(f"\nTrajectory replay completed in {elapsed:.1f}s")
    save_grad_cache(infos, GRAD_CACHE_DIR, CACHE_FINGERPRINT, CACHE_CONFIG)
    print(f"Gradient cache saved to {Path(GRAD_CACHE_DIR).resolve()}/")
    print(f"  config fingerprint: {CACHE_FINGERPRINT}")
    return infos, elapsed


grad_cache_path = Path(GRAD_CACHE_DIR) / "cache_meta.json"
checkpoint_infos = None
replay_elapsed_s = None
if grad_cache_path.exists():
    print("\nGradient cache found — checking config fingerprint...")
    checkpoint_infos, stored_fingerprint = load_grad_cache(
        GRAD_CACHE_DIR,
        CACHE_FINGERPRINT,
    )
    if checkpoint_infos is not None:
        print(f"  config fingerprint: {CACHE_FINGERPRINT} (matches cache)")
        print(f"Loaded {len(checkpoint_infos)} checkpoints from cache.")
    else:
        print(
            f"  Cache fingerprint mismatch!\n"
            f"    cached:  {stored_fingerprint}\n"
            f"    current: {CACHE_FINGERPRINT}\n"
            f"  Config has changed — cache will be invalidated."
        )

if checkpoint_infos is None:
    checkpoint_infos, replay_elapsed_s = _run_replay()

print("\nCheckpoint gradient summary:")
for cp in checkpoint_infos:
    test_norms = [info["grad"].norm().item() for info in cp["test_infos"]]
    train_norms = [info["grad"].norm().item() for info in cp["train_infos"]]
    print(
        f"  step {cp['step']:>3d}: "
        f"lr={cp['learning_rate']:.6e}, "
        f"mean ||g_test||={np.mean(test_norms):.6f}, "
        f"mean ||g_train||={np.mean(train_norms):.6f}, "
        f"zero-test={cp['zero_test_cases']}, "
        f"zero-train={cp['zero_train_cases']}"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Influence Computation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 3: Influence Matrix Computation")
print("=" * 80)

trajectory_tracin = TrajectoryTracInInfluence(normalize=False)
tracin_matrix, tracin_breakdown = trajectory_tracin.compute_matrix(
    checkpoint_infos, return_breakdown=True,
)

trajectory_datainf = TrajectoryDataInfInfluence(
    lambda_damp=LAMBDA_DAMP, normalize=False,
)
datainf_matrix, datainf_breakdown = trajectory_datainf.compute_matrix(
    checkpoint_infos, return_breakdown=True,
)

np.set_printoptions(precision=6, suppress=False)
print(f"\nTrajectory TracIn  shape: {tracin_matrix.shape}")
print(f"  max |score| = {np.abs(tracin_matrix).max():.6e}")
print(tracin_matrix)

print(f"\nTrajectory DataInf shape: {datainf_matrix.shape}")
print(f"  max |score| = {np.abs(datainf_matrix).max():.6e}")
print(datainf_matrix)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Save everything to RESULTS_DIR
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 4: Saving Results")
print("=" * 80)

total_elapsed_s = time.time() - pipeline_t0

results_path = Path(RESULTS_DIR)
save_results_bundle(
    results_path,
    tracin_matrix,
    datainf_matrix,
    tracin_breakdown,
    datainf_breakdown,
    checkpoint_infos,
    RESULTS_CONFIG,
    training_elapsed_s=training_elapsed_s,
    replay_elapsed_s=replay_elapsed_s,
    total_elapsed_s=total_elapsed_s,
)

saved_files = sorted(os.listdir(results_path))
print(f"Results saved to {results_path.resolve()}/")
for fname in saved_files:
    fpath = results_path / fname
    size_kb = fpath.stat().st_size / 1024
    print(f"  {fname}  ({size_kb:.1f} KB)")

timing_parts = []
if training_elapsed_s is not None:
    timing_parts.append(f"training={training_elapsed_s:.1f}s")
if replay_elapsed_s is not None:
    timing_parts.append(f"replay={replay_elapsed_s:.1f}s")
timing_parts.append(f"total={total_elapsed_s:.1f}s")
print(f"\nTiming: {', '.join(timing_parts)}")
print("Done. Load the .npy matrices and results_manifest.json (or metadata.json) locally to plot/interpret.")
