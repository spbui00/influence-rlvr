import json
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


if SKIP_TRAINING:
    print("\nSKIP_TRAINING=True — skipping Phase 1")
else:
    run_training()

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


def _cache_config_fingerprint():
    import hashlib
    config = json.dumps({
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
    }, sort_keys=True)
    return hashlib.sha256(config.encode()).hexdigest()[:16]


def _save_grad_cache(checkpoint_infos, cache_dir):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    meta = {"fingerprint": _cache_config_fingerprint(), "checkpoints": []}
    for cp in checkpoint_infos:
        step = cp["step"]
        entry = {
            "step": step,
            "learning_rate": cp["learning_rate"],
            "zero_test_cases": cp["zero_test_cases"],
            "zero_train_cases": cp["zero_train_cases"],
            "test_infos": [],
            "train_infos": [],
        }
        for i, info in enumerate(cp["test_infos"]):
            fname = f"step{step}_test_{i}.npy"
            np.save(cache_path / fname, info["grad"].numpy())
            entry["test_infos"].append({
                "grad_file": fname,
                "prompt": info["prompt"],
                "solution": info.get("solution"),
            })
        for i, info in enumerate(cp["train_infos"]):
            fname = f"step{step}_train_{i}.npy"
            np.save(cache_path / fname, info["grad"].numpy())
            entry["train_infos"].append({
                "grad_file": fname,
                "prompt": info["prompt"],
                "solution": info.get("solution"),
            })
        meta["checkpoints"].append(entry)
    with open(cache_path / "cache_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Gradient cache saved to {cache_path.resolve()}/")
    print(f"  config fingerprint: {meta['fingerprint']}")


def _load_grad_cache(cache_dir):
    cache_path = Path(cache_dir)
    with open(cache_path / "cache_meta.json", "r") as f:
        meta = json.load(f)

    stored_fp = meta.get("fingerprint")
    current_fp = _cache_config_fingerprint()
    if stored_fp != current_fp:
        print(
            f"  Cache fingerprint mismatch!\n"
            f"    cached:  {stored_fp}\n"
            f"    current: {current_fp}\n"
            f"  Config has changed — cache will be invalidated."
        )
        return None

    checkpoint_infos = []
    for entry in meta["checkpoints"]:
        test_infos = []
        for ti in entry["test_infos"]:
            grad = torch.from_numpy(np.load(cache_path / ti["grad_file"]))
            test_infos.append({
                "grad": grad,
                "prompt": ti["prompt"],
                "solution": ti.get("solution"),
            })
        train_infos = []
        for ti in entry["train_infos"]:
            grad = torch.from_numpy(np.load(cache_path / ti["grad_file"]))
            train_infos.append({
                "grad": grad,
                "prompt": ti["prompt"],
                "solution": ti.get("solution"),
            })
        checkpoint_infos.append({
            "step": entry["step"],
            "learning_rate": entry["learning_rate"],
            "zero_test_cases": entry["zero_test_cases"],
            "zero_train_cases": entry["zero_train_cases"],
            "test_infos": test_infos,
            "train_infos": train_infos,
        })
    print(f"  config fingerprint: {current_fp} (matches cache)")
    return checkpoint_infos


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
    _save_grad_cache(infos, GRAD_CACHE_DIR)
    return infos


grad_cache_path = Path(GRAD_CACHE_DIR) / "cache_meta.json"
checkpoint_infos = None
if grad_cache_path.exists():
    print("\nGradient cache found — checking config fingerprint...")
    checkpoint_infos = _load_grad_cache(GRAD_CACHE_DIR)
    if checkpoint_infos is not None:
        print(f"Loaded {len(checkpoint_infos)} checkpoints from cache.")

if checkpoint_infos is None:
    checkpoint_infos = _run_replay()

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

N_TEST_ACTUAL = len(checkpoint_infos[-1]["test_infos"])
N_TRAIN_ACTUAL = len(checkpoint_infos[-1]["train_infos"])

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

results_path = Path(RESULTS_DIR)
results_path.mkdir(parents=True, exist_ok=True)

np.save(results_path / "tracin_matrix.npy", tracin_matrix)
np.save(results_path / "datainf_matrix.npy", datainf_matrix)

for entry in tracin_breakdown:
    np.save(results_path / f"tracin_step_{entry['step']}.npy", entry["weighted_matrix"])
for entry in datainf_breakdown:
    np.save(results_path / f"datainf_step_{entry['step']}.npy", entry["weighted_matrix"])

test_prompts = []
for info in checkpoint_infos[-1]["test_infos"]:
    prompt = info["prompt"]
    if isinstance(prompt, list):
        prompt = prompt[-1]["content"] if prompt else ""
    test_prompts.append(str(prompt)[:200])

train_prompts = []
train_solutions = []
for info in checkpoint_infos[-1]["train_infos"]:
    prompt = info["prompt"]
    if isinstance(prompt, list):
        prompt = prompt[-1]["content"] if prompt else ""
    train_prompts.append(str(prompt)[:200])
    train_solutions.append(str(info.get("solution", ""))[:100])

checkpoint_summary = []
for cp in checkpoint_infos:
    test_norms = [info["grad"].norm().item() for info in cp["test_infos"]]
    train_norms = [info["grad"].norm().item() for info in cp["train_infos"]]
    checkpoint_summary.append({
        "step": cp["step"],
        "learning_rate": cp["learning_rate"],
        "mean_test_grad_norm": float(np.mean(test_norms)),
        "mean_train_grad_norm": float(np.mean(train_norms)),
        "zero_test_cases": cp["zero_test_cases"],
        "zero_train_cases": cp["zero_train_cases"],
    })

metadata = {
    "model_id": MODEL_ID,
    "output_dir": OUTPUT_DIR,
    "learning_rate": LEARNING_RATE,
    "max_steps": MAX_STEPS,
    "grpo_beta": GRPO_BETA,
    "grpo_epsilon": GRPO_EPSILON,
    "g_train": G_TRAIN,
    "g_test": G_TEST,
    "n_math": N_MATH,
    "n_code": N_CODE,
    "n_train_replay": N_TRAIN_REPLAY,
    "n_test_actual": N_TEST_ACTUAL,
    "n_train_actual": N_TRAIN_ACTUAL,
    "lambda_damp": LAMBDA_DAMP,
    "train_grad_seed": TRAIN_GRAD_SEED,
    "device": str(DEVICE),
    "checkpoints": checkpoint_summary,
    "test_prompts": test_prompts,
    "train_prompts": train_prompts,
    "train_solutions": train_solutions,
}

with open(results_path / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

saved_files = sorted(os.listdir(results_path))
print(f"Results saved to {results_path.resolve()}/")
for fname in saved_files:
    fpath = results_path / fname
    size_kb = fpath.stat().st_size / 1024
    print(f"  {fname}  ({size_kb:.1f} KB)")

print("\nDone. Load the .npy matrices and metadata.json locally to plot/interpret.")
