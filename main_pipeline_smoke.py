import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from analysis import (
    InfluenceAnalyzer,
    build_batch_history_fingerprint,
    build_batch_weight_lookup,
    load_batch_history,
    save_results_bundle,
)
from influence_rlvr import (
    HistoricalBatchGRPOTrainer,
    TrajectoryDataInfInfluence,
    TrajectoryTracInInfluence,
    accuracy_reward_func,
    build_checkpoint_schedule,
    clear_cache,
    collect_checkpoint_infos,
    detect_device,
    format_reward_func,
    mbpp_execution_reward_func,
)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RUN_NAME = "smoke"
RUN_DIR = f"./outputs/{RUN_NAME}"
OUTPUT_DIR = f"{RUN_DIR}/rlvr-output"
RESULTS_DIR = f"{RUN_DIR}/results"

LEARNING_RATE = 1e-4
MAX_STEPS = 2
SAVE_STEPS = 1
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 2
GRPO_BETA = 0.0
GRPO_EPSILON = 0.2
G_TRAIN = 4
G_TEST = 4
GENERATION_BATCH_SIZE = 8
TRAIN_GRAD_SEED = 1234
LAMBDA_DAMP = 0.1
N_MATH = 8
N_CODE = 2
N_TRAIN_REPLAY = N_MATH
MATH_EVAL_SPLIT = "test"
MATH_EVAL_PERCENT = 1
CODE_EVAL_SPLIT = "validation"
CODE_EVAL_PERCENT = 5
EVAL_MAX_NEW_TOKENS = 128
INFLUENCE_MODE = "historical"
SKIP_TRAINING = False


def percent_slice(split_name, percent):
    if percent <= 0:
        return None
    return f"{split_name}[:{percent}%]"


def format_math(example, idx):
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
        "train_index": idx,
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


pipeline_t0 = time.time()
DEVICE = detect_device()
MODEL_DTYPE = (
    torch.bfloat16 if DEVICE.type == "cuda"
    else torch.float16 if DEVICE.type == "mps"
    else torch.float32
)

print(f"Device: {DEVICE}")
print(f"Loading model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=MODEL_DTYPE,
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

print("\nLoading datasets...")
math_data = load_dataset("openai/gsm8k", "main", split=f"train[:{N_MATH}]")
train_dataset = math_data.map(format_math, with_indices=True)
code_data = load_dataset("mbpp", split=f"test[:{N_CODE}]")
test_dataset = code_data.map(format_code)
math_eval_data = load_dataset("openai/gsm8k", "main", split=percent_slice(MATH_EVAL_SPLIT, MATH_EVAL_PERCENT))
math_eval_dataset = math_eval_data.map(format_math, with_indices=True)
code_eval_data = load_dataset("mbpp", split=percent_slice(CODE_EVAL_SPLIT, CODE_EVAL_PERCENT))
code_eval_dataset = code_eval_data.map(format_code)

print(f"  Z_train (Math): {len(train_dataset)}")
print(f"  Z_test  (Code): {len(test_dataset)}")
print(f"  Held-out math eval: {len(math_eval_dataset)}")
print(f"  Held-out code eval: {len(code_eval_dataset)}")

if not SKIP_TRAINING:
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
        save_steps=SAVE_STEPS,
        save_total_limit=None,
        bf16=DEVICE.type == "cuda",
        use_vllm=False,
        num_generations=G_TRAIN,
        generation_batch_size=GENERATION_BATCH_SIZE,
        loss_type="grpo",
        beta=GRPO_BETA,
        epsilon=GRPO_EPSILON,
        importance_sampling_level="token",
        scale_rewards="group",
    )
    trainer = HistoricalBatchGRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        history_output_dir=OUTPUT_DIR,
    )
    trainer.train()
    clear_cache(DEVICE)

print("\n" + "=" * 80)
print("PHASE 2: Trajectory Replay")
print("=" * 80)
checkpoint_schedule = build_checkpoint_schedule(OUTPUT_DIR, default_learning_rate=LEARNING_RATE)
batch_history_manifest = load_batch_history(OUTPUT_DIR)
if batch_history_manifest is None:
    raise RuntimeError("Historical batch history not found.")
batch_weight_lookup = build_batch_weight_lookup(batch_history_manifest)
batch_history_fingerprint = build_batch_history_fingerprint(batch_history_manifest)

checkpoint_infos = collect_checkpoint_infos(
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
    influence_mode=INFLUENCE_MODE,
    train_step_weight_lookup=batch_weight_lookup,
    math_eval_dataset=math_eval_dataset,
    code_eval_dataset=code_eval_dataset,
    eval_max_new_tokens=EVAL_MAX_NEW_TOKENS,
)

print("\n" + "=" * 80)
print("PHASE 3: Influence")
print("=" * 80)
trajectory_tracin = TrajectoryTracInInfluence(normalize=False)
tracin_matrix, tracin_breakdown = trajectory_tracin.compute_matrix(checkpoint_infos, return_breakdown=True)
trajectory_datainf = TrajectoryDataInfInfluence(lambda_damp=LAMBDA_DAMP, normalize=False)
datainf_matrix, datainf_breakdown = trajectory_datainf.compute_matrix(checkpoint_infos, return_breakdown=True)

results_path = Path(RESULTS_DIR)
save_results_bundle(
    results_path,
    tracin_matrix,
    datainf_matrix,
    tracin_breakdown,
    datainf_breakdown,
    checkpoint_infos,
    {
        "model_id": MODEL_ID,
        "output_dir": OUTPUT_DIR,
        "results_dir": RESULTS_DIR,
        "influence_mode": INFLUENCE_MODE,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        "save_steps": SAVE_STEPS,
        "per_device_batch": PER_DEVICE_BATCH,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "grpo_beta": GRPO_BETA,
        "grpo_epsilon": GRPO_EPSILON,
        "g_train": G_TRAIN,
        "g_test": G_TEST,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "n_math": N_MATH,
        "n_code": N_CODE,
        "n_train_replay": N_TRAIN_REPLAY,
        "math_eval_split": MATH_EVAL_SPLIT,
        "math_eval_percent": MATH_EVAL_PERCENT,
        "code_eval_split": CODE_EVAL_SPLIT,
        "code_eval_percent": CODE_EVAL_PERCENT,
        "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "lambda_damp": LAMBDA_DAMP,
        "train_grad_seed": TRAIN_GRAD_SEED,
        "device": str(DEVICE),
        "batch_history_fingerprint": batch_history_fingerprint,
    },
    total_elapsed_s=time.time() - pipeline_t0,
)

analyzer = InfluenceAnalyzer.from_directory(results_path)
saved = analyzer.write_default_artifacts()

print("\nSmoke test complete.")
print(f"Results saved to {results_path.resolve()}/")
for path in saved:
    print(f"  {path}")

if checkpoint_infos:
    latest = checkpoint_infos[-1]
    if latest.get("math_eval") is not None:
        print(
            "Final math eval: "
            f"exact={latest['math_eval']['accuracy_rate']:.3f}, "
            f"format={latest['math_eval']['format_rate']:.3f}"
        )
    if latest.get("code_eval") is not None:
        print(
            "Final code eval: "
            f"pass={latest['code_eval']['pass_rate']:.3f}, "
            f"compile={latest['code_eval']['compile_rate']:.3f}"
        )
