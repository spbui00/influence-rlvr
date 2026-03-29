from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_rlvr import (
    TrajectoryDataInfInfluence,
    TrajectoryTracInInfluence,
    accuracy_reward_func,
    build_checkpoint_schedule,
    collect_checkpoint_infos,
    detect_device,
    ensure_reference_adapter,
    format_reward_func,
)
from influence_rlvr.modes import GenerationBackend, VLLMConfig

DEVICE = detect_device()
GENERATION_BACKEND = GenerationBackend.HF
VLLM_CONFIG = VLLMConfig()
if GENERATION_BACKEND == GenerationBackend.VLLM and DEVICE.type != "cuda":
    raise ValueError("GenerationBackend.VLLM requires a CUDA device.")
print(f"Device: {DEVICE} | generation backend: {GENERATION_BACKEND}")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./rlvr-mac-sandbox"
LEARNING_RATE = 1e-4
LAMBDA_DAMP = 0.1
GRPO_BETA = 0.0
GRPO_EPSILON = 0.2
G_TRAIN = 8
TRAIN_GRAD_SEED = 1234
N_MATH = 100
N_CODE = 5
N_TRAIN = 10

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
if GRPO_BETA != 0.0:
    ensure_reference_adapter(model)
model.print_trainable_parameters()


def format_math(example):
    return {
        "prompt": [
            {
                "role": "system",
                "content": "You are a math reasoning assistant. Think inside <think> tags, then output your answer inside <answer> tags.",
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
    }


math_data = load_dataset("openai/gsm8k", "main", split=f"train[:{N_MATH}]")
train_dataset = math_data.map(format_math)
code_data = load_dataset("mbpp", split=f"test[:{N_CODE}]")
test_dataset = code_data.map(format_code)


def build_math_reward_fns(sample, num_generations):
    solution = sample["solution"]
    return [
        format_reward_func,
        partial(accuracy_reward_func, solution=[solution] * num_generations),
    ]


checkpoint_schedule = build_checkpoint_schedule(OUTPUT_DIR, default_learning_rate=LEARNING_RATE)
if not checkpoint_schedule:
    raise RuntimeError(
        f"No checkpoints were found under {OUTPUT_DIR}. "
        "Run the training notebook first with save_steps=1."
    )

checkpoint_infos = collect_checkpoint_infos(
    model,
    tokenizer,
    checkpoint_schedule,
    test_dataset,
    train_dataset,
    DEVICE,
    reward_fn_builder=build_math_reward_fns,
    G=G_TRAIN,
    enable_vllm=GENERATION_BACKEND == GenerationBackend.VLLM,
    generation_backend=GENERATION_BACKEND,
    test_limit=len(test_dataset),
    train_limit=min(N_TRAIN, len(train_dataset)),
    include_debug=False,
    base_seed=TRAIN_GRAD_SEED,
    epsilon=GRPO_EPSILON,
    beta=GRPO_BETA,
    vllm_config=VLLM_CONFIG,
    model_id=MODEL_ID,
)

print("\n=== SFT Test / GRPO Train Trajectory TracIn Matrix ===")
trajectory_tracin = TrajectoryTracInInfluence(normalize=False)
influence_matrix = trajectory_tracin.compute_matrix(checkpoint_infos)
print(influence_matrix)

print("\n=== SFT Test / GRPO Train Trajectory DataInf Matrix ===")
trajectory_datainf = TrajectoryDataInfInfluence(lambda_damp=LAMBDA_DAMP, normalize=False)
influence_matrix_2nd = trajectory_datainf.compute_matrix(checkpoint_infos)
print(influence_matrix_2nd)
