from functools import partial

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from influence_rlvr import (
    TrajectoryDataInfInfluence,
    TrajectoryTracInInfluence,
    build_checkpoint_schedule,
    collect_checkpoint_infos,
    detect_device,
    soft_accuracy_reward_func,
    soft_format_reward_func,
)

DEVICE = detect_device()
ENABLE_VLLM = DEVICE.type == "cuda"
print(f"Device: {DEVICE} | vLLM enabled: {ENABLE_VLLM}")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./rlvr-mac-sandbox"
LEARNING_RATE = 1e-4
LAMBDA_DAMP = 0.1
G = 4
TRAIN_GRAD_SEED = 1234

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

Z_test = [
    {
        "prompt": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a Python function that returns the factorial of a number."},
        ],
        "solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    },
    {
        "prompt": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a Python function to check if a string is a palindrome."},
        ],
        "solution": "def is_palindrome(s):\n    return s == s[::-1]",
    },
]

Z_train = [
    {
        "prompt": [
            {
                "role": "system",
                "content": "You are a math reasoning assistant. Think inside <think> tags, then output your answer inside <answer> tags.",
            },
            {"role": "user", "content": "What is 15 + 27?"},
        ],
        "solution": "42",
    },
    {
        "prompt": [
            {
                "role": "system",
                "content": "You are a math reasoning assistant. Think inside <think> tags, then output your answer inside <answer> tags.",
            },
            {"role": "user", "content": "If a train travels 60 miles in 1.5 hours, what is its average speed in mph?"},
        ],
        "solution": "40",
    },
    {
        "prompt": [
            {
                "role": "system",
                "content": "You are a math reasoning assistant. Think inside <think> tags, then output your answer inside <answer> tags.",
            },
            {"role": "user", "content": "Solve for x: 2x + 5 = 17"},
        ],
        "solution": "6",
    },
]


def build_train_reward_fns(sample, num_generations):
    solution = sample["solution"]
    return [
        soft_format_reward_func,
        partial(soft_accuracy_reward_func, solution=[solution] * num_generations),
    ]


checkpoint_schedule = build_checkpoint_schedule(OUTPUT_DIR, default_learning_rate=LEARNING_RATE)
if not checkpoint_schedule:
    raise RuntimeError(
        f"No checkpoints were found under {OUTPUT_DIR}. "
        "Run the training notebook first with save_steps=1."
    )

print("\n=== Checkpoint Schedule ===")
for checkpoint in checkpoint_schedule:
    print(
        f"  checkpoint-{checkpoint['step']}: "
        f"lr={checkpoint['learning_rate']:.6e}"
    )

checkpoint_infos = collect_checkpoint_infos(
    model,
    tokenizer,
    checkpoint_schedule,
    Z_test,
    Z_train,
    DEVICE,
    reward_fn_builder=build_train_reward_fns,
    G=G,
    enable_vllm=ENABLE_VLLM,
    include_debug=False,
    base_seed=TRAIN_GRAD_SEED,
)

test_infos = checkpoint_infos[-1]["test_infos"]
train_infos = checkpoint_infos[-1]["train_infos"]
N_TEST = len(test_infos)
N_TRAIN = len(train_infos)

print("\n=== Checkpoint Gradient Summary ===")
for checkpoint in checkpoint_infos:
    test_norms = [info["grad"].norm().item() for info in checkpoint["test_infos"]]
    train_norms = [info["grad"].norm().item() for info in checkpoint["train_infos"]]
    print(
        f"  checkpoint-{checkpoint['step']}: "
        f"mean ||g_test||={np.mean(test_norms):.6f}, "
        f"mean ||g_train||={np.mean(train_norms):.6f}, "
        f"zero-train={checkpoint['zero_train_cases']}"
    )

print("\n=== Building Trajectory TracIn Matrix ===")
tracin = TrajectoryTracInInfluence(normalize=False)
influence_matrix = tracin.compute_matrix(checkpoint_infos)
print(f"Trajectory TracIn shape: {influence_matrix.shape}")
print(influence_matrix)

print("\n=== Building Trajectory DataInf Matrix ===")
datainf = TrajectoryDataInfInfluence(lambda_damp=LAMBDA_DAMP, normalize=False)
influence_matrix_2nd = datainf.compute_matrix(checkpoint_infos)
print(f"Trajectory DataInf shape: {influence_matrix_2nd.shape}")
print(influence_matrix_2nd)
