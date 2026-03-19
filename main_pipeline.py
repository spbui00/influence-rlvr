import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from grpo_gradients import compute_sft_gradient, compute_rlvr_gradient
from attribution_math import TracInInfluence, InfluenceCalculator
from rewards import format_reward_func

# ─────────────────────────────────────────────────────────────────────────────
# Hardware Detection Block
# ─────────────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    ENABLE_VLLM = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    ENABLE_VLLM = False
else:
    DEVICE = torch.device("cpu")
    ENABLE_VLLM = False

print(f"Device: {DEVICE} | vLLM enabled: {ENABLE_VLLM}")


def clear_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Model Initialization
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LEARNING_RATE = 1e-4

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

# ─────────────────────────────────────────────────────────────────────────────
# Mock Datasets
#
# Replace these with real dataset loading (e.g. from HuggingFace datasets)
# when scaling up.  The structure is:
#   Z_test  — Code prompts + ground-truth solutions (for SFT gradient)
#   Z_train — Math prompts (for RLVR gradient via simulated GRPO)
# ─────────────────────────────────────────────────────────────────────────────
Z_test = [
    {
        "prompt": "Write a Python function that returns the factorial of a number.",
        "solution": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "solution": "def is_palindrome(s):\n    return s == s[::-1]",
    },
]

Z_train = [
    {"prompt": "What is 15 + 27?"},
    {"prompt": "If a train travels 60 miles in 1.5 hours, what is its average speed in mph?"},
    {"prompt": "Solve for x: 2x + 5 = 17"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Reward Functions
#
# Wrap or bind any reward functions that need extra kwargs (like `solution`)
# using functools.partial before adding them to this list.
# For this mock run we only use format_reward_func which needs no extra args.
# ─────────────────────────────────────────────────────────────────────────────
reward_funcs = [format_reward_func]

# ─────────────────────────────────────────────────────────────────────────────
# Test Gradient Loop  (g_test via SFT)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Computing test (Code) gradients via SFT ===")
test_infos = []
for idx, sample in enumerate(Z_test):
    print(f"  g_test[{idx}] ...", end=" ", flush=True)
    g = compute_sft_gradient(model, tokenizer, sample["prompt"], sample["solution"], DEVICE)
    test_infos.append({"grad": g})
    clear_cache()
    print(f"norm={g.norm().item():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Train Gradient Loop  (g_train via simulated GRPO)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Computing train (Math) gradients via RLVR ===")
train_infos = []
for idx, sample in enumerate(Z_train):
    print(f"  g_train[{idx}] ...", end=" ", flush=True)
    g = compute_rlvr_gradient(
        model, tokenizer, sample["prompt"], reward_funcs,
        G=4, device=DEVICE, enable_vllm=ENABLE_VLLM,
    )
    train_infos.append({"grad": g})
    clear_cache()
    print(f"norm={g.norm().item():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Influence Matrix  (N_test × M_train)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Building Influence Matrix ===")
calculator = InfluenceCalculator(TracInInfluence(learning_rate=LEARNING_RATE))
influence_matrix = calculator.compute_matrix(test_infos, train_infos)

print(f"\nInfluence Matrix ({len(Z_test)} test × {len(Z_train)} train):")
print(influence_matrix)
