import json
import sys
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
    next_results_dir,
    resolve_results_dir,
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
from influence_rlvr.prompts import (
    build_code_prompt,
    build_r1_math_prompt,
    extract_gsm8k_target,
)

MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
RUN_NAME = "smoke"
RUN_DIR = f"./outputs/{RUN_NAME}"
OUTPUT_DIR = f"{RUN_DIR}/rlvr-output"

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
N_CODE_TRAIN = N_MATH
MATH_EVAL_SPLIT = "test"
MATH_EVAL_PERCENT = 1
CODE_TRAIN_SPLIT = "train"
CODE_EVAL_SPLIT = "validation"
CODE_EVAL_PERCENT = 5
EVAL_MAX_NEW_TOKENS = 128
INFLUENCE_MODE = "historical"
EXPERIMENT_MODE = "math_grpo"
CODE_EVAL_DO_SAMPLE = True
CODE_EVAL_NUM_SAMPLES = 2
CODE_EVAL_TEMPERATURE = 0.6
CODE_EVAL_TOP_P = 0.95
SKIP_TRAINING = False
RESULTS_REUSE_POLICY = "ask"


def percent_slice(split_name, percent):
    if percent <= 0:
        return None
    return f"{split_name}[:{percent}%]"


def normalize_results_reuse_policy(policy):
    value = str(policy).strip().lower()
    if value not in {"ask", "reuse", "new"}:
        raise ValueError(
            "Unsupported RESULTS_REUSE_POLICY="
            f"{policy!r}. Use 'ask', 'reuse', or 'new'."
        )
    return value


def finalize_results_dir(
    run_dir,
    results_path,
    reusing_results_dir,
    results_config_fingerprint,
):
    if not reusing_results_dir:
        print(f"Allocating new results directory: {results_path.resolve()}/")
        return results_path, False, results_config_fingerprint

    if RESULTS_REUSE_POLICY == "reuse":
        print(
            "Reusing existing results directory for matching config: "
            f"{results_path.resolve()}/"
        )
        return results_path, True, results_config_fingerprint

    if RESULTS_REUSE_POLICY == "new":
        new_path = next_results_dir(run_dir)
        print(
            "Matching results directory found, but "
            "RESULTS_REUSE_POLICY='new' so a fresh directory will be used: "
            f"{new_path.resolve()}/"
        )
        return new_path, False, results_config_fingerprint

    if not sys.stdin.isatty():
        new_path = next_results_dir(run_dir)
        print(
            "Matching results directory found, but no interactive terminal is "
            "available for confirmation. Allocating a fresh results directory: "
            f"{new_path.resolve()}/"
        )
        return new_path, False, results_config_fingerprint

    print(
        "Warning: matching results directory already exists and rerunning this "
        f"analysis will overwrite files in {results_path.resolve()}/"
    )
    answer = input(
        "Reuse this results directory? [y]es / [n]ew / [a]bort: "
    ).strip().lower()
    if answer in {"y", "yes"}:
        print(
            "Reusing existing results directory for matching config: "
            f"{results_path.resolve()}/"
        )
        return results_path, True, results_config_fingerprint
    if answer in {"", "n", "new", "no"}:
        new_path = next_results_dir(run_dir)
        print(f"Allocating new results directory: {new_path.resolve()}/")
        return new_path, False, results_config_fingerprint
    raise SystemExit("Aborted to avoid overwriting existing results.")


RESULTS_REUSE_POLICY = normalize_results_reuse_policy(RESULTS_REUSE_POLICY)


def normalize_experiment_mode(mode):
    value = str(mode).strip().lower()
    if value not in {"math_grpo", "code_grpo", "base_eval"}:
        raise ValueError(
            f"Unsupported EXPERIMENT_MODE={mode!r}. "
            "Use 'math_grpo', 'code_grpo', or 'base_eval'."
        )
    return value


EXPERIMENT_MODE = normalize_experiment_mode(EXPERIMENT_MODE)


def format_math(example, idx):
    return {
        "prompt": build_r1_math_prompt(example["question"]),
        "solution": extract_gsm8k_target(example["answer"]),
        "train_index": idx,
    }


def format_code(example, idx=None):
    payload = {
        "prompt": build_code_prompt(example["text"]),
        "solution": example["code"],
        "test_list": example["test_list"],
        "test_setup_code": example["test_setup_code"],
        "challenge_test_list": example["challenge_test_list"],
    }
    if idx is not None:
        payload["train_index"] = idx
    return payload


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
            challenge_test_list=sample.get("challenge_test_list"),
        ),
    ]


if EXPERIMENT_MODE == "code_grpo":
    replay_reward_fn_builder = build_code_reward_fns
else:
    replay_reward_fn_builder = build_math_reward_fns


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


def save_base_checkpoint(peft_model, tokenizer_obj, output_dir):
    checkpoint_dir = Path(output_dir) / "checkpoint-0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(checkpoint_dir)
    tokenizer_obj.save_pretrained(checkpoint_dir)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state_path.write_text(json.dumps({
        "global_step": 0,
        "log_history": [{
            "step": 0,
            "learning_rate": 0.0,
        }],
    }, indent=2))
    return checkpoint_dir


save_base_checkpoint(model, tokenizer, OUTPUT_DIR)

print("\nLoading datasets...")
math_data = load_dataset("openai/gsm8k", "main", split=f"train[:{N_MATH}]")
math_train_dataset = math_data.map(format_math, with_indices=True)
code_train_data = load_dataset("mbpp", split=f"{CODE_TRAIN_SPLIT}[:{N_CODE_TRAIN}]")
code_train_dataset = code_train_data.map(format_code, with_indices=True)
code_data = load_dataset("mbpp", split=f"test[:{N_CODE}]")
test_dataset = code_data.map(format_code)
math_eval_data = load_dataset("openai/gsm8k", "main", split=percent_slice(MATH_EVAL_SPLIT, MATH_EVAL_PERCENT))
math_eval_dataset = math_eval_data.map(format_math, with_indices=True)
code_eval_data = load_dataset("mbpp", split=percent_slice(CODE_EVAL_SPLIT, CODE_EVAL_PERCENT))
code_eval_dataset = code_eval_data.map(format_code)

if EXPERIMENT_MODE == "code_grpo":
    training_domain = "Code"
    training_dataset = code_train_dataset
    training_reward_funcs = [mbpp_execution_reward_func]
else:
    training_domain = "Math"
    training_dataset = math_train_dataset
    training_reward_funcs = [format_reward_func, accuracy_reward_func]

replay_train_dataset = training_dataset
test_domain = "Code"

print(f"  RL train ({training_domain}): {len(training_dataset)}")
print(f"  Replay test ({test_domain}): {len(test_dataset)}")
print(f"  Code train pool ({CODE_TRAIN_SPLIT}): {len(code_train_dataset)}")
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
        reward_funcs=training_reward_funcs,
        args=training_args,
        train_dataset=training_dataset,
        processing_class=tokenizer,
        history_output_dir=OUTPUT_DIR,
    )
    trainer.train()
    clear_cache(DEVICE)
elif EXPERIMENT_MODE == "base_eval":
    print("\nEXPERIMENT_MODE='base_eval' — skipping Phase 1")

print("\n" + "=" * 80)
print("PHASE 2: Trajectory Replay")
print("=" * 80)
checkpoint_schedule = build_checkpoint_schedule(OUTPUT_DIR, default_learning_rate=LEARNING_RATE)

effective_influence_mode = INFLUENCE_MODE
if EXPERIMENT_MODE == "base_eval" and INFLUENCE_MODE == "historical":
    print(
        "EXPERIMENT_MODE='base_eval' has no optimizer-step history; "
        "falling back to dense influence mode."
    )
    effective_influence_mode = "dense"

batch_history_manifest = None
batch_weight_lookup = None
batch_history_fingerprint = None
if effective_influence_mode == "historical":
    batch_history_manifest = load_batch_history(OUTPUT_DIR)
    if batch_history_manifest is None:
        raise RuntimeError("Historical batch history not found.")
    batch_weight_lookup = build_batch_weight_lookup(batch_history_manifest)
    batch_history_fingerprint = build_batch_history_fingerprint(batch_history_manifest)
    batch_weight_lookup[0] = {
        "total_rows": 0,
        "microbatch_count": 0,
        "weights": {},
    }

checkpoint_infos = collect_checkpoint_infos(
    model,
    tokenizer,
    checkpoint_schedule,
    test_dataset,
    replay_train_dataset,
    DEVICE,
    reward_fn_builder=replay_reward_fn_builder,
    G=G_TRAIN,
    enable_vllm=False,
    test_limit=len(test_dataset),
    train_limit=min(N_TRAIN_REPLAY, len(replay_train_dataset)),
    include_debug=False,
    base_seed=TRAIN_GRAD_SEED,
    test_reward_fn_builder=build_code_reward_fns,
    test_G=G_TEST,
    epsilon=GRPO_EPSILON,
    beta=GRPO_BETA,
    influence_mode=effective_influence_mode,
    train_step_weight_lookup=batch_weight_lookup,
    math_eval_dataset=math_eval_dataset,
    code_eval_dataset=code_eval_dataset,
    eval_max_new_tokens=EVAL_MAX_NEW_TOKENS,
    code_eval_do_sample=CODE_EVAL_DO_SAMPLE,
    code_eval_num_samples=CODE_EVAL_NUM_SAMPLES,
    code_eval_temperature=CODE_EVAL_TEMPERATURE,
    code_eval_top_p=CODE_EVAL_TOP_P,
)

print("\n" + "=" * 80)
print("PHASE 3: Influence")
print("=" * 80)
trajectory_tracin = TrajectoryTracInInfluence(normalize=False)
tracin_matrix, tracin_breakdown = trajectory_tracin.compute_matrix(checkpoint_infos, return_breakdown=True)
trajectory_datainf = TrajectoryDataInfInfluence(lambda_damp=LAMBDA_DAMP, normalize=False)
datainf_matrix, datainf_breakdown = trajectory_datainf.compute_matrix(checkpoint_infos, return_breakdown=True)

results_config = {
    "model_id": MODEL_ID,
    "run_name": RUN_NAME,
    "output_dir": OUTPUT_DIR,
    "influence_mode": effective_influence_mode,
    "experiment_mode": EXPERIMENT_MODE,
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
    "n_code_train": N_CODE_TRAIN,
    "n_train_replay": N_TRAIN_REPLAY,
    "code_train_split": CODE_TRAIN_SPLIT,
    "math_eval_split": MATH_EVAL_SPLIT,
    "math_eval_percent": MATH_EVAL_PERCENT,
    "code_eval_split": CODE_EVAL_SPLIT,
    "code_eval_percent": CODE_EVAL_PERCENT,
    "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
    "code_eval_do_sample": CODE_EVAL_DO_SAMPLE,
    "code_eval_num_samples": CODE_EVAL_NUM_SAMPLES,
    "code_eval_temperature": CODE_EVAL_TEMPERATURE,
    "code_eval_top_p": CODE_EVAL_TOP_P,
    "lambda_damp": LAMBDA_DAMP,
    "train_grad_seed": TRAIN_GRAD_SEED,
    "device": str(DEVICE),
    "batch_history_fingerprint": batch_history_fingerprint,
    "train_domain": training_domain,
    "test_domain": test_domain,
}
results_path, reusing_results_dir, results_config_fingerprint = resolve_results_dir(
    RUN_DIR,
    results_config,
)
results_path, reusing_results_dir, results_config_fingerprint = finalize_results_dir(
    RUN_DIR,
    results_path,
    reusing_results_dir,
    results_config_fingerprint,
)
results_config["results_name"] = results_path.name
results_config["results_dir"] = str(results_path)
results_config["results_config_fingerprint"] = results_config_fingerprint

save_results_bundle(
    results_path,
    tracin_matrix,
    datainf_matrix,
    tracin_breakdown,
    datainf_breakdown,
    checkpoint_infos,
    results_config,
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
