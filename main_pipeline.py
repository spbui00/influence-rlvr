import json
import os
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
    build_batch_history_fingerprint,
    build_batch_weight_lookup,
    build_cache_fingerprint,
    load_batch_history,
    load_grad_cache,
    next_results_dir,
    resolve_results_dir,
    save_grad_cache,
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
    ensure_reference_adapter,
    format_reward_func,
    mbpp_execution_reward_func,
)
from influence_rlvr.prompts import (
    build_code_prompt,
    build_r1_math_prompt,
    extract_gsm8k_target,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — edit these before launching
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
RUN_NAME = "run6"
RUN_DIR = f"./outputs/{RUN_NAME}"
OUTPUT_DIR = f"{RUN_DIR}/rlvr-output"

LEARNING_RATE = 1e-4
MAX_STEPS = 200
SAVE_STEPS = 5
PER_DEVICE_BATCH = 8
GRAD_ACCUM_STEPS = 2
GRPO_BETA = 0.04
GRPO_EPSILON = 0.2
G_TRAIN = 8
G_TEST = 8
GENERATION_BATCH_SIZE = 128
TRAIN_GRAD_SEED = 1234
LAMBDA_DAMP = 0.1
N_MATH = 300
N_CODE = 10
N_TRAIN_REPLAY = N_MATH
N_CODE_TRAIN = N_MATH
MATH_EVAL_SPLIT = "test"
MATH_EVAL_PERCENT = 10
CODE_TRAIN_SPLIT = "train"
CODE_EVAL_SPLIT = "validation"
CODE_EVAL_PERCENT = 10
EVAL_MAX_NEW_TOKENS = 256
INFLUENCE_MODE = "historical"
EXPERIMENT_MODE = "math_grpo"
CODE_EVAL_DO_SAMPLE = True
CODE_EVAL_NUM_SAMPLES = 4
CODE_EVAL_TEMPERATURE = 0.6
CODE_EVAL_TOP_P = 0.95

SKIP_TRAINING = False
ENABLE_GRAD_CACHE = False
RESULTS_REUSE_POLICY = "ask"


def normalize_influence_mode(mode):
    value = str(mode).strip().lower()
    if value == "counterfactual":
        value = "dense"
    if value not in {"historical", "dense"}:
        raise ValueError(
            f"Unsupported INFLUENCE_MODE={mode!r}. Use 'historical' or 'dense'."
        )
    return value


INFLUENCE_MODE = normalize_influence_mode(INFLUENCE_MODE)


def normalize_experiment_mode(mode):
    value = str(mode).strip().lower()
    if value not in {"math_grpo", "code_grpo", "base_eval"}:
        raise ValueError(
            f"Unsupported EXPERIMENT_MODE={mode!r}. "
            "Use 'math_grpo', 'code_grpo', or 'base_eval'."
        )
    return value


EXPERIMENT_MODE = normalize_experiment_mode(EXPERIMENT_MODE)


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

optimizer_step_rows = PER_DEVICE_BATCH * GRAD_ACCUM_STEPS
generation_prompt_pool = GENERATION_BATCH_SIZE // max(G_TRAIN, 1)
print(
    "Historical coverage target: "
    f"{optimizer_step_rows} prompt rows/optimizer step, "
    f"{generation_prompt_pool} unique prompts/generation cycle"
)
if generation_prompt_pool < optimizer_step_rows:
    print(
        "Warning: generation_batch_size is smaller than the optimizer-step "
        "prompt demand, so historical attribution may still be sparse."
    )

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


# ═══════════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════════
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


def percent_slice(split_name, percent):
    if percent <= 0:
        return None
    if percent > 100:
        raise ValueError(f"Split percent must be in [0, 100], got {percent}.")
    return f"{split_name}[:{percent}%]"


print("\nLoading datasets...")
math_train_data = load_dataset("openai/gsm8k", "main", split=f"train[:{N_MATH}]")
math_train_dataset = math_train_data.map(format_math, with_indices=True)
code_train_data = load_dataset("mbpp", split=f"{CODE_TRAIN_SPLIT}[:{N_CODE_TRAIN}]")
code_train_dataset = code_train_data.map(format_code, with_indices=True)
code_test_data = load_dataset("mbpp", split=f"test[:{N_CODE}]")
test_dataset = code_test_data.map(format_code)
math_eval_split = percent_slice(MATH_EVAL_SPLIT, MATH_EVAL_PERCENT)
math_eval_dataset = None
if math_eval_split is not None:
    math_eval_data = load_dataset("openai/gsm8k", "main", split=math_eval_split)
    math_eval_dataset = math_eval_data.map(format_math, with_indices=True)
code_eval_split = percent_slice(CODE_EVAL_SPLIT, CODE_EVAL_PERCENT)
code_eval_dataset = None
if code_eval_split is not None:
    code_eval_data = load_dataset("mbpp", split=code_eval_split)
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

print(f"  RL train ({training_domain}): {len(training_dataset)} samples")
print(f"  Replay test ({test_domain}): {len(test_dataset)} samples")
print(f"  Code train pool ({CODE_TRAIN_SPLIT}): {len(code_train_dataset)} samples")
if math_eval_dataset is not None:
    print(f"  Held-out math eval ({MATH_EVAL_SPLIT}): {len(math_eval_dataset)} samples")
if code_eval_dataset is not None:
    print(f"  Held-out code eval ({CODE_EVAL_SPLIT}): {len(code_eval_dataset)} samples")


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
        save_steps=SAVE_STEPS,
        save_total_limit=None,
        bf16=True,
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

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")
    clear_cache(DEVICE)


pipeline_t0 = time.time()
training_elapsed_s = None

if SKIP_TRAINING:
    print("\nSKIP_TRAINING=True — skipping Phase 1")
elif EXPERIMENT_MODE == "base_eval":
    print("\nEXPERIMENT_MODE='base_eval' — skipping Phase 1")
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
        f"Make sure training ran with save_steps={SAVE_STEPS}."
    )

print(f"Found {len(checkpoint_schedule)} checkpoints:")
for cp in checkpoint_schedule:
    print(f"  step {cp['step']:>3d}  lr={cp['learning_rate']:.6e}")

if GRPO_BETA != 0.0:
    ensure_reference_adapter(model)

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
        raise RuntimeError(
            "Historical influence mode requires historical batch metadata, "
            f"but {OUTPUT_DIR}/historical_batch_history.json was not found. "
            "Use a new run that logged batch history, or switch INFLUENCE_MODE='dense'."
        )
    batch_weight_lookup = build_batch_weight_lookup(batch_history_manifest)
    batch_history_fingerprint = build_batch_history_fingerprint(batch_history_manifest)
    print(
        "Loaded historical batch history: "
        f"{len(batch_history_manifest.steps)} optimizer steps "
        f"(fingerprint={batch_history_fingerprint})"
    )
    batch_weight_lookup[0] = {
        "total_rows": 0,
        "microbatch_count": 0,
        "weights": {},
    }
else:
    print("Using dense counterfactual influence mode.")


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


RESULTS_CONFIG = {
    "model_id": MODEL_ID,
    "run_name": RUN_NAME,
    "output_dir": OUTPUT_DIR,
    "influence_mode": effective_influence_mode,
    "experiment_mode": EXPERIMENT_MODE,
    "enable_grad_cache": ENABLE_GRAD_CACHE,
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
    RESULTS_CONFIG,
)
results_path, reusing_results_dir, results_config_fingerprint = finalize_results_dir(
    RUN_DIR,
    results_path,
    reusing_results_dir,
    results_config_fingerprint,
)
RESULTS_DIR = str(results_path)
GRAD_CACHE_DIR = str(results_path / "grad_cache")
RESULTS_CONFIG["results_name"] = results_path.name
RESULTS_CONFIG["results_dir"] = RESULTS_DIR
RESULTS_CONFIG["results_config_fingerprint"] = results_config_fingerprint

CACHE_CONFIG = {
    "model_id": MODEL_ID,
    "output_dir": os.path.abspath(OUTPUT_DIR),
    "influence_mode": effective_influence_mode,
    "experiment_mode": EXPERIMENT_MODE,
    "enable_grad_cache": ENABLE_GRAD_CACHE,
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
    "train_grad_seed": TRAIN_GRAD_SEED,
    "batch_history_fingerprint": batch_history_fingerprint,
    "train_domain": training_domain,
    "test_domain": test_domain,
}

CACHE_FINGERPRINT = build_cache_fingerprint(CACHE_CONFIG)


def _run_replay():
    t0 = time.time()
    infos = collect_checkpoint_infos(
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
    elapsed = time.time() - t0
    print(f"\nTrajectory replay completed in {elapsed:.1f}s")
    if ENABLE_GRAD_CACHE:
        save_grad_cache(infos, GRAD_CACHE_DIR, CACHE_FINGERPRINT, CACHE_CONFIG)
        print(f"Gradient cache saved to {Path(GRAD_CACHE_DIR).resolve()}/")
        print(f"  config fingerprint: {CACHE_FINGERPRINT}")
    else:
        print("Gradient cache disabled; skipping grad_cache save.")
    return infos, elapsed


grad_cache_path = Path(GRAD_CACHE_DIR) / "cache_meta.json"
checkpoint_infos = None
replay_elapsed_s = None
if ENABLE_GRAD_CACHE and grad_cache_path.exists():
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
    nonzero_train_weights = None
    if any("historical_weight" in info for info in cp["train_infos"]):
        nonzero_train_weights = sum(
            1
            for info in cp["train_infos"]
            if float(info.get("historical_weight", 1.0)) > 0.0
        )
    print(
        f"  step {cp['step']:>3d}: "
        f"lr={cp['learning_rate']:.6e}, "
        f"mean ||g_test||={np.mean(test_norms):.6f}, "
        f"mean ||g_train||={np.mean(train_norms):.6f}, "
        f"zero-test={cp['zero_test_cases']}, "
        f"zero-train={cp['zero_train_cases']}"
        + (
            ""
            if cp.get("historical_total_rows") is None
            else (
                f", batch-rows={cp['historical_total_rows']}"
                f", active-train={nonzero_train_weights}"
            )
        )
        + (
            ""
            if cp.get("math_eval") is None
            else (
                f", math-exact={cp['math_eval']['accuracy_rate']:.3f}"
                f", math-format={cp['math_eval']['format_rate']:.3f}"
            )
        )
        + (
            ""
            if cp.get("code_eval") is None
            else (
                f", code-pass={cp['code_eval']['pass_rate']:.3f}"
                f", code-compile={cp['code_eval']['compile_rate']:.3f}"
            )
        )
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
