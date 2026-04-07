import json
import os
import random
import sys
import time
from dataclasses import replace
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
    save_results_bundle,
)
from influence_rlvr import (
    HistoricalBatchGRPOTrainer,
    TrajectoryDataInfInfluence,
    TrajectoryFisherInfluence,
    TrajectoryTracInInfluence,
    accuracy_reward_func,
    build_checkpoint_schedule,
    clear_cache,
    collect_checkpoint_infos,
    detect_device,
    ensure_reference_adapter,
    mbpp_execution_reward_func,
    thin_checkpoint_schedule,
)
from influence_rlvr.modes import (
    CheckpointThinningConfig,
    CheckpointThinningMode,
    CodeEvalConfig,
    ExperimentMode,
    GenerationBackend,
    GeometryFeatureMode,
    GradientObjective,
    InfluenceMode,
    ReplayGradientConfig,
    SecondOrderGeometry,
    VLLMConfig,
)
from influence_rlvr.prompts import (
    append_suffix_to_final_user_message,
    build_code_prompt,
    build_r1_math_prompt,
    extract_gsm8k_target,
)
from influence_rlvr.rewards import format_guardrail_reward_func

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — edit these before launching
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RUN_NAME = "train_script1"
RUN_DIR = f"./outputs/{RUN_NAME}"
OUTPUT_DIR = f"{RUN_DIR}/rlvr-output"
RUN_CONFIG_PATH = None  

LEARNING_RATE = 1e-4
MAX_STEPS = 200
SAVE_STEPS = 5
PER_DEVICE_BATCH = 8
GRAD_ACCUM_STEPS = 2
GRPO_BETA = 0.04
GRPO_EPSILON = 0.2
G_TRAIN = 16
G_TEST = 8
GENERATION_BATCH_SIZE = 128
MAX_COMPLETION_LENGTH = None
TRAIN_GRAD_SEED = 1234
LAMBDA_DAMP = 0.1
N_MATH = 300
N_CODE = 10
N_TRAIN_REPLAY = 1000
N_CODE_TRAIN = N_MATH
TRAIN_REPLAY_SUBSET_SEED = None
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
_MATH_PROMPT_MATCHES_TRAINING_SCRIPT = False
TRAINING_RUN_CONFIG = None
TRAINING_RUN_CONFIG_PATH = None
MATH_EVAL_SPLIT = "test"
MATH_EVAL_PERCENT = 0
CODE_TRAIN_SPLIT = "train"
CODE_EVAL_SPLIT = "validation"
CODE_EVAL_PERCENT = 0
EVAL_MAX_NEW_TOKENS = 1024
VLLM_GPU_MEMORY_UTILIZATION = 0.35
INFLUENCE_MODE = InfluenceMode.HISTORICAL
EXPERIMENT_MODE = ExperimentMode.MATH_GRPO
GENERATION_BACKEND = GenerationBackend.VLLM
VLLM_CONFIG = VLLMConfig(gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION)
CODE_EVAL_CONFIG = CodeEvalConfig(
    do_sample=True,
    num_samples=4,
    temperature=0.6,
    top_p=0.95,
)
REPLAY_GRADIENT_CONFIG = ReplayGradientConfig(
    train_objective=GradientObjective.GRPO_TRAIN,
    test_objective=GradientObjective.EXPECTED_REWARD_PG,
    train_geometry_feature=GeometryFeatureMode.POLICY_SCORE,
    second_order_geometry=SecondOrderGeometry.POLICY_SCORE_FISHER,
    fisher_normalize=False,
    max_new_tokens=EVAL_MAX_NEW_TOKENS,
    temperature=0.7,
    top_p=0.9,
    replay_gradient_batch_size=2,
)

SKIP_TRAINING = True
ENABLE_GRAD_CACHE = True
RESULTS_REUSE_POLICY = "ask"

CHECKPOINT_THINNING = CheckpointThinningConfig(
    mode=CheckpointThinningMode.LEARNING_RATE,
    target_count=100
)

_TRAINING_SCRIPT_MATH_FORMAT_SUFFIX = (
    "After </think>, the last line must contain only the final numeric GSM8K answer in "
    "\\boxed{...} (digits / fraction / decimal). Do not write placeholders, "
    "do not repeat this instruction block, and do not use code fences."
)


def _parse_lora_target_modules(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def _run_config_json_path() -> Path:
    if RUN_CONFIG_PATH:
        return Path(RUN_CONFIG_PATH).expanduser().resolve()
    return Path(OUTPUT_DIR).expanduser().resolve() / "run_config.json"


def _apply_training_run_config(cfg: dict, path: Path) -> None:
    global MODEL_ID, LEARNING_RATE, MAX_STEPS, SAVE_STEPS, PER_DEVICE_BATCH, GRAD_ACCUM_STEPS
    global GRPO_BETA, GRPO_EPSILON, G_TRAIN, GENERATION_BATCH_SIZE, TRAIN_GRAD_SEED
    global N_MATH, LORA_R, LORA_ALPHA, LORA_TARGET_MODULES
    global GENERATION_BACKEND, VLLM_CONFIG, REPLAY_GRADIENT_CONFIG, EVAL_MAX_NEW_TOKENS
    global MAX_COMPLETION_LENGTH, _MATH_PROMPT_MATCHES_TRAINING_SCRIPT
    global TRAINING_RUN_CONFIG, TRAINING_RUN_CONFIG_PATH

    TRAINING_RUN_CONFIG = cfg
    TRAINING_RUN_CONFIG_PATH = str(path)
    _MATH_PROMPT_MATCHES_TRAINING_SCRIPT = True

    MODEL_ID = cfg.get("model_id", MODEL_ID)
    LEARNING_RATE = float(cfg.get("learning_rate", LEARNING_RATE))
    MAX_STEPS = int(cfg.get("max_steps", MAX_STEPS))
    SAVE_STEPS = int(cfg.get("save_steps", SAVE_STEPS))
    PER_DEVICE_BATCH = int(cfg.get("per_device_batch", PER_DEVICE_BATCH))
    GRAD_ACCUM_STEPS = int(cfg.get("grad_accum", GRAD_ACCUM_STEPS))
    GRPO_BETA = float(cfg.get("grpo_beta", GRPO_BETA))
    GRPO_EPSILON = float(cfg.get("grpo_epsilon", GRPO_EPSILON))
    G_TRAIN = int(cfg.get("g_train", G_TRAIN))
    TRAIN_GRAD_SEED = int(cfg.get("seed", TRAIN_GRAD_SEED))

    gbs = cfg.get("generation_batch_size")
    GENERATION_BATCH_SIZE = int(gbs) if gbs is not None else None

    mcl = cfg.get("max_completion_length")
    MAX_COMPLETION_LENGTH = int(mcl) if mcl is not None else None
    if MAX_COMPLETION_LENGTH is not None:
        REPLAY_GRADIENT_CONFIG = replace(
            REPLAY_GRADIENT_CONFIG,
            max_new_tokens=MAX_COMPLETION_LENGTH,
        )

    ev = cfg.get("eval_max_new_tokens")
    if ev is not None:
        EVAL_MAX_NEW_TOKENS = int(ev)

    N_MATH = int(cfg.get("n_math", N_MATH))

    LORA_R = int(cfg.get("lora_r", LORA_R))
    LORA_ALPHA = int(cfg.get("lora_alpha", LORA_ALPHA))
    lt = cfg.get("lora_target_modules")
    if isinstance(lt, str) and lt.strip():
        LORA_TARGET_MODULES = _parse_lora_target_modules(lt)

    if cfg.get("hf"):
        GENERATION_BACKEND = GenerationBackend.HF
    else:
        GENERATION_BACKEND = GenerationBackend.VLLM

    vmlm = cfg.get("vllm_max_model_length")
    vmxlr = cfg.get("vllm_max_lora_rank")
    VLLM_CONFIG = VLLMConfig(
        gpu_memory_utilization=float(VLLM_GPU_MEMORY_UTILIZATION),
        tensor_parallel_size=int(
            cfg.get("vllm_tensor_parallel_size", VLLM_CONFIG.tensor_parallel_size)
        ),
        max_model_len=int(vmlm) if vmlm is not None else VLLM_CONFIG.max_model_len,
        max_num_seqs=VLLM_CONFIG.max_num_seqs,
        max_lora_rank=int(vmxlr) if vmxlr is not None else LORA_R,
        enforce_eager=VLLM_CONFIG.enforce_eager,
        training_use_vllm=not bool(cfg.get("hf")),
    )

    out_cfg = cfg.get("output_dir")
    if out_cfg is not None:
        resolved_cfg = Path(str(out_cfg)).expanduser().resolve()
        resolved_out = Path(OUTPUT_DIR).expanduser().resolve()
        if resolved_cfg != resolved_out:
            print(
                "Warning: run_config.json output_dir does not match OUTPUT_DIR:\n"
                f"  run_config: {resolved_cfg}\n"
                f"  OUTPUT_DIR: {resolved_out}"
            )

    print(f"Loaded training run_config.json — applied settings from {path}")


_run_config_path = _run_config_json_path()
if _run_config_path.is_file():
    try:
        _run_data = json.loads(_run_config_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid run_config.json at {_run_config_path}: {exc}"
        ) from exc
    if EXPERIMENT_MODE != ExperimentMode.MATH_GRPO:
        print(
            "Ignoring run_config.json: EXPERIMENT_MODE must be MATH_GRPO "
            f"(got {EXPERIMENT_MODE!r})."
        )
    else:
        _apply_training_run_config(_run_data, _run_config_path)


def normalize_influence_mode(mode):
    value = str(mode).strip().lower()
    if value == "counterfactual":
        return InfluenceMode.DENSE
    try:
        return InfluenceMode.parse(value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported INFLUENCE_MODE={mode!r}. "
            "Use InfluenceMode.HISTORICAL or InfluenceMode.DENSE."
        ) from exc


INFLUENCE_MODE = normalize_influence_mode(INFLUENCE_MODE)


def normalize_experiment_mode(mode):
    try:
        return ExperimentMode.parse(mode)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported EXPERIMENT_MODE={mode!r}. "
            "Use ExperimentMode.MATH_GRPO, CODE_GRPO, or BASE_EVAL."
        ) from exc


EXPERIMENT_MODE = normalize_experiment_mode(EXPERIMENT_MODE)


def normalize_generation_backend(backend):
    try:
        return GenerationBackend.parse(backend)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported GENERATION_BACKEND={backend!r}. "
            "Use GenerationBackend.HF or GenerationBackend.VLLM."
        ) from exc


GENERATION_BACKEND = normalize_generation_backend(GENERATION_BACKEND)


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


def _pipeline_main():
    optimizer_step_rows = PER_DEVICE_BATCH * GRAD_ACCUM_STEPS
    _effective_generation_batch_size = (
        GENERATION_BATCH_SIZE
        if GENERATION_BATCH_SIZE is not None
        else (PER_DEVICE_BATCH * GRAD_ACCUM_STEPS)
    )
    generation_prompt_pool = _effective_generation_batch_size // max(G_TRAIN, 1)
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
    if VLLM_CONFIG.training_use_vllm and GENERATION_BACKEND != GenerationBackend.VLLM:
        raise ValueError(
            "VLLM_CONFIG.training_use_vllm=True requires GENERATION_BACKEND=GenerationBackend.VLLM."
        )
    if GENERATION_BACKEND == GenerationBackend.VLLM:
        if DEVICE.type != "cuda":
            raise ValueError("GenerationBackend.VLLM requires a CUDA device.")
        if sys.platform != "linux":
            raise ValueError("GenerationBackend.VLLM is only supported on Linux/CUDA hosts.")
    print(f"Device: {DEVICE} | generation backend: {GENERATION_BACKEND}")
    if GENERATION_BACKEND == GenerationBackend.VLLM and not VLLM_CONFIG.training_use_vllm:
        print("Phase 1 training will stay on HF/TRL; vLLM is used for replay/eval only.")

    # ═══════════════════════════════════════════════════════════════════════════════
    # Model + Tokenizer
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\nLoading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    if _MATH_PROMPT_MATCHES_TRAINING_SCRIPT:
        tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    ).to(DEVICE)

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()


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


    _will_run_phase1_training = (
        not SKIP_TRAINING and EXPERIMENT_MODE != ExperimentMode.BASE_EVAL
    )
    if _will_run_phase1_training:
        save_base_checkpoint(model, tokenizer, OUTPUT_DIR)
    else:
        print(
            "Skipping save_base_checkpoint (Phase 1 training disabled); "
            "existing checkpoints under OUTPUT_DIR are left unchanged."
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    # Datasets
    # ═══════════════════════════════════════════════════════════════════════════════
    def format_math(example, idx):
        if _MATH_PROMPT_MATCHES_TRAINING_SCRIPT:
            prompt = append_suffix_to_final_user_message(
                build_r1_math_prompt(example["question"]),
                _TRAINING_SCRIPT_MATH_FORMAT_SUFFIX,
            )
        else:
            prompt = build_r1_math_prompt(example["question"])
        return {
            "prompt": prompt,
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
    _math_train_split = "train" if N_MATH <= 0 else f"train[:{N_MATH}]"
    math_train_data = load_dataset("openai/gsm8k", "main", split=_math_train_split)
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

    if EXPERIMENT_MODE == ExperimentMode.CODE_GRPO:
        training_domain = "Code"
        training_dataset = code_train_dataset
        training_reward_funcs = [mbpp_execution_reward_func]
    else:
        training_domain = "Math"
        training_dataset = math_train_dataset
        if _MATH_PROMPT_MATCHES_TRAINING_SCRIPT:
            training_reward_funcs = [format_guardrail_reward_func, accuracy_reward_func]
        else:
            training_reward_funcs = [accuracy_reward_func]

    replay_train_dataset = training_dataset
    train_replay_pool_size = len(replay_train_dataset)
    train_replay_subset_seed_resolved = (
        TRAIN_REPLAY_SUBSET_SEED
        if TRAIN_REPLAY_SUBSET_SEED is not None
        else (TRAIN_GRAD_SEED + 902891)
    )
    if N_TRAIN_REPLAY > 0:
        k = min(N_TRAIN_REPLAY, train_replay_pool_size)
        if k < train_replay_pool_size:
            rng = random.Random(train_replay_subset_seed_resolved)
            indices = sorted(rng.sample(range(train_replay_pool_size), k))
            replay_train_dataset = replay_train_dataset.select(indices)
    train_replay_effective_n = len(replay_train_dataset)
    test_domain = "Code"

    print(f"  RL train ({training_domain}): {len(training_dataset)} samples")
    print(f"  Replay test ({test_domain}): {len(test_dataset)} samples")
    print(f"  Code train pool ({CODE_TRAIN_SPLIT}): {len(code_train_dataset)} samples")
    if math_eval_dataset is not None:
        print(f"  Held-out math eval ({MATH_EVAL_SPLIT}): {len(math_eval_dataset)} samples")
    if code_eval_dataset is not None:
        print(f"  Held-out code eval ({CODE_EVAL_SPLIT}): {len(code_eval_dataset)} samples")
    print(
        f"  Replay train subset: {train_replay_effective_n}/{train_replay_pool_size} "
        f"(cap={N_TRAIN_REPLAY}, seed={train_replay_subset_seed_resolved})"
    )


    # ═══════════════════════════════════════════════════════════════════════════════
    # Phase 1 — GRPO Training
    # ═══════════════════════════════════════════════════════════════════════════════
    def run_training():
        print("\n" + "=" * 80)
        print("PHASE 1: GRPO Training")
        print("=" * 80)

        _grpo_kw = {
            "output_dir": OUTPUT_DIR,
            "report_to": "wandb",
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "max_steps": MAX_STEPS,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": SAVE_STEPS,
            "save_total_limit": None,
            "bf16": True,
            "use_vllm": (
                GENERATION_BACKEND == GenerationBackend.VLLM
                and VLLM_CONFIG.training_use_vllm
            ),
            "num_generations": G_TRAIN,
            "loss_type": "grpo",
            "beta": GRPO_BETA,
            "epsilon": GRPO_EPSILON,
            "importance_sampling_level": "token",
            "scale_rewards": "group",
        }
        if TRAINING_RUN_CONFIG is not None:
            _grpo_kw["seed"] = TRAIN_GRAD_SEED
        if GENERATION_BATCH_SIZE is not None:
            _grpo_kw["generation_batch_size"] = GENERATION_BATCH_SIZE
        if MAX_COMPLETION_LENGTH is not None:
            _grpo_kw["max_completion_length"] = MAX_COMPLETION_LENGTH
        training_args = GRPOConfig(**_grpo_kw)

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
    elif EXPERIMENT_MODE == ExperimentMode.BASE_EVAL:
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

    checkpoint_schedule_full = build_checkpoint_schedule(
        OUTPUT_DIR, default_learning_rate=LEARNING_RATE,
    )
    if not checkpoint_schedule_full:
        raise RuntimeError(
            f"No checkpoints found under {OUTPUT_DIR}. "
            f"Make sure training ran with save_steps={SAVE_STEPS}."
        )

    print(f"Found {len(checkpoint_schedule_full)} checkpoints on disk:")
    for cp in checkpoint_schedule_full:
        print(f"  step {cp['step']:>3d}  lr={cp['learning_rate']:.6e}")

    checkpoint_schedule = thin_checkpoint_schedule(
        checkpoint_schedule_full,
        CHECKPOINT_THINNING,
        log=True,
    )

    if GRPO_BETA != 0.0:
        ensure_reference_adapter(model)

    effective_influence_mode = INFLUENCE_MODE
    if EXPERIMENT_MODE == ExperimentMode.BASE_EVAL and INFLUENCE_MODE == InfluenceMode.HISTORICAL:
        print(
            "EXPERIMENT_MODE='base_eval' has no optimizer-step history; "
            "falling back to dense influence mode."
        )
        effective_influence_mode = InfluenceMode.DENSE

    batch_history_manifest = None
    batch_weight_lookup = None
    batch_history_fingerprint = None
    if effective_influence_mode == InfluenceMode.HISTORICAL:
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
        acc = partial(accuracy_reward_func, solution=[solution] * num_generations)
        if _MATH_PROMPT_MATCHES_TRAINING_SCRIPT:
            return [format_guardrail_reward_func, acc]
        return [acc]


    def build_code_reward_fns(sample, num_generations):
        return [
            partial(
                mbpp_execution_reward_func,
                test_list=sample["test_list"],
                test_setup_code=sample["test_setup_code"],
                challenge_test_list=sample.get("challenge_test_list"),
            ),
        ]


    if EXPERIMENT_MODE == ExperimentMode.CODE_GRPO:
        replay_reward_fn_builder = build_code_reward_fns
    else:
        replay_reward_fn_builder = build_math_reward_fns


    RESULTS_CONFIG = {
        "model_id": MODEL_ID,
        "run_name": RUN_NAME,
        "output_dir": OUTPUT_DIR,
        "training_run_config_path": TRAINING_RUN_CONFIG_PATH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
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
        "n_train_replay": train_replay_effective_n,
        "n_train_replay_cap": N_TRAIN_REPLAY,
        "train_replay_pool_size": train_replay_pool_size,
        "train_replay_subset_seed": train_replay_subset_seed_resolved,
        "code_train_split": CODE_TRAIN_SPLIT,
        "math_eval_split": MATH_EVAL_SPLIT,
        "math_eval_percent": MATH_EVAL_PERCENT,
        "code_eval_split": CODE_EVAL_SPLIT,
        "code_eval_percent": CODE_EVAL_PERCENT,
        "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "generation_backend": GENERATION_BACKEND,
        **CODE_EVAL_CONFIG.to_config_dict(),
        **REPLAY_GRADIENT_CONFIG.to_config_dict(),
        **VLLM_CONFIG.to_config_dict(),
        **CHECKPOINT_THINNING.to_config_dict(),
        "lambda_damp": LAMBDA_DAMP,
        "train_grad_seed": TRAIN_GRAD_SEED,
        "device": str(DEVICE),
        "batch_history_fingerprint": batch_history_fingerprint,
        "train_domain": training_domain,
        "test_domain": test_domain,
        "checkpoint_steps_on_disk": [cp["step"] for cp in checkpoint_schedule_full],
        "checkpoint_steps_replay": [cp["step"] for cp in checkpoint_schedule],
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
        "training_run_config_path": TRAINING_RUN_CONFIG_PATH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
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
        "n_train_replay": train_replay_effective_n,
        "n_train_replay_cap": N_TRAIN_REPLAY,
        "train_replay_pool_size": train_replay_pool_size,
        "train_replay_subset_seed": train_replay_subset_seed_resolved,
        "code_train_split": CODE_TRAIN_SPLIT,
        "math_eval_split": MATH_EVAL_SPLIT,
        "math_eval_percent": MATH_EVAL_PERCENT,
        "code_eval_split": CODE_EVAL_SPLIT,
        "code_eval_percent": CODE_EVAL_PERCENT,
        "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "generation_backend": GENERATION_BACKEND,
        **CODE_EVAL_CONFIG.to_config_dict(),
        **REPLAY_GRADIENT_CONFIG.to_config_dict(),
        **VLLM_CONFIG.to_config_dict(),
        **CHECKPOINT_THINNING.to_config_dict(),
        "train_grad_seed": TRAIN_GRAD_SEED,
        "batch_history_fingerprint": batch_history_fingerprint,
        "train_domain": training_domain,
        "test_domain": test_domain,
        "checkpoint_steps_on_disk": [cp["step"] for cp in checkpoint_schedule_full],
        "checkpoint_steps_replay": [cp["step"] for cp in checkpoint_schedule],
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
            enable_vllm=GENERATION_BACKEND == GenerationBackend.VLLM,
            generation_backend=GENERATION_BACKEND,
            test_limit=len(test_dataset),
            train_limit=train_replay_effective_n,
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
            vllm_config=VLLM_CONFIG,
            model_id=MODEL_ID,
            **CODE_EVAL_CONFIG.to_kwargs(),
            **REPLAY_GRADIENT_CONFIG.to_kwargs(),
            results_dir=RESULTS_DIR,
            tracin_normalize=False,
            lambda_damp=LAMBDA_DAMP,
            datainf_normalize=False,
            fisher_lambda_damp=LAMBDA_DAMP,
            fisher_normalize=REPLAY_GRADIENT_CONFIG.fisher_normalize,
            enable_grad_cache=ENABLE_GRAD_CACHE,
            grad_cache_dir=GRAD_CACHE_DIR if ENABLE_GRAD_CACHE else None,
            cache_fingerprint=CACHE_FINGERPRINT,
            cache_config=CACHE_CONFIG,
        )
        elapsed = time.time() - t0
        print(f"\nTrajectory replay completed in {elapsed:.1f}s")
        if ENABLE_GRAD_CACHE:
            print(
                "Gradient cache written incrementally to "
                f"{Path(GRAD_CACHE_DIR).resolve()}/"
            )
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
        if "mean_test_grad_norm" in cp:
            mean_test_norm = cp["mean_test_grad_norm"]
            mean_train_norm = cp["mean_train_grad_norm"]
        else:
            test_norms = [info["grad"].norm().item() for info in cp["test_infos"]]
            train_norms = [info["grad"].norm().item() for info in cp["train_infos"]]
            mean_test_norm = float(np.mean(test_norms)) if test_norms else 0.0
            mean_train_norm = float(np.mean(train_norms)) if train_norms else 0.0
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
            f"mean ||g_test||={mean_test_norm:.6f}, "
            f"mean ||g_train||={mean_train_norm:.6f}, "
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

    trajectory_fisher = TrajectoryFisherInfluence(
        lambda_damp=LAMBDA_DAMP,
        normalize=REPLAY_GRADIENT_CONFIG.fisher_normalize,
    )
    fisher_matrix, fisher_breakdown = trajectory_fisher.compute_matrix(
        checkpoint_infos, return_breakdown=True,
    )

    np.set_printoptions(precision=6, suppress=False)
    print(f"\nTrajectory TracIn  shape: {tracin_matrix.shape}")
    print(f"  max |score| = {np.abs(tracin_matrix).max():.6e}")
    print(tracin_matrix)

    print(f"\nTrajectory DataInf shape: {datainf_matrix.shape}")
    print(f"  max |score| = {np.abs(datainf_matrix).max():.6e}")
    print(datainf_matrix)

    print(f"\nTrajectory Fisher shape: {fisher_matrix.shape}")
    print(f"  max |score| = {np.abs(fisher_matrix).max():.6e}")
    print(fisher_matrix)

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
        fisher_matrix,
        tracin_breakdown,
        datainf_breakdown,
        fisher_breakdown,
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


if __name__ == "__main__":
    _pipeline_main()
