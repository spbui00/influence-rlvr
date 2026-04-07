import json
import os
import re
import time

from peft import load_peft_weights, set_peft_model_state_dict

from .eval import evaluate_code_dataset, evaluate_math_dataset
from .generation import clear_vllm_engine_cache
from .gradients import (
    compute_policy_gradient_bundle,
    compute_sft_gradient,
)
from .modes import (
    CodeEvalConfig,
    GenerationBackend,
    GeometryFeatureMode,
    GradientObjective,
    InfluenceMode,
    ReplayGradientConfig,
    VLLMConfig,
)
from .utils import clear_cache


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
_CHECKPOINT_SEED_STRIDE = 100000
_TEST_SEED_OFFSET = 10000


def _progress_print(message, enabled):
    if enabled:
        print(message, flush=True)


def _checkpoint_prefix(checkpoint_index, checkpoint_count, checkpoint_step):
    if checkpoint_index is None or checkpoint_count is None:
        return f"[checkpoint step {checkpoint_step}]"
    return f"[checkpoint {checkpoint_index}/{checkpoint_count} | step {checkpoint_step}]"


def checkpoint_step(checkpoint_dir):
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    match = _CHECKPOINT_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory: {checkpoint_dir}")
    return int(match.group(1))


def list_checkpoint_dirs(output_dir):
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Checkpoint output directory not found: {output_dir}")

    checkpoint_dirs = []
    for entry in os.listdir(output_dir):
        path = os.path.join(output_dir, entry)
        if os.path.isdir(path) and _CHECKPOINT_RE.match(entry):
            checkpoint_dirs.append(path)

    checkpoint_dirs.sort(key=checkpoint_step)
    return checkpoint_dirs


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _lookup_learning_rate(trainer_state, step, default_learning_rate):
    last_seen = None
    for item in trainer_state.get("log_history", []):
        item_step = item.get("step")
        if item_step is None or "learning_rate" not in item:
            continue
        if item_step == step:
            return float(item["learning_rate"])
        if item_step <= step and (last_seen is None or item_step > last_seen[0]):
            last_seen = (item_step, float(item["learning_rate"]))

    if last_seen is not None:
        return last_seen[1]
    return float(default_learning_rate)


def build_checkpoint_schedule(output_dir, default_learning_rate):
    schedule = []
    for path in list_checkpoint_dirs(output_dir):
        trainer_state_path = os.path.join(path, "trainer_state.json")
        trainer_state = _load_json(trainer_state_path) if os.path.exists(trainer_state_path) else {}
        step = int(trainer_state.get("global_step", checkpoint_step(path)))
        learning_rate = _lookup_learning_rate(trainer_state, step, default_learning_rate)
        schedule.append({
            "step": step,
            "path": path,
            "learning_rate": learning_rate,
        })

    schedule.sort(key=lambda item: item["step"])
    return schedule


def load_adapter_checkpoint(peft_model, checkpoint_dir, adapter_name="default"):
    adapter_weights = load_peft_weights(checkpoint_dir, device="cpu")
    set_peft_model_state_dict(peft_model, adapter_weights, adapter_name=adapter_name)
    if hasattr(peft_model, "set_adapter"):
        peft_model.set_adapter(adapter_name)
    peft_model.zero_grad()
    return peft_model


def ensure_reference_adapter(peft_model, source_adapter="default", ref_adapter="ref"):
    if not hasattr(peft_model, "peft_config"):
        return peft_model
    if ref_adapter in peft_model.peft_config:
        return peft_model

    peft_model.add_adapter(ref_adapter, peft_model.peft_config[source_adapter])
    for name, param in peft_model.named_parameters():
        if f".{source_adapter}." not in name:
            continue
        ref_name = name.replace(f".{source_adapter}.", f".{ref_adapter}.")
        ref_param = peft_model.get_parameter(ref_name)
        ref_param.data.copy_(param.data)
    return peft_model


def collect_test_infos(
    peft_model,
    tokenizer,
    dataset,
    device,
    limit=None,
    progress=False,
    progress_prefix="",
):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    test_infos = []

    for idx in range(count):
        sample_start = time.time()
        sample = dataset[idx]
        grad = compute_sft_gradient(
            peft_model,
            tokenizer,
            sample["prompt"],
            sample["solution"],
            device,
        )
        test_infos.append({
            "grad": grad,
            "prompt": sample["prompt"],
            "solution": sample["solution"],
        })
        grad_norm = grad.norm().item()
        elapsed = time.time() - sample_start
        _progress_print(
            (
                f"{progress_prefix} test sample {idx + 1}/{count} done "
                f"in {elapsed:.1f}s | ||g||={grad_norm:.6f}"
            ).strip(),
            progress,
        )

    return test_infos


def collect_reward_infos(
    peft_model,
    tokenizer,
    dataset,
    device,
    reward_fn_builder,
    G=4,
    enable_vllm=False,
    generation_backend=None,
    limit=None,
    include_debug=False,
    seed_base=None,
    epsilon=0.2,
    beta=0.0,
    ref_model=None,
    sample_weight_lookup=None,
    sample_index_key=None,
    gradient_objective_mode=GradientObjective.GRPO_TRAIN,
    geometry_feature_mode=GeometryFeatureMode.NONE,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    vllm_config=None,
    adapter_path=None,
    model_id=None,
    progress=False,
    progress_prefix="",
):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    reward_infos = []
    zero_cases = []

    for idx in range(count):
        sample_start = time.time()
        sample = dataset[idx]
        reward_funcs = reward_fn_builder(sample, G)
        seed = None if seed_base is None else seed_base + idx
        result = compute_policy_gradient_bundle(
            peft_model,
            tokenizer,
            sample["prompt"],
            reward_funcs,
            G=G,
            device=device,
            enable_vllm=enable_vllm,
            generation_backend=generation_backend,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            epsilon=epsilon,
            beta=beta,
            ref_model=ref_model,
            objective_mode=gradient_objective_mode,
            geometry_feature_mode=geometry_feature_mode,
            vllm_config=vllm_config,
            adapter_path=adapter_path,
            model_id=model_id,
        )

        grad = result["grad"]
        debug = result["debug"] if include_debug else None
        geometry_feature = result.get("geometry_feature")

        info = {
            "grad": grad,
            "prompt": sample["prompt"],
            "solution": sample.get("solution"),
        }
        if geometry_feature is not None:
            info["geometry_feature"] = geometry_feature
        if sample_index_key is not None:
            sample_index = int(sample.get(sample_index_key, idx))
            info["train_index"] = sample_index
            if sample_weight_lookup is not None:
                info["historical_weight"] = float(
                    sample_weight_lookup.get(sample_index, 0.0)
                )
        if debug is not None:
            info["debug"] = debug
        reward_infos.append(info)

        grad_norm = grad.norm().item()
        if grad_norm <= 1e-12:
            zero_cases.append(idx)
        elapsed = time.time() - sample_start
        _progress_print(
            (
                f"{progress_prefix} sample {idx + 1}/{count} done "
                f"in {elapsed:.1f}s | ||g||={grad_norm:.6f}"
            ).strip(),
            progress,
        )

    return reward_infos, zero_cases


def collect_train_infos(
    peft_model,
    tokenizer,
    dataset,
    device,
    reward_fn_builder,
    G=4,
    enable_vllm=False,
    generation_backend=None,
    limit=None,
    include_debug=False,
    seed_base=None,
    epsilon=0.2,
    beta=0.0,
    ref_model=None,
    sample_weight_lookup=None,
    gradient_objective_mode=GradientObjective.GRPO_TRAIN,
    geometry_feature_mode=GeometryFeatureMode.NONE,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    vllm_config=None,
    adapter_path=None,
    model_id=None,
    progress=False,
    progress_prefix="",
):
    return collect_reward_infos(
        peft_model,
        tokenizer,
        dataset,
        device,
        reward_fn_builder,
        G=G,
        enable_vllm=enable_vllm,
        generation_backend=generation_backend,
        limit=limit,
        include_debug=include_debug,
        seed_base=seed_base,
        epsilon=epsilon,
        beta=beta,
        ref_model=ref_model,
        sample_weight_lookup=sample_weight_lookup,
        sample_index_key="train_index",
        gradient_objective_mode=gradient_objective_mode,
        geometry_feature_mode=geometry_feature_mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        vllm_config=vllm_config,
        adapter_path=adapter_path,
        model_id=model_id,
        progress=progress,
        progress_prefix=progress_prefix,
    )


def collect_checkpoint_infos(
    peft_model,
    tokenizer,
    checkpoint_schedule,
    test_dataset,
    train_dataset,
    device,
    reward_fn_builder,
    G=4,
    enable_vllm=False,
    generation_backend=None,
    test_limit=None,
    train_limit=None,
    include_debug=False,
    base_seed=None,
    test_reward_fn_builder=None,
    test_G=None,
    epsilon=0.2,
    beta=0.0,
    ref_model=None,
    influence_mode=InfluenceMode.DENSE,
    train_step_weight_lookup=None,
    math_eval_dataset=None,
    code_eval_dataset=None,
    eval_max_new_tokens=256,
    code_eval_do_sample=False,
    code_eval_num_samples=1,
    code_eval_temperature=0.6,
    code_eval_top_p=0.95,
    train_gradient_objective_mode=GradientObjective.GRPO_TRAIN,
    test_gradient_objective_mode=GradientObjective.GRPO_TRAIN,
    train_geometry_feature_mode=GeometryFeatureMode.NONE,
    replay_max_new_tokens=256,
    replay_temperature=0.7,
    replay_top_p=0.9,
    vllm_config=None,
    model_id=None,
    progress=True,
):
    influence_mode = InfluenceMode.parse(influence_mode)
    generation_backend = (
        GenerationBackend.VLLM
        if generation_backend is None and enable_vllm
        else GenerationBackend.parse(
            GenerationBackend.HF if generation_backend is None else generation_backend
        )
    )
    code_eval_config = CodeEvalConfig(
        do_sample=code_eval_do_sample,
        num_samples=code_eval_num_samples,
        temperature=code_eval_temperature,
        top_p=code_eval_top_p,
    )
    replay_gradient_config = ReplayGradientConfig(
        train_objective=GradientObjective.parse(train_gradient_objective_mode),
        test_objective=GradientObjective.parse(test_gradient_objective_mode),
        train_geometry_feature=GeometryFeatureMode.parse(train_geometry_feature_mode),
        max_new_tokens=replay_max_new_tokens,
        temperature=replay_temperature,
        top_p=replay_top_p,
    )
    if vllm_config is None:
        vllm_config = VLLMConfig()

    checkpoint_infos = []
    checkpoint_count = len(checkpoint_schedule)

    for checkpoint_index, checkpoint in enumerate(checkpoint_schedule, start=1):
        checkpoint_start = time.time()
        prefix = _checkpoint_prefix(
            checkpoint_index, checkpoint_count, checkpoint["step"]
        )
        _progress_print(
            f"{prefix} loading adapter from {checkpoint['path']}",
            progress,
        )
        load_adapter_checkpoint(peft_model, checkpoint["path"])

        math_eval = None
        code_eval = None
        if math_eval_dataset is not None:
            _progress_print(f"{prefix} evaluating held-out math", progress)
            math_eval = evaluate_math_dataset(
                peft_model,
                tokenizer,
                math_eval_dataset,
                device,
                max_new_tokens=eval_max_new_tokens,
                progress=False,
                enable_vllm=enable_vllm,
                generation_backend=generation_backend,
                vllm_config=vllm_config,
                adapter_path=checkpoint["path"],
                model_id=model_id,
            )
        if code_eval_dataset is not None:
            _progress_print(f"{prefix} evaluating held-out code", progress)
            code_eval = evaluate_code_dataset(
                peft_model,
                tokenizer,
                code_eval_dataset,
                device,
                max_new_tokens=eval_max_new_tokens,
                do_sample=code_eval_config.do_sample,
                num_samples=code_eval_config.num_samples,
                temperature=code_eval_config.temperature,
                top_p=code_eval_config.top_p,
                progress=False,
                enable_vllm=enable_vllm,
                generation_backend=generation_backend,
                vllm_config=vllm_config,
                adapter_path=checkpoint["path"],
                model_id=model_id,
            )

        if (
            enable_vllm
            and generation_backend == GenerationBackend.VLLM
            and (math_eval_dataset is not None or code_eval_dataset is not None)
        ):
            clear_vllm_engine_cache()
            # clear_cache(device)

        train_seed_base = None
        test_seed_base = None
        if base_seed is not None:
            train_seed_base = base_seed + checkpoint["step"] * _CHECKPOINT_SEED_STRIDE
            test_seed_base = train_seed_base + _TEST_SEED_OFFSET

        if test_reward_fn_builder is None:
            test_infos = collect_test_infos(
                peft_model,
                tokenizer,
                test_dataset,
                device,
                limit=test_limit,
                progress=progress,
                progress_prefix=f"{prefix} [test]",
            )
            zero_test_cases = []
        else:
            _progress_print(f"{prefix} collecting reward-based test gradients", progress)
            test_infos, zero_test_cases = collect_reward_infos(
                peft_model,
                tokenizer,
                test_dataset,
                device,
                test_reward_fn_builder,
                G=G if test_G is None else test_G,
                enable_vllm=enable_vllm,
                generation_backend=generation_backend,
                limit=test_limit,
                include_debug=False,
                seed_base=test_seed_base,
                epsilon=epsilon,
                beta=beta,
                ref_model=ref_model,
                gradient_objective_mode=replay_gradient_config.test_objective,
                max_new_tokens=replay_gradient_config.max_new_tokens,
                temperature=replay_gradient_config.temperature,
                top_p=replay_gradient_config.top_p,
                vllm_config=vllm_config,
                adapter_path=checkpoint["path"],
                model_id=model_id,
                progress=progress,
                progress_prefix=f"{prefix} [test]",
            )

        _progress_print(f"{prefix} collecting train gradients", progress)
        step_weight_info = None
        if train_step_weight_lookup is not None:
            step_weight_info = train_step_weight_lookup.get(int(checkpoint["step"]))
            if influence_mode == InfluenceMode.HISTORICAL and step_weight_info is None:
                raise RuntimeError(
                    f"Missing historical batch metadata for checkpoint step {checkpoint['step']}."
                )
        train_infos, zero_train_cases = collect_train_infos(
            peft_model,
            tokenizer,
            train_dataset,
            device,
            reward_fn_builder,
            G=G,
            enable_vllm=enable_vllm,
            generation_backend=generation_backend,
            limit=train_limit,
            include_debug=include_debug,
            seed_base=train_seed_base,
            epsilon=epsilon,
            beta=beta,
            ref_model=ref_model,
            sample_weight_lookup=(
                step_weight_info.get("weights")
                if influence_mode == InfluenceMode.HISTORICAL and step_weight_info is not None
                else None
            ),
            gradient_objective_mode=replay_gradient_config.train_objective,
            geometry_feature_mode=replay_gradient_config.train_geometry_feature,
            max_new_tokens=replay_gradient_config.max_new_tokens,
            temperature=replay_gradient_config.temperature,
            top_p=replay_gradient_config.top_p,
            vllm_config=vllm_config,
            adapter_path=checkpoint["path"],
            model_id=model_id,
            progress=progress,
            progress_prefix=f"{prefix} [train]",
        )

        checkpoint_infos.append({
            "step": checkpoint["step"],
            "path": checkpoint["path"],
            "learning_rate": checkpoint["learning_rate"],
            "test_infos": test_infos,
            "zero_test_cases": zero_test_cases,
            "train_infos": train_infos,
            "zero_train_cases": zero_train_cases,
            "math_eval": math_eval,
            "code_eval": code_eval,
            "historical_total_rows": (
                None
                if step_weight_info is None
                else step_weight_info.get("total_rows")
            ),
        })
        checkpoint_elapsed = time.time() - checkpoint_start
        _progress_print(
            (
                f"{prefix} completed in {checkpoint_elapsed:.1f}s | "
                f"zero-test={len(zero_test_cases)} zero-train={len(zero_train_cases)}"
            ),
            progress,
        )
        # clear_cache(device)

    return checkpoint_infos
