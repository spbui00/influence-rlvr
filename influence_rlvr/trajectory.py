import json
import os
import re

from peft import load_peft_weights, set_peft_model_state_dict

from .gradients import compute_rlvr_gradient, compute_sft_gradient
from .utils import clear_cache


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")
_CHECKPOINT_SEED_STRIDE = 100000
_TEST_SEED_OFFSET = 10000


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


def collect_test_infos(peft_model, tokenizer, dataset, device, limit=None):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    test_infos = []

    for idx in range(count):
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
        clear_cache(device)

    return test_infos


def collect_reward_infos(
    peft_model,
    tokenizer,
    dataset,
    device,
    reward_fn_builder,
    G=4,
    enable_vllm=False,
    limit=None,
    include_debug=False,
    seed_base=None,
    epsilon=0.2,
    beta=0.0,
    ref_model=None,
):
    count = len(dataset) if limit is None else min(limit, len(dataset))
    reward_infos = []
    zero_cases = []

    for idx in range(count):
        sample = dataset[idx]
        reward_funcs = reward_fn_builder(sample, G)
        seed = None if seed_base is None else seed_base + idx
        result = compute_rlvr_gradient(
            peft_model,
            tokenizer,
            sample["prompt"],
            reward_funcs,
            G=G,
            device=device,
            enable_vllm=enable_vllm,
            return_debug=include_debug,
            seed=seed,
            epsilon=epsilon,
            beta=beta,
            ref_model=ref_model,
        )

        if include_debug:
            grad, debug = result
        else:
            grad = result
            debug = None

        info = {
            "grad": grad,
            "prompt": sample["prompt"],
            "solution": sample.get("solution"),
        }
        if debug is not None:
            info["debug"] = debug
        reward_infos.append(info)

        if grad.norm().item() <= 1e-12:
            zero_cases.append(idx)
        clear_cache(device)

    return reward_infos, zero_cases


def collect_train_infos(
    peft_model,
    tokenizer,
    dataset,
    device,
    reward_fn_builder,
    G=4,
    enable_vllm=False,
    limit=None,
    include_debug=False,
    seed_base=None,
    epsilon=0.2,
    beta=0.0,
    ref_model=None,
):
    return collect_reward_infos(
        peft_model,
        tokenizer,
        dataset,
        device,
        reward_fn_builder,
        G=G,
        enable_vllm=enable_vllm,
        limit=limit,
        include_debug=include_debug,
        seed_base=seed_base,
        epsilon=epsilon,
        beta=beta,
        ref_model=ref_model,
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
    test_limit=None,
    train_limit=None,
    include_debug=False,
    base_seed=None,
    test_reward_fn_builder=None,
    test_G=None,
    epsilon=0.2,
    beta=0.0,
    ref_model=None,
):
    checkpoint_infos = []

    for checkpoint in checkpoint_schedule:
        load_adapter_checkpoint(peft_model, checkpoint["path"])

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
            )
            zero_test_cases = []
        else:
            test_infos, zero_test_cases = collect_reward_infos(
                peft_model,
                tokenizer,
                test_dataset,
                device,
                test_reward_fn_builder,
                G=G if test_G is None else test_G,
                enable_vllm=enable_vllm,
                limit=test_limit,
                include_debug=False,
                seed_base=test_seed_base,
                epsilon=epsilon,
                beta=beta,
                ref_model=ref_model,
            )

        train_infos, zero_train_cases = collect_train_infos(
            peft_model,
            tokenizer,
            train_dataset,
            device,
            reward_fn_builder,
            G=G,
            enable_vllm=enable_vllm,
            limit=train_limit,
            include_debug=include_debug,
            seed_base=train_seed_base,
            epsilon=epsilon,
            beta=beta,
            ref_model=ref_model,
        )

        checkpoint_infos.append({
            "step": checkpoint["step"],
            "path": checkpoint["path"],
            "learning_rate": checkpoint["learning_rate"],
            "test_infos": test_infos,
            "zero_test_cases": zero_test_cases,
            "train_infos": train_infos,
            "zero_train_cases": zero_train_cases,
        })
        clear_cache(device)

    return checkpoint_infos
