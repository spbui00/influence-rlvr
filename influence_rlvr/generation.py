from __future__ import annotations

import atexit
import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .modes import GenerationBackend, VLLMConfig

_VLLM_ENGINE_CACHE: dict[tuple[Any, ...], object] = {}


@dataclass
class RolloutBatch:
    texts: list[str]
    token_ids: torch.Tensor
    response_mask: torch.Tensor


def rollout_to_completions(rollout: RolloutBatch) -> list[list[dict[str, str]]]:
    return [[{"role": "assistant", "content": text}] for text in rollout.texts]


def _destroy_vllm_process_group(llm_engine: object) -> None:
    dp_group = getattr(llm_engine, "dp_group", None)
    if dp_group is None or getattr(llm_engine, "external_launcher_dp", False):
        return
    try:
        dist_utils = importlib.import_module("vllm.distributed.utils")
        destroy_pg = getattr(
            dist_utils,
            "stateless_destroy_torch_distributed_process_group",
        )
    except Exception:
        shutdown = getattr(dp_group, "shutdown", None)
        if callable(shutdown):
            shutdown()
        else:
            destroy = getattr(torch.distributed, "destroy_process_group", None)
            if callable(destroy):
                destroy(dp_group)
    else:
        destroy_pg(dp_group)
    try:
        llm_engine.dp_group = None
    except Exception:
        pass


def _shutdown_vllm_llm_instance(engine: object) -> None:
    llm_engine = getattr(engine, "llm_engine", None)
    if llm_engine is None:
        return
    core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None) if core is not None else None
    try:
        if callable(shutdown):
            shutdown()
    finally:
        try:
            llm_engine.engine_core = None
        except Exception:
            pass
        _destroy_vllm_process_group(llm_engine)


def clear_vllm_engine_cache() -> None:
    for llm in list(_VLLM_ENGINE_CACHE.values()):
        try:
            _shutdown_vllm_llm_instance(llm)
        except Exception:
            pass
    _VLLM_ENGINE_CACHE.clear()


atexit.register(clear_vllm_engine_cache)


def _normalize_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _validate_generation_request(num_samples: int, do_sample: bool) -> None:
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
    if not do_sample and num_samples != 1:
        raise ValueError(
            "Greedy generation only supports num_samples=1. "
            "Set do_sample=True to request multiple samples."
        )


def _stack_token_sequences(
    token_sequences: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_sequences:
        raise RuntimeError("No response tokens were generated.")
    max_len = max(int(ids.shape[0]) for ids in token_sequences)
    if max_len <= 0:
        raise RuntimeError("No response tokens were generated.")

    token_ids = torch.full(
        (len(token_sequences), max_len),
        pad_token_id,
        dtype=token_sequences[0].dtype,
        device=device,
    )
    response_mask = torch.zeros(
        (len(token_sequences), max_len),
        dtype=torch.long,
        device=device,
    )
    for row_idx, ids in enumerate(token_sequences):
        length = int(ids.shape[0])
        token_ids[row_idx, :length] = ids.to(device)
        response_mask[row_idx, :length] = 1
    return token_ids, response_mask


def rollout_batch_from_token_sequences(
    tokenizer,
    token_sequences: list[torch.Tensor],
    *,
    device: str | torch.device,
) -> RolloutBatch:
    runtime_device = _normalize_device(device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    token_ids, response_mask = _stack_token_sequences(
        token_sequences,
        pad_token_id=pad_token_id,
        device=runtime_device,
    )
    texts = [
        tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        for ids in token_sequences
    ]
    return RolloutBatch(
        texts=texts,
        token_ids=token_ids,
        response_mask=response_mask,
    )


def _trim_hf_continuation(
    continuation: torch.Tensor,
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    token_ids = continuation.detach().to(dtype=torch.long).cpu()
    if token_ids.ndim != 1:
        token_ids = token_ids.reshape(-1)

    if eos_token_id is not None:
        eos_hits = (token_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_hits.numel() > 0:
            stop = int(eos_hits[0].item()) + 1
            return token_ids[:stop]

    if pad_token_id is not None and pad_token_id != eos_token_id:
        pad_hits = (token_ids == pad_token_id).nonzero(as_tuple=False)
        if pad_hits.numel() > 0:
            stop = int(pad_hits[0].item())
            return token_ids[:stop]

    return token_ids


def _hf_generate_rollout_batch(
    sampling_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    num_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> RolloutBatch:
    sampling_model.eval()
    generate_kwargs = {
        "input_ids": prompt_ids,
        "attention_mask": prompt_attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_samples,
        })
    with torch.inference_mode():
        generated = sampling_model.generate(**generate_kwargs)

    prompt_len = int(prompt_ids.shape[1])
    token_sequences = [
        _trim_hf_continuation(
            sequence[prompt_len:],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        for sequence in generated[:num_samples]
    ]
    return rollout_batch_from_token_sequences(
        tokenizer,
        token_sequences,
        device=prompt_ids.device,
    )


def _require_vllm_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("The vLLM backend requires a CUDA device.")
    if sys.platform != "linux":
        raise RuntimeError("The vLLM backend is only supported on Linux/CUDA hosts.")


def _resolve_model_id(sampling_model, tokenizer, model_id: str | None) -> str:
    if model_id:
        return model_id

    candidates = [
        getattr(tokenizer, "name_or_path", None),
        getattr(getattr(sampling_model, "config", None), "_name_or_path", None),
    ]
    base_model = getattr(sampling_model, "base_model", None)
    base_inner = getattr(base_model, "model", None)
    candidates.append(getattr(getattr(base_inner, "config", None), "_name_or_path", None))
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise ValueError(
        "Could not infer the base model id for vLLM. "
        "Pass model_id explicitly when using the vLLM backend."
    )


def _vllm_engine_key(
    model_id: str,
    tokenizer,
    config: VLLMConfig,
) -> tuple[Any, ...]:
    return (
        model_id,
        getattr(tokenizer, "name_or_path", model_id),
        float(config.gpu_memory_utilization),
        int(config.tensor_parallel_size),
        config.max_model_len,
        config.max_num_seqs,
        bool(config.enforce_eager),
    )


def _load_vllm_types():
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install the optional dependency first, "
            "for example with `uv sync --extra vllm`."
        ) from exc
    return LLM, SamplingParams, LoRARequest


def _get_vllm_engine(
    model_id: str,
    tokenizer,
    config: VLLMConfig,
):
    key = _vllm_engine_key(model_id, tokenizer, config)
    engine = _VLLM_ENGINE_CACHE.get(key)
    if engine is not None:
        return engine

    LLM, _, _ = _load_vllm_types()
    kwargs = {
        "model": model_id,
        "tokenizer": getattr(tokenizer, "name_or_path", model_id),
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enable_lora": True,
        "enforce_eager": config.enforce_eager,
    }
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    if config.max_num_seqs is not None:
        kwargs["max_num_seqs"] = config.max_num_seqs
    engine = LLM(**kwargs)
    _VLLM_ENGINE_CACHE[key] = engine
    return engine


def _build_lora_request(adapter_path: str | Path | None):
    if adapter_path is None:
        return None
    adapter = Path(adapter_path)
    if not adapter.exists():
        raise FileNotFoundError(f"vLLM adapter path does not exist: {adapter}")
    _, _, LoRARequest = _load_vllm_types()
    digest = hashlib.sha256(str(adapter.resolve()).encode()).hexdigest()
    request_id = int(digest[:12], 16)
    if request_id <= 0:
        request_id = 1
    return LoRARequest(
        lora_name=adapter.name,
        lora_int_id=request_id,
        lora_path=str(adapter),
    )


def _call_vllm_generate(engine, prompt_token_ids, sampling_params, lora_request):
    prompt_list = [prompt_token_ids]
    kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": False,
    }
    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    try:
        return engine.generate(prompt_token_ids=prompt_list, **kwargs)
    except TypeError:
        pass

    prompts = [{"prompt_token_ids": prompt_token_ids}]
    try:
        return engine.generate(prompts, **kwargs)
    except TypeError:
        return engine.generate(prompts=prompts, **kwargs)


def _vllm_generate_rollout_batch(
    sampling_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    num_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int | None,
    vllm_config: VLLMConfig,
    adapter_path: str | Path | None,
    model_id: str | None,
) -> RolloutBatch:
    device = prompt_ids.device
    _require_vllm_runtime(device)
    model_name = _resolve_model_id(sampling_model, tokenizer, model_id)
    engine = _get_vllm_engine(model_name, tokenizer, vllm_config)
    _, SamplingParams, _ = _load_vllm_types()

    sampling_kwargs: dict[str, Any] = {
        "n": num_samples,
        "max_tokens": max_new_tokens,
        "temperature": temperature if do_sample else 0.0,
        "top_p": top_p if do_sample else 1.0,
    }
    if seed is not None:
        sampling_kwargs["seed"] = seed
    sampling_params = SamplingParams(**sampling_kwargs)

    outputs = _call_vllm_generate(
        engine,
        prompt_ids.squeeze(0).detach().cpu().tolist(),
        sampling_params,
        _build_lora_request(adapter_path),
    )
    if not outputs:
        raise RuntimeError("vLLM returned no outputs for the requested prompt.")

    request_output = outputs[0]
    token_sequences = []
    texts = []
    for output in request_output.outputs[:num_samples]:
        output_token_ids = torch.tensor(list(output.token_ids), dtype=torch.long)
        token_sequences.append(output_token_ids)
        text = getattr(output, "text", None)
        if text is None:
            text = tokenizer.decode(output_token_ids.tolist(), skip_special_tokens=True)
        texts.append(text)

    rollout = rollout_batch_from_token_sequences(
        tokenizer,
        token_sequences,
        device=device,
    )
    rollout.texts = texts
    return rollout


def generate_rollout_batch(
    sampling_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    backend: GenerationBackend | str = GenerationBackend.HF,
    num_samples: int = 1,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int | None = None,
    vllm_config: VLLMConfig | None = None,
    adapter_path: str | Path | None = None,
    model_id: str | None = None,
) -> RolloutBatch:
    _validate_generation_request(num_samples, do_sample)
    backend = GenerationBackend.parse(backend)
    if backend == GenerationBackend.HF:
        return _hf_generate_rollout_batch(
            sampling_model,
            tokenizer,
            prompt_ids,
            prompt_attention_mask,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    return _vllm_generate_rollout_batch(
        sampling_model,
        tokenizer,
        prompt_ids,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        vllm_config=VLLMConfig() if vllm_config is None else vllm_config,
        adapter_path=adapter_path,
        model_id=model_id,
    )
