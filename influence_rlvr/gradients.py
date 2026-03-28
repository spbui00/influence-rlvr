import warnings
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .modes import GeometryFeatureMode, GradientObjective
from .utils import clear_cache, tokenize_prompt, extract_lora_gradients, get_reward_name


def _set_generation_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_sampling_model(peft_model, old_peft_model=None):
    return peft_model if old_peft_model is None else old_peft_model


@contextmanager
def _use_adapter(model, adapter_name):
    if adapter_name is None or not hasattr(model, "set_adapter"):
        yield
        return

    previous_adapter = getattr(model, "active_adapter", None)
    model.set_adapter(adapter_name)
    try:
        yield
    finally:
        if previous_adapter is not None:
            model.set_adapter(previous_adapter)


def _tokenize_responses(tokenizer, response_texts, device):
    tokenized = []
    max_len = 0
    for text in response_texts:
        response_ids = tokenizer(
            text + tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.squeeze(0)
        tokenized.append(response_ids)
        max_len = max(max_len, response_ids.shape[0])

    if max_len == 0:
        raise RuntimeError("No response tokens were generated.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    response_ids = torch.full(
        (len(tokenized), max_len),
        pad_token_id,
        dtype=tokenized[0].dtype,
        device=device,
    )
    response_mask = torch.zeros((len(tokenized), max_len), dtype=torch.long, device=device)

    for idx, ids in enumerate(tokenized):
        length = ids.shape[0]
        response_ids[idx, :length] = ids.to(device)
        response_mask[idx, :length] = 1

    return response_ids, response_mask


def _compute_per_token_logps(model, prompt_ids, prompt_attention_mask, response_ids, response_mask):
    prompt_batch = prompt_ids.repeat(response_ids.shape[0], 1)
    prompt_mask_batch = prompt_attention_mask.repeat(response_ids.shape[0], 1)
    input_ids = torch.cat([prompt_batch, response_ids], dim=1)
    attention_mask = torch.cat([prompt_mask_batch, response_mask], dim=1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    prompt_len = prompt_ids.shape[1]
    completion_logits = outputs.logits[:, prompt_len - 1:prompt_len - 1 + response_ids.shape[1], :]
    log_probs = F.log_softmax(completion_logits, dim=-1)
    per_token_logps = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    return per_token_logps


def _lora_trainable_params(peft_model):
    return [
        param
        for _, param in peft_model.named_parameters()
        if param.requires_grad
    ]


def _grad_vector_from_scalar(peft_model, scalar, *, retain_graph=False):
    params = _lora_trainable_params(peft_model)
    grads = torch.autograd.grad(
        scalar,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat_grads = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat_grads.append(torch.zeros_like(param, dtype=torch.float32).view(-1).cpu())
        else:
            flat_grads.append(grad.detach().float().view(-1).cpu().clone())
    if not flat_grads:
        raise RuntimeError("No LoRA gradients were found for the requested scalar objective.")
    return torch.cat(flat_grads)


def _sanitize_grad_vector(grad_vector, *, context):
    if torch.isfinite(grad_vector).all():
        return grad_vector
    n_bad = int((~torch.isfinite(grad_vector)).sum().item())
    warnings.warn(
        f"{context}: {n_bad}/{grad_vector.numel()} non-finite gradient entries detected. "
        "Replacing with zeros.",
        stacklevel=2,
    )
    return torch.nan_to_num(grad_vector, nan=0.0, posinf=0.0, neginf=0.0)


def _generate_response_texts(
    sampling_model,
    tokenizer,
    prompt_ids,
    prompt_attention_mask,
    *,
    G,
    max_new_tokens,
    temperature,
    top_p,
):
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        generated = sampling_model.generate(
            input_ids=prompt_ids.repeat(G, 1),
            attention_mask=prompt_attention_mask.repeat(G, 1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_texts = []
    for i in range(G):
        gen_ids = generated[i, prompt_len:]
        response_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return response_texts


def _evaluate_rewards(reward_funcs, completions_trl, device):
    total_rewards = torch.zeros(len(completions_trl), device=device)
    reward_breakdown = {}
    for reward_fn in reward_funcs:
        scores = reward_fn(completions_trl)
        reward_breakdown[get_reward_name(reward_fn)] = [float(score) for score in scores]
        total_rewards += torch.tensor(scores, device=device, dtype=torch.float32)
    return total_rewards, reward_breakdown


def _sequence_log_stats(per_token_logps, response_mask):
    token_mask = response_mask.float()
    token_counts = token_mask.sum(dim=1).clamp(min=1.0)
    sequence_log_probs = (per_token_logps * token_mask).sum(dim=1)
    mean_sequence_log_probs = sequence_log_probs / token_counts
    return token_mask, token_counts, sequence_log_probs, mean_sequence_log_probs


def _compute_grpo_policy_loss(
    total_rewards,
    per_token_logps,
    old_per_token_logps,
    response_mask,
    *,
    epsilon,
    beta,
    ref_per_token_logps=None,
    advantage_eps=1e-4,
):
    advantages = (total_rewards - total_rewards.mean()) / (total_rewards.std() + advantage_eps)
    log_ratio = per_token_logps - old_per_token_logps
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0)
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratios = torch.exp(log_ratio)
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

    advantages = advantages.unsqueeze(1)
    per_token_objective_1 = ratios * advantages
    per_token_objective_2 = clipped_ratios * advantages
    per_token_objective = torch.min(per_token_objective_1, per_token_objective_2)

    per_token_kl = torch.zeros_like(per_token_logps)
    if beta != 0.0:
        if ref_per_token_logps is None:
            raise ValueError("A reference policy is required when beta is non-zero.")
        ref_log_diff = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(ref_log_diff) - ref_log_diff - 1.0

    token_mask, token_counts, _, _ = _sequence_log_stats(per_token_logps, response_mask)
    per_sequence_objective = (
        (per_token_objective - beta * per_token_kl) * token_mask
    ).sum(dim=1) / token_counts
    policy_loss = -per_sequence_objective.mean()
    return policy_loss, advantages.squeeze(1), per_token_kl


def _compute_expected_reward_policy_loss(total_rewards, sequence_log_probs):
    return -(total_rewards.detach() * sequence_log_probs).mean()


def compute_policy_gradient_bundle(
    peft_model,
    tokenizer,
    prompt,
    reward_funcs,
    *,
    G=4,
    device="cpu",
    enable_vllm=False,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    seed=None,
    epsilon=0.2,
    beta=0.0,
    old_peft_model=None,
    ref_model=None,
    advantage_eps=1e-4,
    objective_mode=GradientObjective.GRPO_TRAIN,
    geometry_feature_mode=GeometryFeatureMode.NONE,
):
    objective_mode = GradientObjective.parse(objective_mode)
    geometry_feature_mode = GeometryFeatureMode.parse(geometry_feature_mode)

    peft_model.eval()
    peft_model.zero_grad()
    sampling_model = _get_sampling_model(peft_model, old_peft_model)
    sampling_model.eval()
    if old_peft_model is not None:
        old_peft_model.eval()
    if ref_model is not None:
        ref_model.eval()

    prompt_text, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)

    if enable_vllm:
        raise NotImplementedError(
            "vLLM generation backend is not yet implemented. "
            "Set enable_vllm=False to use standard HF generate()."
        )

    _set_generation_seed(seed)
    response_texts = _generate_response_texts(
        sampling_model,
        tokenizer,
        prompt_ids,
        prompt_attention_mask,
        G=G,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    completions_trl = [[{"role": "assistant", "content": text}] for text in response_texts]
    total_rewards, reward_breakdown = _evaluate_rewards(reward_funcs, completions_trl, device)

    response_ids, response_mask = _tokenize_responses(tokenizer, response_texts, device)
    per_token_logps = _compute_per_token_logps(
        peft_model,
        prompt_ids,
        prompt_attention_mask,
        response_ids,
        response_mask,
    )
    if old_peft_model is None:
        old_per_token_logps = per_token_logps.detach()
    else:
        old_per_token_logps = _compute_old_per_token_logps(
            peft_model,
            old_peft_model,
            prompt_ids,
            prompt_attention_mask,
            response_ids,
            response_mask,
        )

    ref_per_token_logps = None
    if beta != 0.0:
        ref_per_token_logps = _compute_ref_per_token_logps(
            peft_model,
            ref_model,
            prompt_ids,
            prompt_attention_mask,
            response_ids,
            response_mask,
        )

    token_mask, token_counts, sequence_log_probs, mean_sequence_log_probs = _sequence_log_stats(
        per_token_logps,
        response_mask,
    )

    if objective_mode == GradientObjective.GRPO_TRAIN:
        objective, advantages, per_token_kl = _compute_grpo_policy_loss(
            total_rewards,
            per_token_logps,
            old_per_token_logps,
            response_mask,
            epsilon=epsilon,
            beta=beta,
            ref_per_token_logps=ref_per_token_logps,
            advantage_eps=advantage_eps,
        )
        objective_name = "grpo_policy_loss"
    elif objective_mode == GradientObjective.EXPECTED_REWARD_PG:
        objective = _compute_expected_reward_policy_loss(total_rewards, sequence_log_probs)
        advantages = total_rewards.detach()
        per_token_kl = torch.zeros_like(per_token_logps)
        objective_name = "expected_reward_pg_loss"
    else:
        raise ValueError(f"Unsupported objective_mode={objective_mode!r}.")

    geometry_feature = None
    retain_graph = geometry_feature_mode != GeometryFeatureMode.NONE
    grad_vector = _grad_vector_from_scalar(
        peft_model,
        objective,
        retain_graph=retain_graph,
    )
    grad_vector = _sanitize_grad_vector(
        grad_vector,
        context=f"compute_policy_gradient_bundle[{objective_mode}]",
    )

    if geometry_feature_mode == GeometryFeatureMode.POLICY_SCORE:
        geometry_scalar = sequence_log_probs.mean()
        geometry_feature = _grad_vector_from_scalar(
            peft_model,
            geometry_scalar,
            retain_graph=False,
        )
        geometry_feature = _sanitize_grad_vector(
            geometry_feature,
            context="compute_policy_gradient_bundle[policy_score]",
        )
    elif geometry_feature_mode != GeometryFeatureMode.NONE:
        raise ValueError(
            f"Unsupported geometry_feature_mode={geometry_feature_mode!r}."
        )

    debug_info = {
        "prompt_text": prompt_text,
        "responses": response_texts,
        "reward_breakdown": reward_breakdown,
        "total_rewards": total_rewards.detach().cpu().tolist(),
        "advantages": advantages.detach().cpu().tolist(),
        "log_probs": sequence_log_probs.detach().float().cpu().tolist(),
        "sequence_log_probs": sequence_log_probs.detach().float().cpu().tolist(),
        "mean_sequence_log_probs": mean_sequence_log_probs.detach().float().cpu().tolist(),
        "policy_loss": float(objective.detach().float().cpu()),
        "mean_kl": float(((per_token_kl * token_mask).sum(dim=1) / token_counts).mean().detach().cpu()),
        "epsilon": epsilon,
        "beta": beta,
        "seed": seed,
        "objective_mode": objective_mode,
        "objective_name": objective_name,
        "geometry_feature_mode": geometry_feature_mode,
    }
    if geometry_feature is not None:
        debug_info["geometry_feature_norm"] = float(geometry_feature.norm().item())

    del per_token_logps, old_per_token_logps, objective
    clear_cache(device)
    return {
        "grad": grad_vector,
        "geometry_feature": geometry_feature,
        "debug": debug_info,
    }


def _compute_old_per_token_logps(
    peft_model,
    old_peft_model,
    prompt_ids,
    prompt_attention_mask,
    response_ids,
    response_mask,
):
    if old_peft_model is None:
        return _compute_per_token_logps(
            peft_model,
            prompt_ids,
            prompt_attention_mask,
            response_ids,
            response_mask,
        ).detach()

    with torch.no_grad():
        return _compute_per_token_logps(
            old_peft_model,
            prompt_ids,
            prompt_attention_mask,
            response_ids,
            response_mask,
        )


def _compute_ref_per_token_logps(
    peft_model,
    ref_model,
    prompt_ids,
    prompt_attention_mask,
    response_ids,
    response_mask,
):
    if ref_model is None and not hasattr(peft_model, "peft_config"):
        return None

    if ref_model is not None:
        with torch.no_grad():
            return _compute_per_token_logps(
                ref_model,
                prompt_ids,
                prompt_attention_mask,
                response_ids,
                response_mask,
            )

    ref_adapter_name = "ref" if "ref" in peft_model.peft_config else None
    if ref_adapter_name is not None:
        with torch.no_grad(), _use_adapter(peft_model, ref_adapter_name):
            return _compute_per_token_logps(
                peft_model,
                prompt_ids,
                prompt_attention_mask,
                response_ids,
                response_mask,
            )

    with torch.no_grad(), peft_model.disable_adapter():
        return _compute_per_token_logps(
            peft_model,
            prompt_ids,
            prompt_attention_mask,
            response_ids,
            response_mask,
        )


def compute_sft_gradient(peft_model, tokenizer, prompt, target, device):
    peft_model.eval()
    peft_model.zero_grad()

    _, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)
    target_ids = tokenizer(
        target + tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    target_attention_mask = torch.ones_like(target_ids)
    attention_mask = torch.cat([prompt_attention_mask, target_attention_mask], dim=1)

    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    grad_vector = extract_lora_gradients(peft_model)
    clear_cache(device)
    return grad_vector


def compute_rlvr_gradient(
    peft_model, tokenizer, prompt, reward_funcs,
    G=4, device="cpu", enable_vllm=False,
    max_new_tokens=256, temperature=0.7, top_p=0.9,
    return_debug=False,
    seed=None,
    epsilon=0.2,
    beta=0.0,
    old_peft_model=None,
    ref_model=None,
    advantage_eps=1e-4,
):
    result = compute_policy_gradient_bundle(
        peft_model,
        tokenizer,
        prompt,
        reward_funcs,
        G=G,
        device=device,
        enable_vllm=enable_vllm,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        epsilon=epsilon,
        beta=beta,
        old_peft_model=old_peft_model,
        ref_model=ref_model,
        advantage_eps=advantage_eps,
        objective_mode=GradientObjective.GRPO_TRAIN,
        geometry_feature_mode=GeometryFeatureMode.NONE,
    )
    if return_debug:
        return result["grad"], result["debug"]
    return result["grad"]
