import warnings
from contextlib import contextmanager
from functools import partial

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch.utils.checkpoint import checkpoint

from .generation import RolloutBatch, generate_rollout_batch, rollout_to_completions
from .modes import GenerationBackend, GeometryFeatureMode, GradientObjective, VLLMConfig
from .utils import extract_lora_gradients, get_reward_name, tokenize_prompt, tokenize_prompts_batch


def _set_generation_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_sampling_model(peft_model, old_peft_model=None):
    return peft_model if old_peft_model is None else old_peft_model


def _resolve_generation_backend(enable_vllm, generation_backend):
    if generation_backend is None:
        return GenerationBackend.VLLM if enable_vllm else GenerationBackend.HF
    backend = GenerationBackend.parse(generation_backend)
    if enable_vllm and backend != GenerationBackend.VLLM:
        raise ValueError(
            "Received enable_vllm=True with generation_backend set to a non-vLLM backend."
        )
    return backend


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


def _microbatch_token_logps_forward(
    model,
    prompt_len,
    res_len,
    input_ids,
    attention_mask,
    response_token_ids,
):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + res_len, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(2, response_token_ids.unsqueeze(-1)).squeeze(-1)


def _compute_per_token_logps(
    model,
    prompt_ids,
    prompt_attention_mask,
    response_ids,
    response_mask,
    micro_batch_size=4,
    use_gradient_checkpointing=None,
):
    n_p = prompt_ids.shape[0]
    n_r = response_ids.shape[0]
    if n_p not in (1, n_r):
        raise ValueError(
            f"prompt batch {n_p} incompatible with response batch {n_r} "
            "(expected 1 or equal batch sizes)."
        )
    if n_r == 0:
        return response_ids.new_zeros((0, response_ids.shape[1]))
    if micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")

    if use_gradient_checkpointing is None:
        use_gradient_checkpointing = torch.is_grad_enabled()

    prompt_len = prompt_ids.shape[1]
    res_len = response_ids.shape[1]
    all_logps = []
    forward_fn = partial(
        _microbatch_token_logps_forward,
        model,
        prompt_len,
        res_len,
    )

    for i in range(0, n_r, micro_batch_size):
        end = min(i + micro_batch_size, n_r)
        curr_res_ids = response_ids[i:end]
        curr_res_mask = response_mask[i:end]

        if n_p == 1:
            b = curr_res_ids.shape[0]
            curr_prompt_ids = prompt_ids.expand(b, -1)
            curr_prompt_mask = prompt_attention_mask.expand(b, -1)
        else:
            curr_prompt_ids = prompt_ids[i:end]
            curr_prompt_mask = prompt_attention_mask[i:end]

        input_ids = torch.cat([curr_prompt_ids, curr_res_ids], dim=1)
        attention_mask = torch.cat([curr_prompt_mask, curr_res_mask], dim=1)

        if use_gradient_checkpointing:
            per_token_logps = checkpoint(
                forward_fn,
                input_ids,
                attention_mask,
                curr_res_ids,
                use_reentrant=False,
            )
        else:
            per_token_logps = forward_fn(input_ids, attention_mask, curr_res_ids)

        all_logps.append(per_token_logps)

    return torch.cat(all_logps, dim=0)


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
            flat_grads.append(torch.zeros(param.numel(), dtype=torch.float32, device=param.device))
        else:
            flat_grads.append(grad.detach().to(dtype=torch.float32).reshape(-1))
    if not flat_grads:
        raise RuntimeError("No LoRA gradients were found for the requested scalar objective.")
    return torch.cat(flat_grads).cpu()


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


def _compute_grpo_policy_loss_per_prompt(
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
    mean_r = total_rewards.mean(dim=1, keepdim=True)
    std_r = total_rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.0)
    advantages = (total_rewards - mean_r) / (std_r + advantage_eps)
    log_ratio = per_token_logps - old_per_token_logps
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0)
    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
    ratios = torch.exp(log_ratio)
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

    advantages = advantages.unsqueeze(2)
    per_token_objective_1 = ratios * advantages
    per_token_objective_2 = clipped_ratios * advantages
    per_token_objective = torch.min(per_token_objective_1, per_token_objective_2)

    per_token_kl = torch.zeros_like(per_token_logps)
    if beta != 0.0:
        if ref_per_token_logps is None:
            raise ValueError("A reference policy is required when beta is non-zero.")
        ref_log_diff = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(ref_log_diff) - ref_log_diff - 1.0

    token_mask = response_mask.float()
    token_counts = token_mask.sum(dim=2).clamp(min=1.0)
    per_sequence_objective = (
        (per_token_objective - beta * per_token_kl) * token_mask
    ).sum(dim=2) / token_counts
    policy_loss_vector = -per_sequence_objective.mean(dim=1)
    return policy_loss_vector, advantages.squeeze(2), per_token_kl


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
    lv, adv, kl = _compute_grpo_policy_loss_per_prompt(
        total_rewards.unsqueeze(0),
        per_token_logps.unsqueeze(0),
        old_per_token_logps.unsqueeze(0),
        response_mask.unsqueeze(0),
        epsilon=epsilon,
        beta=beta,
        ref_per_token_logps=(
            None if ref_per_token_logps is None else ref_per_token_logps.unsqueeze(0)
        ),
        advantage_eps=advantage_eps,
    )
    return lv.squeeze(0), adv.squeeze(0), kl.squeeze(0)


def _compute_expected_reward_policy_loss(total_rewards, sequence_log_probs):
    return -(total_rewards.detach() * sequence_log_probs).mean()


def _compute_expected_reward_policy_loss_per_prompt(total_rewards, sequence_log_probs):
    return -(total_rewards.detach() * sequence_log_probs).mean(dim=1)


def _sequence_log_stats_batched(per_token_logps, response_mask):
    token_mask = response_mask.float()
    token_counts = token_mask.sum(dim=2).clamp(min=1.0)
    sequence_log_probs = (per_token_logps * token_mask).sum(dim=2) / token_counts
    mean_sequence_log_probs = sequence_log_probs.mean(dim=1)
    return token_mask, token_counts, sequence_log_probs, mean_sequence_log_probs


def _lora_trainable_param_names_tuple(peft_model) -> tuple[str, ...]:
    return tuple(n for n, p in peft_model.named_parameters() if p.requires_grad)


def _state_dict_for_functional(
    peft_model,
    ptuple: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    name_set = set(names)
    by_name = dict(zip(names, ptuple))
    out: dict[str, torch.Tensor] = {}
    for k, v in peft_model.state_dict().items():
        if k in name_set:
            out[k] = by_name[k]
        else:
            out[k] = v.detach()
    return out


def _forward_per_token_logps_functional(
    peft_model,
    state_dict: dict[str, torch.Tensor],
    p_ids: torch.Tensor,
    p_am: torch.Tensor,
    r_ids: torch.Tensor,
    r_m: torch.Tensor,
) -> torch.Tensor:
    g, rt = r_ids.shape
    pl = p_ids.shape[0]
    pr = p_ids.unsqueeze(0).expand(g, pl)
    pam = p_am.unsqueeze(0).expand(g, pl)
    inp = torch.cat([pr, r_ids], dim=1)
    am = torch.cat([pam, r_m], dim=1)
    out = functional_call(
        peft_model,
        state_dict,
        (),
        {"input_ids": inp, "attention_mask": am, "use_cache": False},
        tie_weights=False
    )
    logits = out.logits if hasattr(out, "logits") else out[0]
    completion_logits = logits[:, pl - 1 : pl - 1 + rt, :]
    log_probs = F.log_softmax(completion_logits, dim=-1)
    return log_probs.gather(2, r_ids.unsqueeze(-1)).squeeze(-1)


def _loss_grpo_functional(
    ptuple: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
    peft_model,
    p_ids: torch.Tensor,
    p_am: torch.Tensor,
    r_ids: torch.Tensor,
    r_m: torch.Tensor,
    tr: torch.Tensor,
    olp: torch.Tensor,
    rfp: torch.Tensor | None,
    epsilon: float,
    beta: float,
    advantage_eps: float,
) -> torch.Tensor:
    state_dict = _state_dict_for_functional(peft_model, ptuple, names)
    per_token_logps = _forward_per_token_logps_functional(
        peft_model, state_dict, p_ids, p_am, r_ids, r_m
    )
    loss, _, _ = _compute_grpo_policy_loss(
        tr,
        per_token_logps,
        olp,
        r_m,
        epsilon=epsilon,
        beta=beta,
        ref_per_token_logps=None if beta == 0 else rfp,
        advantage_eps=advantage_eps,
    )
    return loss


def _loss_exp_reward_functional(
    ptuple: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
    peft_model,
    p_ids: torch.Tensor,
    p_am: torch.Tensor,
    r_ids: torch.Tensor,
    r_m: torch.Tensor,
    tr: torch.Tensor,
) -> torch.Tensor:
    state_dict = _state_dict_for_functional(peft_model, ptuple, names)
    per_token_logps = _forward_per_token_logps_functional(
        peft_model, state_dict, p_ids, p_am, r_ids, r_m
    )
    token_mask = r_m.float()
    token_counts = token_mask.sum(dim=1).clamp(min=1.0)
    sequence_log_probs = (per_token_logps * token_mask).sum(dim=1) / token_counts
    return -(tr.detach() * sequence_log_probs).mean()


def _loss_geom_functional(
    ptuple: tuple[torch.Tensor, ...],
    names: tuple[str, ...],
    peft_model,
    p_ids: torch.Tensor,
    p_am: torch.Tensor,
    r_ids: torch.Tensor,
    r_m: torch.Tensor,
) -> torch.Tensor:
    state_dict = _state_dict_for_functional(peft_model, ptuple, names)
    per_token_logps = _forward_per_token_logps_functional(
        peft_model, state_dict, p_ids, p_am, r_ids, r_m
    )
    token_mask = r_m.float()
    token_counts = token_mask.sum(dim=1).clamp(min=1.0)
    sequence_log_probs = (per_token_logps * token_mask).sum(dim=1) / token_counts
    return sequence_log_probs.mean()


def _per_sample_grads_vmap(
    loss_fn,
    ptuple: tuple[torch.Tensor, ...],
    stacked_args: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    g_fn = grad(loss_fn)
    return vmap(g_fn, in_dims=(None,) + (0,) * len(stacked_args))(ptuple, *stacked_args)


def _flatten_vmap_grads(
    batched_grads: tuple[torch.Tensor | None, ...],
    batch_size: int,
    param_tuple: tuple[torch.Tensor, ...],
) -> list[torch.Tensor]:
    if len(batched_grads) != len(param_tuple):
        raise RuntimeError("Gradient tuple length does not match parameter tuple.")
    vectors = []
    for b in range(batch_size):
        parts = []
        for g, p in zip(batched_grads, param_tuple):
            if g is None:
                parts.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
            else:
                parts.append(g[b].reshape(-1).to(dtype=torch.float32))
        vectors.append(torch.cat(parts).cpu())
    return vectors


def _per_sample_gradients_batch(
    peft_model,
    objective_mode: GradientObjective,
    names: tuple[str, ...],
    batched_prompt_ids: torch.Tensor,
    batched_prompt_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    total_rewards: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor | None,
    *,
    epsilon: float,
    beta: float,
    advantage_eps: float,
) -> list[torch.Tensor]:
    ptuple = tuple(peft_model.get_parameter(n) for n in names)
    b = int(batched_prompt_ids.shape[0])
    if objective_mode == GradientObjective.GRPO_TRAIN:

        def loss_fn_local(p, *args):
            return _loss_grpo_functional(
                p,
                names,
                peft_model,
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                epsilon=epsilon,
                beta=beta,
                advantage_eps=advantage_eps,
            )

        bg = _per_sample_grads_vmap(
            loss_fn_local,
            ptuple,
            (
                batched_prompt_ids,
                batched_prompt_mask,
                response_ids,
                response_mask,
                total_rewards,
                old_per_token_logps,
                ref_per_token_logps
                if ref_per_token_logps is not None
                else torch.zeros_like(old_per_token_logps),
            ),
        )
    elif objective_mode == GradientObjective.EXPECTED_REWARD_PG:

        def loss_fn_local(p, *args):
            return _loss_exp_reward_functional(
                p,
                names,
                peft_model,
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
            )

        bg = _per_sample_grads_vmap(
            loss_fn_local,
            ptuple,
            (
                batched_prompt_ids,
                batched_prompt_mask,
                response_ids,
                response_mask,
                total_rewards,
            ),
        )
    else:
        raise ValueError(f"Unsupported objective_mode={objective_mode!r} for batched vmap.")
    return _flatten_vmap_grads(bg, b, ptuple)


def _per_sample_geometry_gradients_batch(
    peft_model,
    names: tuple[str, ...],
    batched_prompt_ids: torch.Tensor,
    batched_prompt_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> list[torch.Tensor]:
    ptuple = tuple(peft_model.get_parameter(n) for n in names)

    def loss_fn_local(p, *args):
        return _loss_geom_functional(
            p,
            names,
            peft_model,
            args[0],
            args[1],
            args[2],
            args[3],
        )

    bg = _per_sample_grads_vmap(
        loss_fn_local,
        ptuple,
        (
            batched_prompt_ids,
            batched_prompt_mask,
            response_ids,
            response_mask,
        ),
    )
    return _flatten_vmap_grads(bg, int(batched_prompt_ids.shape[0]), ptuple)


def compute_policy_gradient_bundle_batch(
    peft_model,
    tokenizer,
    prompts: list,
    reward_funcs_batch: list,
    *,
    G: int = 4,
    device: str | torch.device = "cpu",
    enable_vllm: bool = False,
    generation_backend=None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = None,
    epsilon: float = 0.2,
    beta: float = 0.0,
    old_peft_model=None,
    ref_model=None,
    advantage_eps: float = 1e-4,
    objective_mode=GradientObjective.GRPO_TRAIN,
    geometry_feature_mode=GeometryFeatureMode.NONE,
    vllm_config=None,
    adapter_path=None,
    model_id=None,
):
    objective_mode = GradientObjective.parse(objective_mode)
    geometry_feature_mode = GeometryFeatureMode.parse(geometry_feature_mode)
    generation_backend = _resolve_generation_backend(enable_vllm, generation_backend)
    if vllm_config is None:
        vllm_config = VLLMConfig()

    bsz = len(prompts)
    if bsz != len(reward_funcs_batch):
        raise ValueError("prompts and reward_funcs_batch must have the same length.")
    if bsz < 1:
        raise ValueError("Batch must be non-empty.")

    peft_model.eval()
    peft_model.zero_grad()
    sampling_model = _get_sampling_model(peft_model, old_peft_model)
    sampling_model.eval()
    if old_peft_model is not None:
        old_peft_model.eval()
    if ref_model is not None:
        ref_model.eval()

    dev = torch.device(device) if isinstance(device, str) else device
    prompt_texts, prompt_ids, prompt_attention_mask = tokenize_prompts_batch(
        tokenizer, prompts, dev,
    )
    if generation_backend == GenerationBackend.VLLM and old_peft_model is not None:
        raise NotImplementedError(
            "vLLM sampling with old_peft_model is not supported yet."
        )

    if generation_backend == GenerationBackend.HF:
        _set_generation_seed(seed)

    rollout = generate_rollout_batch(
        sampling_model,
        tokenizer,
        prompt_ids,
        prompt_attention_mask,
        backend=generation_backend,
        num_samples=G,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        vllm_config=vllm_config,
        adapter_path=adapter_path,
        model_id=model_id,
    )
    if rollout.num_prompts * rollout.num_samples != len(rollout.texts):
        raise RuntimeError("Rollout layout does not match num_prompts * num_samples.")

    grad_rows = []
    geometry_rows = [] if geometry_feature_mode != GeometryFeatureMode.NONE else None
    debug_rows = []

    for bi in range(bsz):
        start_idx = bi * G
        end_idx = (bi + 1) * G

        # Extract data for this single prompt
        single_prompt_text = prompt_texts[bi]
        single_prompt_ids = prompt_ids[bi : bi + 1]
        single_prompt_mask = prompt_attention_mask[bi : bi + 1]
        
        single_texts = rollout.texts[start_idx:end_idx]
        single_token_ids = rollout.token_ids[start_idx:end_idx]
        single_response_mask = rollout.response_mask[start_idx:end_idx]

        # Evaluate rewards for this single prompt
        slice_trl = rollout_to_completions(
            RolloutBatch(
                texts=single_texts,
                token_ids=single_token_ids,
                response_mask=single_response_mask,
                num_prompts=1,
                num_samples=G,
            )
        )
        total_rewards, reward_breakdown = _evaluate_rewards(reward_funcs_batch[bi], slice_trl, dev)

        # Forward Pass (G sequences only - extremely safe for memory)
        per_token_logps = _compute_per_token_logps(
            peft_model,
            single_prompt_ids,
            single_prompt_mask,
            single_token_ids,
            single_response_mask,
        )

        if old_peft_model is None:
            old_per_token_logps = per_token_logps.detach()
        else:
            old_per_token_logps = _compute_old_per_token_logps(
                peft_model, old_peft_model, single_prompt_ids, single_prompt_mask, single_token_ids, single_response_mask
            )

        ref_per_token_logps = None
        if beta != 0.0:
            ref_per_token_logps = _compute_ref_per_token_logps(
                peft_model, ref_model, single_prompt_ids, single_prompt_mask, single_token_ids, single_response_mask
            )

        token_mask, token_counts, sequence_log_probs, mean_sequence_log_probs = _sequence_log_stats(
            per_token_logps, single_response_mask
        )

        # Calculate Objective (Returns a single scalar loss for this prompt)
        if objective_mode == GradientObjective.GRPO_TRAIN:
            objective, advantages, per_token_kl = _compute_grpo_policy_loss(
                total_rewards,
                per_token_logps,
                old_per_token_logps,
                single_response_mask,
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

        if geometry_feature_mode == GeometryFeatureMode.POLICY_SCORE:
            peft_model.zero_grad()
            geometry_scalar = sequence_log_probs.mean()
            # Retain the graph here!
            geometry_vector = _grad_vector_from_scalar(peft_model, geometry_scalar, retain_graph=True)
            geometry_vector = _sanitize_grad_vector(geometry_vector, context=f"batch_idx_{bi}_policy_score")
            geometry_rows.append(geometry_vector)

        peft_model.zero_grad()
        grad_vector = _grad_vector_from_scalar(peft_model, objective, retain_graph=False)
        grad_vector = _sanitize_grad_vector(grad_vector, context=f"batch_idx_{bi}_{objective_mode}")
        grad_rows.append(grad_vector)

        # Append Debug Info
        debug_rows.append({
            "prompt_text": single_prompt_text,
            "responses": single_texts,
            "reward_breakdown": reward_breakdown,
            "total_rewards": total_rewards.detach().cpu().tolist(),
            "advantages": advantages.detach().cpu().tolist(),
            "log_probs": sequence_log_probs.detach().float().cpu().tolist(),
            "sequence_log_probs": sequence_log_probs.detach().float().cpu().tolist(),
            "mean_sequence_log_probs": float(mean_sequence_log_probs.mean().detach().cpu()),
            "response_lengths": single_response_mask.sum(dim=1).detach().cpu().tolist(),
            "policy_loss": float(objective.detach().float().cpu()),
            "mean_kl": float(((per_token_kl * token_mask).sum(dim=1) / token_counts).mean().detach().cpu()),
            "epsilon": epsilon,
            "beta": beta,
            "seed": seed,
            "generation_backend": generation_backend,
            "objective_mode": objective_mode,
            "objective_name": objective_name,
            "geometry_feature_mode": geometry_feature_mode,
        })
        if geometry_feature_mode != GeometryFeatureMode.NONE:
             debug_rows[-1]["geometry_feature_norm"] = float(geometry_vector.norm().item())

        import gc
        
        del objective, per_token_logps, sequence_log_probs, token_mask, token_counts, per_token_kl
        
        if 'old_per_token_logps' in locals(): del old_per_token_logps
        if 'ref_per_token_logps' in locals() and ref_per_token_logps is not None: del ref_per_token_logps
        if 'advantages' in locals(): del advantages
        if 'total_rewards' in locals(): del total_rewards
        if 'slice_trl' in locals(): del slice_trl
        
        if geometry_feature_mode == GeometryFeatureMode.POLICY_SCORE:
            del geometry_scalar
            
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "grad": grad_rows,
        "geometry_feature": geometry_rows,
        "debug": debug_rows,
    }


def compute_policy_gradient_bundle(
    peft_model,
    tokenizer,
    prompt,
    reward_funcs,
    *,
    G=4,
    device="cpu",
    enable_vllm=False,
    generation_backend=None,
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
    vllm_config=None,
    adapter_path=None,
    model_id=None,
):
    objective_mode = GradientObjective.parse(objective_mode)
    geometry_feature_mode = GeometryFeatureMode.parse(geometry_feature_mode)
    generation_backend = _resolve_generation_backend(enable_vllm, generation_backend)
    if vllm_config is None:
        vllm_config = VLLMConfig()

    peft_model.eval()
    peft_model.zero_grad()
    sampling_model = _get_sampling_model(peft_model, old_peft_model)
    sampling_model.eval()
    if old_peft_model is not None:
        old_peft_model.eval()
    if ref_model is not None:
        ref_model.eval()

    prompt_text, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)
    if generation_backend == GenerationBackend.VLLM and old_peft_model is not None:
        raise NotImplementedError(
            "vLLM sampling with old_peft_model is not supported yet."
        )

    if generation_backend == GenerationBackend.HF:
        _set_generation_seed(seed)
    rollout = generate_rollout_batch(
        sampling_model,
        tokenizer,
        prompt_ids,
        prompt_attention_mask,
        backend=generation_backend,
        num_samples=G,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        vllm_config=vllm_config,
        adapter_path=adapter_path,
        model_id=model_id,
    )
    completions_trl = rollout_to_completions(rollout)
    total_rewards, reward_breakdown = _evaluate_rewards(reward_funcs, completions_trl, device)

    response_ids = rollout.token_ids
    response_mask = rollout.response_mask
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
        "responses": rollout.texts,
        "reward_breakdown": reward_breakdown,
        "total_rewards": total_rewards.detach().cpu().tolist(),
        "advantages": advantages.detach().cpu().tolist(),
        "log_probs": sequence_log_probs.detach().float().cpu().tolist(),
        "sequence_log_probs": sequence_log_probs.detach().float().cpu().tolist(),
        "mean_sequence_log_probs": mean_sequence_log_probs.detach().float().cpu().tolist(),
        "response_lengths": response_mask.sum(dim=1).detach().cpu().tolist(),
        "policy_loss": float(objective.detach().float().cpu()),
        "mean_kl": float(((per_token_kl * token_mask).sum(dim=1) / token_counts).mean().detach().cpu()),
        "epsilon": epsilon,
        "beta": beta,
        "seed": seed,
        "generation_backend": generation_backend,
        "objective_mode": objective_mode,
        "objective_name": objective_name,
        "geometry_feature_mode": geometry_feature_mode,
    }
    if geometry_feature is not None:
        debug_info["geometry_feature_norm"] = float(geometry_feature.norm().item())

    del per_token_logps, old_per_token_logps, objective
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
        with torch.inference_mode():
            return _compute_per_token_logps(
                peft_model,
                prompt_ids,
                prompt_attention_mask,
                response_ids,
                response_mask,
            ).detach()

    with torch.inference_mode():
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
        with torch.inference_mode():
            return _compute_per_token_logps(
                ref_model,
                prompt_ids,
                prompt_attention_mask,
                response_ids,
                response_mask,
            )

    ref_adapter_name = "ref" if "ref" in peft_model.peft_config else None
    if ref_adapter_name is not None:
        with torch.inference_mode(), _use_adapter(peft_model, ref_adapter_name):
            return _compute_per_token_logps(
                peft_model,
                prompt_ids,
                prompt_attention_mask,
                response_ids,
                response_mask,
            )

    with torch.inference_mode(), peft_model.disable_adapter():
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
    return grad_vector


def compute_rlvr_gradient(
    peft_model, tokenizer, prompt, reward_funcs,
    G=4, device="cpu", enable_vllm=False,
    generation_backend=None,
    max_new_tokens=256, temperature=0.7, top_p=0.9,
    return_debug=False,
    seed=None,
    epsilon=0.2,
    beta=0.0,
    old_peft_model=None,
    ref_model=None,
    advantage_eps=1e-4,
    vllm_config=None,
    adapter_path=None,
    model_id=None,
):
    result = compute_policy_gradient_bundle(
        peft_model,
        tokenizer,
        prompt,
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
        old_peft_model=old_peft_model,
        ref_model=ref_model,
        advantage_eps=advantage_eps,
        objective_mode=GradientObjective.GRPO_TRAIN,
        geometry_feature_mode=GeometryFeatureMode.NONE,
        vllm_config=vllm_config,
        adapter_path=adapter_path,
        model_id=model_id,
    )
    if return_debug:
        return result["grad"], result["debug"]
    return result["grad"]
