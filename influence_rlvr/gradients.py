import warnings
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

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


def _compute_per_token_logps(model, prompt_ids, prompt_attention_mask, response_ids, response_mask):
    n_p = prompt_ids.shape[0]
    n_r = response_ids.shape[0]
    if n_p == 1 and n_r > 1:
        prompt_batch = prompt_ids.repeat(n_r, 1)
        prompt_mask_batch = prompt_attention_mask.repeat(n_r, 1)
    elif n_p == n_r:
        prompt_batch = prompt_ids
        prompt_mask_batch = prompt_attention_mask
    else:
        raise ValueError(
            f"prompt batch {n_p} incompatible with response batch {n_r} "
            "(expected 1 or equal batch sizes)."
        )
    input_ids = torch.cat([prompt_batch, response_ids], dim=1)
    attention_mask = torch.cat([prompt_mask_batch, response_mask], dim=1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
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

    total_reward_rows = []
    reward_breakdown_batch = []
    for bi in range(bsz):
        slice_trl = rollout_to_completions(
            RolloutBatch(
                texts=rollout.texts[bi * G : (bi + 1) * G],
                token_ids=rollout.token_ids[bi * G : (bi + 1) * G],
                response_mask=rollout.response_mask[bi * G : (bi + 1) * G],
                num_prompts=1,
                num_samples=G,
            )
        )
        tr, rb = _evaluate_rewards(reward_funcs_batch[bi], slice_trl, dev)
        total_reward_rows.append(tr)
        reward_breakdown_batch.append(rb)
    total_rewards = torch.stack(total_reward_rows, dim=0)

    rl = rollout.token_ids.shape[1]
    response_ids = rollout.token_ids.view(bsz, G, rl)
    response_mask = rollout.response_mask.view(bsz, G, rl)

    bg = bsz * G
    pe = prompt_ids.unsqueeze(1).expand(bsz, G, prompt_ids.shape[1]).reshape(bg, prompt_ids.shape[1])
    pme = prompt_attention_mask.unsqueeze(1).expand(bsz, G, prompt_attention_mask.shape[1]).reshape(
        bg, prompt_attention_mask.shape[1],
    )
    rf = response_ids.reshape(bg, rl)
    rm = response_mask.reshape(bg, rl)

    per_token_logps_flat = _compute_per_token_logps(
        peft_model,
        pe,
        pme,
        rf,
        rm,
    )
    tlen = per_token_logps_flat.shape[1]
    per_token_logps = per_token_logps_flat.view(bsz, G, tlen)

    if old_peft_model is None:
        old_per_token_logps = per_token_logps.detach()
    else:
        old_flat = _compute_old_per_token_logps(
            peft_model,
            old_peft_model,
            pe,
            pme,
            rf,
            rm,
        )
        old_per_token_logps = old_flat.view(bsz, G, tlen)

    ref_per_token_logps = None
    if beta != 0.0:
        ref_flat = _compute_ref_per_token_logps(
            peft_model,
            ref_model,
            pe,
            pme,
            rf,
            rm,
        )
        ref_per_token_logps = ref_flat.view(bsz, G, tlen)

    token_mask, token_counts, sequence_log_probs, mean_sequence_log_probs = (
        _sequence_log_stats_batched(per_token_logps, response_mask)
    )

    if objective_mode == GradientObjective.GRPO_TRAIN:
        loss_vec, advantages, per_token_kl = _compute_grpo_policy_loss_per_prompt(
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
        loss_vec = _compute_expected_reward_policy_loss_per_prompt(
            total_rewards,
            sequence_log_probs,
        )
        advantages = total_rewards.detach()
        per_token_kl = torch.zeros_like(per_token_logps)
        objective_name = "expected_reward_pg_loss"
    else:
        raise ValueError(f"Unsupported objective_mode={objective_mode!r}.")

    names = _lora_trainable_param_names_tuple(peft_model)
    if not names:
        raise RuntimeError("No trainable parameters found for per-sample gradients.")

    grad_rows = _per_sample_gradients_batch(
        peft_model,
        objective_mode,
        names,
        prompt_ids,
        prompt_attention_mask,
        response_ids,
        response_mask,
        total_rewards,
        old_per_token_logps,
        ref_per_token_logps,
        epsilon=epsilon,
        beta=beta,
        advantage_eps=advantage_eps,
    )
    grad_rows = [
        _sanitize_grad_vector(
            g,
            context=f"compute_policy_gradient_bundle_batch[{objective_mode}][{j}]",
        )
        for j, g in enumerate(grad_rows)
    ]

    geometry_rows = None
    if geometry_feature_mode == GeometryFeatureMode.POLICY_SCORE:
        geometry_rows = _per_sample_geometry_gradients_batch(
            peft_model,
            names,
            prompt_ids,
            prompt_attention_mask,
            response_ids,
            response_mask,
        )
        geometry_rows = [
            _sanitize_grad_vector(
                g,
                context=f"compute_policy_gradient_bundle_batch[policy_score][{j}]",
            )
            for j, g in enumerate(geometry_rows)
        ]
    elif geometry_feature_mode != GeometryFeatureMode.NONE:
        raise ValueError(
            f"Unsupported geometry_feature_mode={geometry_feature_mode!r}."
        )

    debug_rows = []
    for bi in range(bsz):
        debug_rows.append({
            "prompt_text": prompt_texts[bi],
            "responses": rollout.texts[bi * G : (bi + 1) * G],
            "reward_breakdown": reward_breakdown_batch[bi],
            "total_rewards": total_rewards[bi].detach().cpu().tolist(),
            "advantages": advantages[bi].detach().cpu().tolist(),
            "log_probs": sequence_log_probs[bi].detach().float().cpu().tolist(),
            "sequence_log_probs": sequence_log_probs[bi].detach().float().cpu().tolist(),
            "mean_sequence_log_probs": float(mean_sequence_log_probs[bi].detach().cpu()),
            "response_lengths": response_mask[bi].sum(dim=1).detach().cpu().tolist(),
            "policy_loss": float(loss_vec[bi].detach().float().cpu()),
            "mean_kl": float(
                ((per_token_kl[bi] * token_mask[bi]).sum(dim=1) / token_counts[bi])
                .mean()
                .detach()
                .cpu()
            ),
            "epsilon": epsilon,
            "beta": beta,
            "seed": seed,
            "generation_backend": generation_backend,
            "objective_mode": objective_mode,
            "objective_name": objective_name,
            "geometry_feature_mode": geometry_feature_mode,
        })
        if geometry_rows is not None:
            debug_rows[-1]["geometry_feature_norm"] = float(geometry_rows[bi].norm().item())

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
