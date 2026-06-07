"""Adam diagonal preconditioner for TracIn-style influence.

Adam updates each coordinate by  Δθ_d ∝ −lr · m̂_d / (√v̂_d + ε), so the *direction*
a training gradient actually moves the parameters is the raw gradient rescaled by
the per-coordinate diagonal

    P_d = 1 / (√v̂_d + ε),     v̂ = v / (1 − β₂^t)   (Adam's bias-corrected 2nd moment).

For data attribution under Adam, the faithful first-order influence of a train
example on a target is therefore plain TracIn with one side preconditioned by P:

    IF(train → target) ≈ g_targetᵀ (P ⊙ g_train) = (P ⊙ g_target)ᵀ g_train,

since P is diagonal (the elementwise product is symmetric, so it may be folded
into either gradient). The global learning rate is a positive scalar that cancels
under ranking and is dropped.

This module builds P as a flat D-vector aligned to a model's trainable-parameter
order — the SAME order `gradients._lora_trainable_params` flattens g_train/g_test
(`named_parameters()` filtered to `requires_grad`) — from a saved optimizer state
dict (`optimizer.pt`). The alignment is positional and validated by parameter
shape, which is exact for the LoRA setup here (all trainable adapters are weights,
so HF's no-weight-decay group is empty and the optimizer order equals the
named-parameter order); a decay/no-decay regrouping is the fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


def _is_no_decay_name(name: str) -> bool:
    """Mirror HF Trainer's optimizer grouping: biases and norm weights skip weight
    decay and go into a SECOND param group, which shifts the optimizer-state
    indices relative to ``named_parameters()``. LoRA adapters (``…lora_A/B.…weight``
    with ``bias="none"``) are all decay params, so this project's no-decay group is
    empty and this branch is only the fallback alignment."""
    lname = name.lower()
    return lname.endswith(".bias") or "norm" in lname


def _ordered_states(optimizer_state_dict: dict) -> list[dict]:
    """Per-param Adam state in optimizer index order (torch keys it by the integer
    position of each param across the concatenated param groups)."""
    state = optimizer_state_dict.get("state", {})
    return [state[i] for i in sorted(state.keys(), key=int)]


def adam_diagonal_preconditioner(
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
    optimizer_state_dict: dict,
    *,
    device,
    dtype: torch.dtype = torch.float32,
    eps_override: float | None = None,
    eps_fallback: float = 1e-8,
    beta2_fallback: float = 0.999,
) -> torch.Tensor:
    """Flat D-vector ``P_d = 1/(√v̂_d + ε)`` in ``named_params`` order.

    ``named_params`` must be the trainable params in the SAME order the influence
    gradients are flattened (``named_parameters()`` filtered to ``requires_grad``).
    β₂/ε are read from the optimizer's first param group (with fallbacks), and v is
    bias-corrected per param by its own ``step`` count.

    ``eps_override`` (>0) replaces Adam's optimization ε with a larger floor on the
    denominator. Adam's ε≈1e-8 is tiny, so coordinates with a near-zero 2nd moment
    (dormant during training) get a huge P≈1/ε and can dominate the influence dot;
    a larger ε caps that dynamic range, trading exact-Adam faithfulness for ranking
    stability. Default (None) keeps the faithful optimizer ε.
    """
    params = [p for _, p in named_params]
    states = _ordered_states(optimizer_state_dict)
    if len(states) != len(params):
        raise ValueError(
            f"optimizer state has {len(states)} params but the model has "
            f"{len(params)} trainable params; cannot build an Adam preconditioner."
        )
    if states and "exp_avg_sq" not in states[0]:
        raise ValueError(
            "optimizer state has no 'exp_avg_sq' — tracin-adam requires an "
            "Adam/AdamW optimizer (HF Trainer's default). Use --if-method dot."
        )

    groups = optimizer_state_dict.get("param_groups", []) or [{}]
    g0 = groups[0]
    beta2 = float(g0.get("betas", (0.9, beta2_fallback))[1])
    eps = float(g0.get("eps", eps_fallback))
    if eps_override is not None and eps_override > 0:
        eps = float(eps_override)

    def shapes_match(order: Sequence[torch.nn.Parameter]) -> bool:
        return all(
            tuple(s["exp_avg_sq"].shape) == tuple(p.shape)
            for s, p in zip(states, order)
        )

    # Positional alignment is exact when the no-decay group is empty (LoRA case).
    if shapes_match(params):
        state_for = dict(zip((id(p) for p in params), states))
    else:
        # Fallback: HF concatenates decay params then no-decay params, each in
        # named_parameters() order. Reconstruct that order and map by identity.
        regrouped = (
            [p for n, p in named_params if not _is_no_decay_name(n)]
            + [p for n, p in named_params if _is_no_decay_name(n)]
        )
        if not shapes_match(regrouped):
            raise ValueError(
                "Could not align optimizer state to model parameters by shape; "
                "refusing to build a possibly-misaligned Adam preconditioner. "
                "Use --if-method dot instead."
            )
        state_for = {id(p): s for s, p in zip(states, regrouped)}

    parts: list[torch.Tensor] = []
    for p in params:
        s = state_for[id(p)]
        v = s["exp_avg_sq"].to(device=device, dtype=torch.float32)
        step = s.get("step", None)
        if step is not None:
            t = float(step.item()) if torch.is_tensor(step) else float(step)
            bias_correction2 = 1.0 - beta2 ** t if t > 0 else 1.0
        else:
            bias_correction2 = 1.0
        v_hat = v / bias_correction2
        # By the prune step every LoRA coordinate has a nonzero 2nd moment, so the
        # +eps only guards against division-by-zero; sqrt(v_hat) dominates it.
        precond = 1.0 / (v_hat.sqrt() + eps)
        parts.append(precond.reshape(-1).to(dtype))
    return torch.cat(parts)


def load_adam_preconditioner_from_checkpoint(
    model,
    checkpoint_dir,
    *,
    device,
    dtype: torch.dtype = torch.float32,
    eps: float | None = None,
) -> torch.Tensor | None:
    """Load ``optimizer.pt`` from ``checkpoint_dir`` and build P, or return None.

    Returns None (no error) when the checkpoint has no optimizer state, so callers
    can fall back to an un-preconditioned dot product instead of crashing a run.
    ``eps`` (>0) overrides Adam's optimization ε to cap P's dynamic range; see
    ``adam_diagonal_preconditioner``.
    """
    opt_path = Path(checkpoint_dir) / "optimizer.pt"
    if not opt_path.is_file():
        return None
    # Trusted local checkpoint; optimizer state needs weights_only=False to load
    # its param_groups / step scalars on torch>=2.6.
    sd = torch.load(opt_path, map_location="cpu", weights_only=False)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    return adam_diagonal_preconditioner(
        named, sd, device=device, dtype=dtype, eps_override=eps
    )
