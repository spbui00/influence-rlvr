"""Matrix-free policy Fisher-vector product for LLM/LoRA scale (the "true" Fisher).

This is the LLM-scale FVP the toy's `policy_fisher_fvp_from_grad_cache` docstring
points to: instead of caching per-completion sequence-score gradients (a low-rank
*sampled* Fisher), we use the **analytic per-token Gauss-Newton/Fisher**

    F = (1/N) Σ_seq Σ_t  J_tᵀ M_t J_t ,    M_t = diag(p_t) − p_t p_tᵀ,

where p_t = softmax(logits_t) is the model's own next-token distribution and
J_t = ∂logits_t/∂θ. Taking the over-vocabulary expectation in closed form (via
M_t) makes this full-rank and low-variance — no sampling of completions for the
Fisher itself; the completion only supplies realistic teacher-forced prefixes.

`F·v` is computed **matrix-free** with reverse-mode autodiff only (a JVP built
from a double-backward), so it works through SDPA/eager attention without
forward-mode AD:

    JᵀM J v  =  grad_θ( logits, grad_outputs = M ⊙ Jv ),
    Jv       =  grad_u( < grad_θ(logits, grad_outputs=u), v > ),   (u a dummy)

Each `F·v` is one forward + a few backwards over the (small) Fisher batch. Keep
the Fisher batch small and truncated; CG calls this `cg_iters` times per target.

NOTE: disable gradient checkpointing on the model before use — double-backward
does not compose with activation checkpointing.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@contextmanager
def _math_attention():
    """Force the math SDPA backend: flash/mem-efficient attention has no
    backward-of-backward, which the double-backward FVP needs. Math attention is
    plain differentiable ops, so second-order grads work. No-op on older torch."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        yield
        return
    with sdpa_kernel(SDPBackend.MATH):
        yield


@dataclass
class FisherRow:
    """One teacher-forced sequence: prompt + a single completion, on the model device."""
    prompt_ids: torch.Tensor      # (1, Lp)
    prompt_mask: torch.Tensor     # (1, Lp)
    response_ids: torch.Tensor    # (1, Lr)
    response_mask: torch.Tensor   # (1, Lr)


def _response_logits(model, row: FisherRow) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits at the response positions (graph attached to θ), and their mask."""
    input_ids = torch.cat([row.prompt_ids, row.response_ids], dim=1)
    attention_mask = torch.cat([row.prompt_mask, row.response_mask], dim=1)
    with _math_attention():   # double-backward needs math (not flash) attention
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    lp = int(row.prompt_ids.shape[1])
    lr = int(row.response_ids.shape[1])
    # Position i predicts token i+1, so completion logits start at lp-1.
    resp_logits = logits[:, lp - 1 : lp - 1 + lr, :]      # (1, Lr, V)
    return resp_logits, row.response_mask


def build_policy_fisher_fvp(
    model,
    fisher_rows: Sequence[FisherRow],
    *,
    normalize: int | None = None,
    spectral_normalize: bool = False,
    n_power_iters: int = 15,
    power_seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return `fvp(v) -> F·v` for the analytic per-token policy Fisher.

    `v` and the returned vector are flat D-vectors over the model's trainable
    (LoRA) params, in `model.parameters()` order — matching the convention of
    `gradients._grad_vector_from_scalar`, so they are directly comparable to the
    `g_test`/`g_train` produced elsewhere.

    With `spectral_normalize=True`, F is rescaled by its largest eigenvalue
    (estimated by `n_power_iters` power iterations) so its spectrum lies in [0, 1].
    This makes the CG damping λ a *scale-free* knob: (F/λ_max + λI) with λ∈(0,1)
    damps relative to the top curvature direction, instead of depending on the raw
    (batch/token-count-dependent) Fisher magnitude. The same λ then transfers
    across checkpoints/pool sizes.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    numels = [p.numel() for p in params]
    shapes = [p.shape for p in params]
    n_norm = len(fisher_rows) if normalize is None else int(normalize)
    n_norm = max(n_norm, 1)

    def _split(v: torch.Tensor) -> list[torch.Tensor]:
        out, off = [], 0
        for n, s in zip(numels, shapes):
            out.append(v[off : off + n].view(s))
            off += n
        return out

    def _flatten(tensors) -> torch.Tensor:
        parts = []
        for t, p in zip(tensors, params):
            parts.append((t if t is not None else torch.zeros_like(p)).reshape(-1).float())
        return torch.cat(parts)

    def fvp(v: torch.Tensor) -> torch.Tensor:
        v = v.to(device=params[0].device, dtype=torch.float32)
        v_parts = _split(v)
        total: torch.Tensor | None = None

        for row in fisher_rows:
            resp_logits, resp_mask = _response_logits(model, row)        # (1, Lr, V)
            p = torch.softmax(resp_logits.detach().float(), dim=-1)      # (1, Lr, V)

            # Jv (JVP) via double-backward, reverse-mode only.
            u = torch.zeros_like(resp_logits, requires_grad=True)
            jt_u = torch.autograd.grad(
                resp_logits, params, grad_outputs=u,
                create_graph=True, retain_graph=True, allow_unused=True,
            )
            terms = [(g * vp).sum() for g, vp in zip(jt_u, v_parts) if g is not None]
            if not terms:
                raise RuntimeError("Fisher FVP: no trainable params connected to logits.")
            inner = torch.stack(terms).sum()
            jv = torch.autograd.grad(inner, u, retain_graph=True)[0].float()  # (1, Lr, V)

            # M·Jv = p⊙Jv − p·(pᵀJv), masked to valid response positions.
            mask = resp_mask.unsqueeze(-1).float()                       # (1, Lr, 1)
            m_jv = (p * jv - p * (p * jv).sum(dim=-1, keepdim=True)) * mask

            # Jᵀ(M·Jv) — accumulate this sequence's Fisher·v.
            fv_parts = torch.autograd.grad(
                resp_logits, params, grad_outputs=m_jv.to(resp_logits.dtype),
                retain_graph=False, allow_unused=True,
            )
            fv = _flatten(fv_parts)
            total = fv if total is None else total + fv

        if total is None:
            return torch.zeros(sum(numels), dtype=torch.float32, device=params[0].device)
        return total / n_norm

    if not spectral_normalize:
        return fvp

    # Estimate the top eigenvalue of F by power iteration, then rescale so the
    # spectrum lies in [0, 1]. Each iteration is one (expensive) FVP, but a handful
    # is cheap next to a full CG solve and only happens once at build time.
    D = int(sum(numels))
    device = params[0].device
    gen = torch.Generator(device=device).manual_seed(power_seed)
    v = torch.randn(D, generator=gen, device=device, dtype=torch.float32)
    v = v / v.norm().clamp(min=1e-20)
    lam_max = torch.tensor(1.0, device=device)
    for _ in range(max(1, n_power_iters)):
        Fv = fvp(v)
        nrm = Fv.norm()
        if nrm < 1e-20:
            break
        v = Fv / nrm
    Fv = fvp(v)
    lam_max = torch.dot(v, Fv).clamp(min=1e-20)  # Rayleigh quotient (||v||=1)
    scale = float(lam_max.item())
    print(f"  Fisher spectral-normalize: λ_max≈{scale:.4g} (eigenvalues rescaled to [0,1])")

    def fvp_normalized(vec: torch.Tensor) -> torch.Tensor:
        return fvp(vec) / scale

    return fvp_normalized
