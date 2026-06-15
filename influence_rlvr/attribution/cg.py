from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from .base import BaseInfluenceMethod


class CGInfluence(BaseInfluenceMethod):
    """
    Conjugate-Gradient influence with Fisher-vector products (PBRF-style).

        F = E_{z~B, y~π_θ(·|z)}[∇log π(y|z) ∇log π(y|z)^T]   (policy Fisher)
        h = (F + λI)^{-1} g_test                              (CG with FVPs)
        IF(z_train) = -g_train^T h                            (one dot per train ex)

    Sign convention follows the PBRF derivation: IF(z_train) < 0 indicates a
    helpful example (training on it pushes θ toward higher test reward).
    Callers that want the "helpful → positive score" LDS convention should
    negate the output.

    F is never materialized as a D×D matrix. The caller passes `fvp_fn(p) -> F·p`,
    so the same class works at any scale — toy models (cached per-(z, y) gradients,
    see `policy_fisher_fvp_from_grad_cache`) or LLMs (two-pass JVP+VJP through the
    network on a fixed Fisher batch).

    Required keys:
      test_info["grad"]   — flat D-vector test-side gradient.
      train_info["grad"]  — flat D-vector training-side gradient.

    h is cached per `id(test_info)` so `compute_all_scores` and repeated
    `compute_score` calls on the same test_info reuse the single CG solve.
    Caveat: `id()` is recycled after garbage collection — at LLM scale, switch
    to a content-addressed cache keyed by hash(test grad bytes) or an explicit
    test_id supplied by the caller.
    """

    def __init__(
        self,
        fvp_fn: Callable[[torch.Tensor], torch.Tensor],
        lambda_damp: float = 1.0,
        cg_iters: int = 50,
        cg_tol: float = 1e-6,
        verbose: bool = False,
    ):
        self.fvp_fn = fvp_fn
        self.lambda_damp = float(lambda_damp)
        self.cg_iters = int(cg_iters)
        self.cg_tol = float(cg_tol)
        self.verbose = bool(verbose)
        self._h_cache: dict[int, tuple[torch.Tensor, dict]] = {}

    def solve(self, g_test: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """CG solve of (F + λI) h = g_test. Returns (h, diagnostics)."""
        g = g_test.detach().to(dtype=torch.float32)
        h = torch.zeros_like(g)
        r = g.clone()
        p = r.clone()
        rho = torch.dot(r, r)
        g_norm = g.norm().clamp(min=1e-12)

        residuals: list[float] = []
        n_iters = 0
        for k in range(self.cg_iters):
            n_iters = k + 1
            Ap = self.fvp_fn(p) + self.lambda_damp * p
            pAp = torch.dot(p, Ap).clamp(min=1e-20)
            alpha = rho / pAp
            h = h + alpha * p
            r = r - alpha * Ap
            resid = (r.norm() / g_norm).item()
            residuals.append(resid)
            if self.verbose:
                print(f"      CG iter {k+1}: ||r||/||g_test||={resid:.4e}")
            if resid < self.cg_tol:
                break
            rho_new = torch.dot(r, r)
            beta = rho_new / rho.clamp(min=1e-20)
            p = r + beta * p
            rho = rho_new

        info = {
            "n_iters": n_iters,
            "final_residual": residuals[-1] if residuals else None,
            "g_test_dot_h": float(torch.dot(g, h).item()),  # SPD sanity check: should be >= 0
            "residual_trace": residuals,
        }
        return h, info

    def _h_for(self, test_info: dict) -> torch.Tensor:
        key = id(test_info)
        if key not in self._h_cache:
            self._h_cache[key] = self.solve(test_info["grad"])
        return self._h_cache[key][0]

    def cg_info_for(self, test_info: dict) -> dict | None:
        """Return CG diagnostics for the last solve on this test_info (or None)."""
        entry = self._h_cache.get(id(test_info))
        return entry[1] if entry is not None else None

    def compute_score(self, test_info: dict, train_info: dict) -> float:
        h = self._h_for(test_info)
        g_train = train_info["grad"].detach().to(dtype=torch.float32, device=h.device)
        return float(torch.dot(g_train, h).item())

    def compute_all_scores(self, test_info: dict, train_infos: Sequence[dict]) -> np.ndarray:
        h = self._h_for(test_info)
        out = np.zeros(len(train_infos), dtype=np.float32)
        for i, info in enumerate(train_infos):
            g_train = info["grad"].detach().to(dtype=torch.float32, device=h.device)
            out[i] = float(torch.dot(g_train, h).item())
        return out


def policy_fisher_fvp_from_grad_cache(
    grad_cache: Sequence[torch.Tensor],
    prob_cache: Sequence[torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    FVP factory for small-D regimes (e.g. toy models).

    Caches per-sample log-prob gradients once and reduces each FVP to two
    matvecs per training example per CG iteration:

        F p = (1/N) Σ_z G_z^T · (π_z * (G_z p))

    where G_z is an (n_y, D) stack of ∇log π(y|z) and π_z is (n_y,).

    For LLM-scale use, the equivalent recipe avoids the cache and computes
    `F p` with one JVP (logits, tangent=p) → `M·u` token-wise → one VJP.
    """
    if len(grad_cache) != len(prob_cache):
        raise ValueError("grad_cache and prob_cache must have the same length.")
    n = len(grad_cache)

    def fvp(p_vec: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(p_vec)
        for G_z, probs in zip(grad_cache, prob_cache):
            G_dev = G_z.to(device=p_vec.device, dtype=p_vec.dtype)
            pi_dev = probs.to(device=p_vec.device, dtype=p_vec.dtype)
            Gp = G_dev @ p_vec
            total = total + G_dev.T @ (pi_dev * Gp)
        return total / max(n, 1)

    return fvp
