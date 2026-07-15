"""EK-FAC correctness tests.

Three layers of validation, strongest first:
  1. Hook fidelity — the per-layer δaᵀ blocks assembled from hooks must equal the
     autograd gradients exactly (validates capture, bias folding, and flat packing).
  2. Solver algebra — (F̂ + λI) applied to solve(g) must reproduce g exactly, since
     solve inverts the SAME fitted operator apply_fisher implements.
  3. Exact-Kronecker limit — with one fixed input, the true Fisher IS Kronecker
     (F = aaᵀ ⊗ S) and diagonal in the K-FAC eigenbasis, so the EK-FAC solve must
     match a dense (F + λI)⁻¹ solve built from raw per-sample gradients.
"""
import math
import unittest

import torch
from torch import nn

from influence_rlvr.attribution import EKFACInfluence, fit_ekfac


def _flat_grad(model, loss):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
    return torch.cat([g.reshape(-1) for g in grads])


def _make_mlp(seed=0, in_dim=6, hidden=5, out_dim=3, bias=True):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, hidden, bias=bias),
        nn.Tanh(),
        nn.Linear(hidden, out_dim, bias=bias),
    )


def _softmax_samples(model, n_inputs=7, in_dim=6, out_dim=3, seed=1):
    """(weight, closure) pairs for the exact softmax 'policy' Fisher of the MLP:
    one pair per (input z, class y), weight = softmax(y|z)/n_inputs."""
    torch.manual_seed(seed)
    zs = [torch.randn(1, in_dim) for _ in range(n_inputs)]
    samples = []
    for z in zs:
        with torch.no_grad():
            probs = torch.softmax(model(z), dim=-1).squeeze(0)
        for y in range(out_dim):
            samples.append((
                float(probs[y]) / n_inputs,
                lambda z=z, y=y: torch.log_softmax(model(z), dim=-1)[0, y],
            ))
    return samples


class EKFACHookFidelityTests(unittest.TestCase):
    def test_block_gradients_match_autograd(self):
        model = _make_mlp()
        samples = _softmax_samples(model)
        factors = fit_ekfac(model, samples)

        # λ's fitting identity: Σ_ij λ_ij == E_w[‖g_covered‖²] (rotations preserve
        # the norm), with the right-hand side computed independently via autograd —
        # any hook/bias-fold/packing bug breaks the equality.
        total_sq = 0.0
        w_total = 0.0
        for w, closure in samples:
            g = _flat_grad(model, closure())
            total_sq += w * float(g[factors.covered].pow(2).sum())
            w_total += w
        lam_sum = sum(float(blk.lam.sum()) for blk in factors.blocks if blk.lam is not None)
        self.assertTrue(
            math.isclose(lam_sum, total_sq / w_total, rel_tol=1e-6),
            f"Σλ={lam_sum} vs E‖g‖²={total_sq / w_total} — hook-captured gradients "
            f"disagree with autograd (capture/packing bug)",
        )
        # everything in this MLP is a Linear → the whole vector must be covered
        self.assertTrue(bool(factors.covered.all()))

    def test_solver_inverts_fitted_operator(self):
        model = _make_mlp(seed=3)
        samples = _softmax_samples(model, seed=4)
        ekfac = EKFACInfluence(fit_ekfac(model, samples), lambda_damp=0.05)
        torch.manual_seed(5)
        g = torch.randn(ekfac.factors.n_params)
        h = ekfac.solve(g)
        g_back = ekfac.apply_fisher(h) + 0.05 * h
        self.assertTrue(
            torch.allclose(g_back, g, atol=1e-4, rtol=1e-4),
            f"(F̂+λ)·solve(g) ≠ g; max err {float((g_back - g).abs().max())}",
        )


class EKFACExactKroneckerTests(unittest.TestCase):
    def test_matches_dense_solve_when_fisher_is_kronecker(self):
        # Single Linear, single fixed input → F = aaᵀ ⊗ S exactly, and diagonal in
        # the K-FAC eigenbasis, so EK-FAC must equal the dense damped inverse.
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(5, 4, bias=True))
        z = torch.randn(1, 5)
        with torch.no_grad():
            probs = torch.softmax(model(z), dim=-1).squeeze(0)
        samples = [
            (float(probs[y]), lambda y=y: torch.log_softmax(model(z), dim=-1)[0, y])
            for y in range(4)
        ]
        factors = fit_ekfac(model, samples)
        lam = 0.03
        ekfac = EKFACInfluence(factors, lambda_damp=lam)

        # dense Fisher from raw per-sample gradients
        G = torch.stack([_flat_grad(model, c()) for _, c in samples]).double()
        w = torch.tensor([wt for wt, _ in samples], dtype=torch.float64)
        F = (G * w.unsqueeze(1)).T @ G / w.sum()
        # fit_ekfac normalizes by Σw as well, so operators are on the same scale
        g = torch.randn(G.shape[1])
        h_dense = torch.linalg.solve(
            F + lam * torch.eye(F.shape[0], dtype=torch.float64), g.double()
        ).float()
        h_ekfac = ekfac.solve(g)
        self.assertTrue(
            torch.allclose(h_ekfac, h_dense, atol=1e-3, rtol=1e-3),
            f"exact-Kronecker case: max err {float((h_ekfac - h_dense).abs().max())}",
        )


if __name__ == "__main__":
    unittest.main()
