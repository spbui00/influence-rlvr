import unittest

import numpy as np
import torch

from influence_rlvr.attribution.cg import (
    CGInfluence,
    policy_fisher_fvp_from_grad_cache,
)


def _build_F_and_caches(D=12, n_y=4, N=20, seed=0):
    """Mimic the policy-Fisher setup that the toy script feeds to CGInfluence."""
    g = torch.Generator().manual_seed(seed)
    grad_cache = [torch.randn(n_y, D, generator=g) for _ in range(N)]
    prob_cache = [torch.softmax(torch.randn(n_y, generator=g), dim=0) for _ in range(N)]
    # F = (1/N) Σ_z G_z^T diag(π_z) G_z — the operator the FVP factory computes.
    F = sum(Gz.T @ torch.diag(pz) @ Gz for Gz, pz in zip(grad_cache, prob_cache)) / N
    return grad_cache, prob_cache, F


class CGInfluenceTests(unittest.TestCase):
    def test_cg_matches_explicit_inverse(self):
        """CG h must match (F + λI)^{-1} g_test to better than cg_tol."""
        D = 12
        grad_cache, prob_cache, F = _build_F_and_caches(D=D)
        lam = 0.1
        tol = 1e-8
        g_test = torch.randn(D, generator=torch.Generator().manual_seed(123))

        h_ref = torch.linalg.solve(F + lam * torch.eye(D), g_test)

        fvp = policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache)
        cg = CGInfluence(fvp_fn=fvp, lambda_damp=lam, cg_iters=200, cg_tol=tol)
        h_cg, info = cg.solve(g_test)

        rel_err = ((h_cg - h_ref).norm() / h_ref.norm()).item()
        self.assertLess(rel_err, max(tol * 100, 1e-6))
        self.assertTrue(info["converged"])
        self.assertGreaterEqual(info["g_test_dot_h"], 0.0)  # SPD invariant

    def test_zero_g_test_returns_zero_h(self):
        """||g_test|| = 0 should short-circuit to h = 0 instead of dividing by zero."""
        D = 6
        grad_cache, prob_cache, _ = _build_F_and_caches(D=D, n_y=3, N=4, seed=7)
        fvp = policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache)
        cg = CGInfluence(fvp_fn=fvp, lambda_damp=0.1)
        h, info = cg.solve(torch.zeros(D))
        self.assertTrue(torch.equal(h, torch.zeros(D)))
        self.assertEqual(info["status"], "trivial")
        self.assertTrue(info["converged"])

    def test_compute_all_scores_matches_compute_score_loop(self):
        """Vectorized compute_all_scores must agree with the per-call path."""
        D = 8
        grad_cache, prob_cache, _ = _build_F_and_caches(D=D, n_y=3, N=6, seed=11)
        fvp = policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache)
        cg = CGInfluence(fvp_fn=fvp, lambda_damp=0.5, cg_iters=100, cg_tol=1e-9)

        g_test = torch.randn(D, generator=torch.Generator().manual_seed(2))
        test_info = {"grad": g_test}
        train_infos = [
            {"grad": torch.randn(D, generator=torch.Generator().manual_seed(100 + i))}
            for i in range(5)
        ]

        vec = cg.compute_all_scores(test_info, train_infos)
        loop = np.array([cg.compute_score(test_info, ti) for ti in train_infos], dtype=np.float32)
        np.testing.assert_allclose(vec, loop, rtol=1e-5, atol=1e-6)

    def test_h_cache_keyed_by_content_not_identity(self):
        """Two dicts with identical g_test bytes should hit the same cache entry."""
        D = 4
        grad_cache, prob_cache, _ = _build_F_and_caches(D=D, n_y=2, N=3, seed=42)
        fvp = policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache)
        cg = CGInfluence(fvp_fn=fvp, lambda_damp=0.1)

        g = torch.randn(D, generator=torch.Generator().manual_seed(99))
        ti_1 = {"grad": g.clone()}
        cg._h_for(ti_1)
        ti_2 = {"grad": g.clone()}  # different dict, same bytes
        # Should not trigger a second CG solve.
        cache_size_before = len(cg._h_cache)
        cg._h_for(ti_2)
        self.assertEqual(len(cg._h_cache), cache_size_before)


if __name__ == "__main__":
    unittest.main()
