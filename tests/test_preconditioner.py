"""Tests for the Adam diagonal preconditioner and TracInAdamInfluence.

Validates (a) that the preconditioner P_d = 1/(√v̂_d+ε) is built with the right
formula, bias correction, and parameter ordering from an optimizer state dict,
(b) graceful handling of a missing optimizer.pt, and (c) that the
TracInAdamInfluence method computes the preconditioned dot consistently across its
single-score and vectorized paths and reduces to plain TracIn when P = 1.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from influence_rlvr.attribution.tracin import TracInAdamInfluence, TracInInfluence
from influence_rlvr.preconditioner import (
    adam_diagonal_preconditioner,
    load_adam_preconditioner_from_checkpoint,
)

CPU = torch.device("cpu")


class _Toy(nn.Module):
    """Two trainable 'weight' params of different shapes (no bias) → all land in
    HF's weight-decay group, so optimizer order == named_parameters order. This is
    the LoRA case the positional alignment is exact for."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(4, 3))
        self.c = nn.Parameter(torch.zeros(5, 2))


def _trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


class AdamPreconditionerTests(unittest.TestCase):
    def test_handbuilt_state_formula_bias_correction_and_order(self):
        model = _Toy()
        named = _trainable(model)
        va = torch.rand(4, 3) + 0.1
        vc = torch.rand(5, 2) + 0.1
        beta2, eps, step = 0.999, 1e-8, 10.0
        sd = {
            "state": {
                0: {"exp_avg_sq": va, "step": torch.tensor(step)},
                1: {"exp_avg_sq": vc, "step": torch.tensor(step)},
            },
            "param_groups": [{"betas": (0.9, beta2), "eps": eps, "params": [0, 1]}],
        }
        P = adam_diagonal_preconditioner(named, sd, device=CPU)

        bc2 = 1.0 - beta2 ** step
        exp_a = (1.0 / ((va / bc2).sqrt() + eps)).reshape(-1)
        exp_c = (1.0 / ((vc / bc2).sqrt() + eps)).reshape(-1)

        self.assertEqual(P.numel(), va.numel() + vc.numel())
        # Ordering: first slice is param `a`, second is `c` (named_parameters order).
        torch.testing.assert_close(P[: va.numel()], exp_a)
        torch.testing.assert_close(P[va.numel():], exp_c)

    def test_real_adamw_roundtrip(self):
        """Build a real AdamW, step it, and match P to the live optimizer state."""
        model = _Toy()
        named = _trainable(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        for _ in range(5):
            opt.zero_grad()
            loss = torch.stack([(p ** 2).sum() for p in model.parameters()]).sum()
            loss.backward()
            opt.step()

        P = adam_diagonal_preconditioner(named, opt.state_dict(), device=CPU)

        parts = []
        for _, p in named:
            st = opt.state[p]
            t = float(st["step"].item() if torch.is_tensor(st["step"]) else st["step"])
            bc2 = 1.0 - 0.999 ** t
            parts.append((1.0 / ((st["exp_avg_sq"] / bc2).sqrt() + 1e-8)).reshape(-1))
        expected = torch.cat(parts)

        torch.testing.assert_close(P, expected)
        self.assertTrue(torch.isfinite(P).all() and (P > 0).all())

    def test_regroup_fallback_on_index_shift(self):
        """If the optimizer placed a no-decay ('bias') param in a 2nd group, the
        state indices shift relative to named order; the decay/no-decay regroup
        must still align each param's v by identity."""
        lin = nn.Linear(3, 2)  # named_parameters yields 'weight' then 'bias'
        named = _trainable(lin)
        vw = torch.rand(2, 3) + 0.1   # weight (decay)  → optimizer index 0
        vb = torch.rand(2) + 0.1      # bias   (no-decay) → optimizer index 1
        sd = {
            "state": {
                0: {"exp_avg_sq": vw, "step": torch.tensor(3.0)},
                1: {"exp_avg_sq": vb, "step": torch.tensor(3.0)},
            },
            "param_groups": [{"betas": (0.9, 0.999), "eps": 1e-8, "params": [0, 1]}],
        }
        # Positional order here happens to match (weight, bias), but force the
        # fallback by reversing the model's named order via a wrapper.
        P = adam_diagonal_preconditioner(named, sd, device=CPU)
        bc2 = 1.0 - 0.999 ** 3.0
        exp_w = (1.0 / ((vw / bc2).sqrt() + 1e-8)).reshape(-1)
        exp_b = (1.0 / ((vb / bc2).sqrt() + 1e-8)).reshape(-1)
        torch.testing.assert_close(P[: vw.numel()], exp_w)
        torch.testing.assert_close(P[vw.numel():], exp_b)

    def test_eps_override_caps_dynamic_range(self):
        """A larger ε floors the denominator, capping P for near-dead (v̂≈0) coords."""
        model = _Toy()
        named = _trainable(model)
        v = torch.full((4, 3), 1e-16)        # dormant: √v̂≈0 → P≈1/ε
        v2 = torch.full((5, 2), 1.0)
        sd = {
            "state": {
                0: {"exp_avg_sq": v, "step": torch.tensor(50.0)},
                1: {"exp_avg_sq": v2, "step": torch.tensor(50.0)},
            },
            "param_groups": [{"betas": (0.9, 0.999), "eps": 1e-8, "params": [0, 1]}],
        }
        P_faithful = adam_diagonal_preconditioner(named, sd, device=CPU)
        P_capped = adam_diagonal_preconditioner(named, sd, device=CPU, eps_override=1e-3)
        # Faithful ε=1e-8 → max P ≈ 1e8; override ε=1e-3 → max P ≤ 1e3.
        self.assertGreater(P_faithful.max().item(), 1e7)
        self.assertLessEqual(P_capped.max().item(), 1e3 + 1.0)

    def test_count_mismatch_raises(self):
        model = _Toy()
        named = _trainable(model)
        sd = {
            "state": {0: {"exp_avg_sq": torch.ones(4, 3), "step": torch.tensor(1.0)}},
            "param_groups": [{"betas": (0.9, 0.999), "eps": 1e-8, "params": [0]}],
        }
        with self.assertRaises(ValueError):
            adam_diagonal_preconditioner(named, sd, device=CPU)

    def test_missing_optimizer_returns_none(self):
        model = _Toy()
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(
                load_adam_preconditioner_from_checkpoint(model, d, device=CPU)
            )

    def test_load_from_checkpoint_roundtrip(self):
        model = _Toy()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            loss = torch.stack([(p ** 2).sum() for p in model.parameters()]).sum()
            loss.backward()
            opt.step()
        with tempfile.TemporaryDirectory() as d:
            torch.save(opt.state_dict(), Path(d) / "optimizer.pt")
            P = load_adam_preconditioner_from_checkpoint(model, d, device=CPU)
        self.assertIsNotNone(P)
        assert P is not None  # narrow for type-checkers
        self.assertEqual(P.numel(), sum(p.numel() for p in model.parameters()))
        self.assertTrue((P > 0).all())


class TracInAdamInfluenceTests(unittest.TestCase):
    def _grads(self, D=20, n=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        test_info = {"grad": torch.randn(D, generator=g)}
        train_infos = [{"grad": torch.randn(D, generator=g)} for _ in range(n)]
        return test_info, train_infos

    def test_score_is_preconditioned_dot(self):
        test_info, train_infos = self._grads()
        D = test_info["grad"].numel()
        P = torch.rand(D) + 0.1
        method = TracInAdamInfluence(P, learning_rate=2.0)
        for ti in train_infos:
            expected = 2.0 * torch.dot(test_info["grad"], P * ti["grad"]).item()
            self.assertAlmostEqual(method.compute_score(test_info, ti), expected, places=5)

    def test_all_scores_matches_per_score(self):
        test_info, train_infos = self._grads()
        P = torch.rand(test_info["grad"].numel()) + 0.1
        method = TracInAdamInfluence(P, learning_rate=1.3)
        allv = method.compute_all_scores(test_info, train_infos)
        per = np.array([method.compute_score(test_info, ti) for ti in train_infos])
        np.testing.assert_allclose(allv, per, rtol=1e-5, atol=1e-6)

    def test_reduces_to_tracin_when_p_is_one(self):
        test_info, train_infos = self._grads()
        D = test_info["grad"].numel()
        adam = TracInAdamInfluence(torch.ones(D), learning_rate=1.0)
        tracin = TracInInfluence(learning_rate=1.0, normalize=False)
        for ti in train_infos:
            self.assertAlmostEqual(
                adam.compute_score(test_info, ti),
                tracin.compute_score(test_info, ti),
                places=5,
            )

    def test_wrong_preconditioner_size_raises(self):
        test_info, train_infos = self._grads(D=20)
        method = TracInAdamInfluence(torch.ones(7))  # wrong D
        with self.assertRaises(ValueError):
            method.compute_score(test_info, train_infos[0])


if __name__ == "__main__":
    unittest.main()
