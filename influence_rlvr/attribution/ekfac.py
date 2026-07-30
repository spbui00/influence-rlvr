"""EK-FAC influence (Eigenvalue-corrected Kronecker-Factored Approximate Curvature).

Approximates the same damped-Fisher inverse as `CGInfluence`, but with a per-layer
Kronecker-factored parametric model of F fitted ONCE, so each test-side solve is two
small matmuls per layer instead of a CG run (Grosse et al. 2023 use exactly this to
scale IF to LLMs):

    K-FAC:   F_l ≈ A_l ⊗ S_l,   A_l = E[a aᵀ]  (layer inputs),
                                S_l = E[δ δᵀ]  (grads w.r.t. layer pre-activations)
    EK-FAC:  keep the K-FAC eigenbasis Q_A, Q_S but refit the eigenvalues against
             the true per-sample gradients G_l = δ aᵀ:
                 Λ_ij = E[(Q_Sᵀ G_l Q_A)²_ij]
    solve:   H_l = Q_S [ (Q_Sᵀ G_l Q_A) / (Λ + λ) ] Q_Aᵀ     ⇔  h ≈ (F̂ + λI)⁻¹ g

Scope: stacks of `nn.Linear` layers (the toy MLPs; LoRA A/B mats would also fit).
A layer's bias is folded into its weight block via a homogeneous input coordinate
(a → [a, 1]) so one factor pair covers weight+bias jointly. Trainable parameters
NOT belonging to a traced Linear fall back to h = g/λ (i.e. F̂ = 0 there).

The expectation defining F is supplied by the caller as `(weight, closure)` pairs —
for the policy Fisher, one pair per (z, y) with weight π(y|z)/n_z and the closure
returning log π(y|z). This is the same distribution `policy_fisher_fvp_from_grad_cache`
uses, so EK-FAC and CG approximate the SAME operator and any LDS gap between them is
attributable to the approximation, not the operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .base import BaseInfluenceMethod

FitSample = tuple[float, Callable[[], torch.Tensor]]


def _eigh_basis(mat: torch.Tensor) -> torch.Tensor:
    """Eigenbasis of a symmetric PSD covariance, robust to near-rank-deficiency.

    cusolver's fp32 syevd fails to converge (LinAlgError code 33) on the nearly
    low-rank covariances a converged policy produces (clusters of ~0 repeated
    eigenvalues) — seen at checkpoint-200 of the P1 ref run. Ladder: decompose in
    fp64 (robust to exactly this), then fp64 + diagonal jitter, then CPU LAPACK.
    Returns the basis in the input dtype/device."""
    m64 = mat.to(torch.float64)
    try:
        _, q = torch.linalg.eigh(m64)
    except torch.linalg.LinAlgError:
        n = m64.shape[0]
        jitter = max(float(m64.diagonal().abs().mean()), 1e-30) * 1e-8
        eye = torch.eye(n, dtype=m64.dtype, device=m64.device)
        try:
            _, q = torch.linalg.eigh(m64 + jitter * eye)
        except torch.linalg.LinAlgError:
            _, q = torch.linalg.eigh((m64 + jitter * eye).cpu())
            q = q.to(m64.device)
    return q.to(mat.dtype)


def _delta(entry: dict, name: str) -> torch.Tensor:
    """The backward-pass gradient captured for one forward call — with a loud
    diagnosis when it is missing (a forward that the backward never walked)."""
    if "delta" not in entry:
        raise RuntimeError(
            f"fit_ekfac: module {name!r} recorded a forward with no matching "
            f"backward gradient. The usual cause is gradient checkpointing (model-"
            f"level OR a checkpoint() inside the closure, e.g. _compute_per_token_"
            f"logps's auto-checkpoint): the recompute pass re-fires the forward "
            f"hooks with no backward. Run the fit closures with checkpointing OFF.")
    return entry["delta"]


@dataclass
class _LinearBlock:
    name: str
    in_dim: int                # in_features (+1 if bias folded in)
    out_dim: int
    has_bias: bool
    weight_slice: slice        # position of the weight in the flat parameter vector
    bias_slice: slice | None
    q_a: torch.Tensor | None = None      # (in_dim, in_dim) eigenbasis of A
    q_s: torch.Tensor | None = None      # (out_dim, out_dim) eigenbasis of S
    lam: torch.Tensor | None = None      # (out_dim, in_dim) EK-FAC corrected eigenvalues

    def unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """Flat vector → (out_dim, in_dim) block, bias as the last column."""
        w = flat[self.weight_slice].view(self.out_dim, -1)
        if not self.has_bias:
            return w
        b = flat[self.bias_slice].view(self.out_dim, 1)
        return torch.cat([w, b], dim=1)

    def write_flat(self, block: torch.Tensor, out: torch.Tensor) -> None:
        if self.has_bias:
            out[self.weight_slice] = block[:, :-1].reshape(-1)
            out[self.bias_slice] = block[:, -1].reshape(-1)
        else:
            out[self.weight_slice] = block.reshape(-1)


@dataclass
class EKFACFactors:
    blocks: list[_LinearBlock]
    n_params: int
    covered: torch.Tensor  # bool mask over the flat vector

    def uncovered_mask(self) -> torch.Tensor:
        return ~self.covered


def _flat_slices(model: nn.Module) -> tuple[dict[int, slice], int]:
    """Map id(param) → slice in the `model.parameters()` (requires_grad) flat order —
    the same order `flatten_trainable_parameters` / the toy grad bundles use."""
    slices: dict[int, slice] = {}
    offset = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        slices[id(p)] = slice(offset, offset + p.numel())
        offset += p.numel()
    return slices, offset


def _traced_blocks(model: nn.Module) -> tuple[list[tuple[nn.Linear, _LinearBlock]], int]:
    slices, n_params = _flat_slices(model)
    out: list[tuple[nn.Linear, _LinearBlock]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not mod.weight.requires_grad:
            continue
        has_bias = mod.bias is not None and mod.bias.requires_grad
        out.append((mod, _LinearBlock(
            name=name,
            in_dim=mod.in_features + (1 if has_bias else 0),
            out_dim=mod.out_features,
            has_bias=has_bias,
            weight_slice=slices[id(mod.weight)],
            bias_slice=slices[id(mod.bias)] if has_bias else None,
        )))
    return out, n_params


def fit_ekfac(
    model: nn.Module,
    samples: Sequence[FitSample],
    *,
    dtype: torch.dtype = torch.float64,
    progress: bool = False,
) -> EKFACFactors:
    """Fit EK-FAC factors for every trainable nn.Linear in `model`.

    samples — (weight, closure) pairs defining the Fisher expectation. Each closure
    runs its own forward and returns the scalar to differentiate (e.g. log π(y|z));
    it is invoked twice (covariance pass + eigenvalue-correction pass). Weights need
    not be normalized. A module called k times inside one closure contributes k
    activation/δ rows to A and S (token-level convention) and the SUMMED δaᵀ to the
    per-sample gradient used for the eigenvalue correction.

    Factors accumulate on the model's own device in `dtype` (fp64 for the toy;
    use fp32 on GPU at LLM scale — LoRA-r32 on a 1.5B carries ~29 GB of factors,
    freed block-by-block as each covariance is eigendecomposed). Gradient
    checkpointing must be OFF: the recompute pass runs the forward under no_grad,
    where the δ-capture hook cannot attach.
    """
    if not samples:
        raise ValueError("fit_ekfac needs at least one (weight, closure) sample.")
    traced, n_params = _traced_blocks(model)
    if not traced:
        raise ValueError("fit_ekfac found no trainable nn.Linear layers to trace.")
    params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device

    # forward hooks capture the layer input; a tensor hook on the output captures
    # δ = ∂loss/∂preactivation. Tensor hooks fire under torch.autograd.grad too.
    captures: dict[str, list[dict]] = {blk.name: [] for _, blk in traced}

    def make_hook(name: str):
        def fwd(module, inputs, output):
            entry = {"a": inputs[0].detach().reshape(-1, inputs[0].shape[-1])}
            captures[name].append(entry)
            output.register_hook(lambda g: entry.__setitem__(
                "delta", g.detach().reshape(-1, g.shape[-1])))
        return fwd

    handles = [mod.register_forward_hook(make_hook(blk.name)) for mod, blk in traced]

    def run(closure):
        for lst in captures.values():
            lst.clear()
        loss = closure()
        torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

    def aug(a: torch.Tensor, has_bias: bool) -> torch.Tensor:
        if not has_bias:
            return a
        return torch.cat([a, torch.ones_like(a[:, :1])], dim=1)

    # Accumulate in fp64 regardless of the storage dtype: on a CONVERGED policy,
    # rare near-zero-probability sampled tokens carry extreme score gradients whose
    # squares overflow fp32 (inf − inf → NaN poisoned the whole checkpoint-200 fit).
    acc = torch.float64

    def closure_ok() -> bool:
        """Screen the just-run closure: any non-finite activation/δ capture drops
        the WHOLE sample (deterministic — pass 2 recomputes the same values)."""
        for lst in captures.values():
            for e in lst:
                if not torch.isfinite(e["a"]).all():
                    return False
                d = e.get("delta")
                if d is not None and not torch.isfinite(d).all():
                    return False
        return True

    n_skip = 0
    try:
        # ── pass 1: covariances ────────────────────────────────────────────────
        A = {b.name: torch.zeros(b.in_dim, b.in_dim, dtype=acc, device=device)
             for _, b in traced}
        S = {b.name: torch.zeros(b.out_dim, b.out_dim, dtype=acc, device=device)
             for _, b in traced}
        w_total = 0.0
        for i, (w, closure) in enumerate(samples):
            run(closure)
            if not closure_ok():
                n_skip += 1
                continue
            w_total += w
            for _, blk in traced:
                for e in captures[blk.name]:
                    a = aug(e["a"].to(acc), blk.has_bias)
                    d = _delta(e, blk.name).to(acc)
                    A[blk.name] += w * (a.T @ a)
                    S[blk.name] += w * (d.T @ d)
            if progress and (i + 1) % 50 == 0:
                print(f"    [ekfac] covariance pass {i + 1}/{len(samples)}", flush=True)
        if n_skip:
            print(f"    [ekfac] WARNING: dropped {n_skip}/{len(samples)} samples with "
                  f"non-finite captures (extreme low-probability tokens)", flush=True)
        if w_total <= 0:
            raise RuntimeError("fit_ekfac: every sample had non-finite captures.")
        # Eigendecompose block-by-block, FREEING each covariance as its eigenbasis
        # lands — halves the peak (A/S and Q never fully coexist).
        for _, blk in traced:
            blk.q_a = _eigh_basis(A.pop(blk.name) / w_total).to(dtype)
            blk.q_s = _eigh_basis(S.pop(blk.name) / w_total).to(dtype)
        if progress:
            print(f"    [ekfac] eigenbases done ({len(traced)} blocks)", flush=True)

        # ── pass 2: eigenvalue correction on true per-sample gradients ────────
        lam = {b.name: torch.zeros(b.out_dim, b.in_dim, dtype=acc, device=device)
               for _, b in traced}
        for i, (w, closure) in enumerate(samples):
            run(closure)
            if not closure_ok():
                continue                     # deterministically the same skips as pass 1
            for _, blk in traced:
                assert blk.q_a is not None and blk.q_s is not None
                G = torch.zeros(blk.out_dim, blk.in_dim, dtype=acc, device=device)
                for e in captures[blk.name]:
                    a = aug(e["a"].to(acc), blk.has_bias)
                    G += _delta(e, blk.name).to(acc).T @ a
                G_tilde = blk.q_s.to(acc).T @ G @ blk.q_a.to(acc)
                lam[blk.name] += w * G_tilde.pow(2)
            if progress and (i + 1) % 50 == 0:
                print(f"    [ekfac] eigenvalue pass {i + 1}/{len(samples)}", flush=True)
        for _, blk in traced:
            blk.lam = (lam[blk.name] / w_total).to(dtype)
            assert blk.q_a is not None and blk.q_s is not None
            if not (torch.isfinite(blk.q_a).all() and torch.isfinite(blk.q_s).all()
                    and torch.isfinite(blk.lam).all()):
                raise RuntimeError(
                    f"fit_ekfac: non-finite factors for block {blk.name!r} despite "
                    f"fp64 accumulation and sample screening — inspect the Fisher batch.")
    finally:
        for h in handles:
            h.remove()

    covered = torch.zeros(n_params, dtype=torch.bool)
    for _, blk in traced:
        covered[blk.weight_slice] = True
        if blk.has_bias:
            covered[blk.bias_slice] = True
    return EKFACFactors(blocks=[b for _, b in traced], n_params=n_params, covered=covered)


class EKFACInfluence(BaseInfluenceMethod):
    """(F̂_ekfac + λI)⁻¹-preconditioned influence — drop-in analogue of `CGInfluence`.

        h = (F̂ + λI)⁻¹ g_test    (closed form per layer, no CG iterations)
        IF(z_train) = g_trainᵀ h

    Sign convention matches `CGInfluence.compute_score` (raw dot; PBRF-negation is
    the caller's business). Solves are cached per id(test_info), same caveat as CG.
    """

    def __init__(self, factors: EKFACFactors, lambda_damp: float = 1.0,
                 rel_damp: float | None = None):
        """rel_damp — per-block RELATIVE damping λ_b = rel_damp · mean(Λ_b) (the
        kronfluence heuristic; robust to the scale differences between LoRA A and
        B blocks). None (default) keeps the single absolute lambda_damp. The
        off-block fallback always uses the absolute lambda_damp."""
        self.factors = factors
        self.lambda_damp = float(lambda_damp)
        self.rel_damp = None if rel_damp is None else float(rel_damp)
        self._h_cache: dict[int, torch.Tensor] = {}

    def _block_damp(self, blk: _LinearBlock) -> float:
        assert blk.lam is not None
        if self.rel_damp is None:
            return self.lambda_damp
        return max(self.rel_damp * float(blk.lam.mean()), 1e-12)

    def _dtype(self) -> torch.dtype:
        for blk in self.factors.blocks:
            if blk.lam is not None:
                return blk.lam.dtype
        return torch.float64

    def solve(self, g_test: torch.Tensor) -> torch.Tensor:
        dt = self._dtype()
        g = g_test.detach().to(dtype=dt).reshape(-1)
        if g.numel() != self.factors.n_params:
            raise ValueError(f"gradient has {g.numel()} entries, factors were fit "
                             f"for {self.factors.n_params}.")
        h = torch.empty_like(g)
        # F̂ = 0 off the traced blocks → (0 + λ)⁻¹ g there.
        h[self.factors.uncovered_mask()] = g[self.factors.uncovered_mask()] / self.lambda_damp
        for blk in self.factors.blocks:
            assert blk.q_a is not None and blk.q_s is not None and blk.lam is not None
            G = blk.unflatten(g).to(blk.lam.device)
            H = blk.q_s @ ((blk.q_s.T @ G @ blk.q_a) / (blk.lam + self._block_damp(blk))) @ blk.q_a.T
            blk.write_flat(H.to(h.device), h)
        return h.to(dtype=torch.float32)

    def apply_fisher(self, v: torch.Tensor) -> torch.Tensor:
        """F̂ v — the fitted Kronecker Fisher applied to a flat vector (validation:
        `apply_fisher(solve(g)) + λ·solve(g)` must reproduce `g` exactly)."""
        dt = self._dtype()
        x = v.detach().to(dtype=dt).reshape(-1)
        out = torch.zeros_like(x)
        for blk in self.factors.blocks:
            assert blk.q_a is not None and blk.q_s is not None and blk.lam is not None
            V = blk.unflatten(x).to(blk.lam.device)
            FV = blk.q_s @ ((blk.q_s.T @ V @ blk.q_a) * blk.lam) @ blk.q_a.T
            blk.write_flat(FV.to(out.device), out)
        return out.to(dtype=torch.float32)

    def _h_for(self, test_info: dict) -> torch.Tensor:
        key = id(test_info)
        if key not in self._h_cache:
            self._h_cache[key] = self.solve(test_info["grad"])
        return self._h_cache[key]

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
