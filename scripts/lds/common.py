"""Shared plumbing for the two toy LDS protocol scripts (forward_lds / history_lds).

Reuses the canonical pieces — ToyAutoregressiveMLP, the clustered dataset generator,
the exhaustive-GRPO objective, and the per-method influence helpers — and adds only
what the protocols need: a training loop with a per-step callback, a checkpoint-level
method scorer, mask/visit/Spearman utilities, and a ground-truth cache.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

_SCRIPTS = Path(__file__).resolve().parents[1]
_ROOT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from lds_eval_toy_grpo import (  # noqa: E402
    ToyAutoregressiveMLP,
    _toy_adam_preconditioner,
    generate_clustered_dataset,
)
from influence_rlvr.attribution import (  # noqa: E402
    CGInfluence,
    EKFACInfluence,
    fit_ekfac,
    policy_fisher_fvp_from_grad_cache,
)
from influence_rlvr.toy_grpo import (  # noqa: E402
    GradientObjective,
    ToyRolloutMode,
    _ALL_TWO_TOKEN_SEQUENCES,
    _toy_objective_and_debug,
    build_toy_policy_fisher_inputs,
    clone_toy_model,
    exact_expected_reward,
)

METHODS = ("dot", "tracin-adam", "ekfac", "cg")


# ── args shared by both protocols ────────────────────────────────────────────
def add_common_args(parser) -> None:
    d = parser.add_argument
    d("--n-clusters", type=int, default=60)
    d("--per-cluster", type=int, default=3)
    d("--input-dim", type=int, default=65,
      help="must be >= n_clusters (onehot) or >= signature_dim (overlap)")
    d("--cluster-signal", type=float, default=3.0)
    d("--cluster-target-mode", choices=["parity", "random"], default="parity")
    # ── discrimination levers (default off → reproduces the original fixture) ──
    d("--hard-fraction", type=float, default=0.0,
      help="fraction of clusters knocked to --signal-hard (heterogeneous saturation)")
    d("--signal-hard", type=float, default=None,
      help="signal for hard clusters (default cluster_signal/3)")
    d("--signature-mode", choices=["onehot", "overlap"], default="onehot",
      help="overlap = clusters share a feature block → real curvature for ekfac/cg")
    d("--signature-dim", type=int, default=8, help="shared-block width for --signature-mode overlap")
    d("--test-signal-span", choices=["strong", "span"], default="strong",
      help="span = test clusters cover the signal range (use with --hard-fraction)")
    d("--conflict-fraction", type=float, default=0.0,
      help="fraction of each cluster's members flipped to the complement target (clean hard negatives)")
    d("--n-test", type=int, default=5)
    d("--hidden-dim", type=int, default=16)
    d("--lr", type=float, default=1e-3)
    d("--beta", type=float, default=0.1)
    d("--no-adam", action="store_true")
    d("--seed", type=int, default=42)
    d("--methods", type=str, default="dot,tracin-adam,ekfac,cg",
      help=f"comma list from {METHODS}")
    d("--lambda-damp", type=float, default=0.1)
    d("--cg-iters", type=int, default=100)
    d("--cg-tol", type=float, default=1e-6)
    d("--m-subsets", type=int, default=100)
    d("--output-dir", type=str, default="outputs/lds_protocols")
    d("--cache-dir", type=str, default="outputs/lds_protocols/_cache")
    d("--no-cache", action="store_true")


def parse_methods(spec: str) -> list[str]:
    methods = [m.strip() for m in spec.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"Unknown method {m!r}; choices: {METHODS}")
    return methods


def build_fixture(args):
    """-> (train_examples, test_examples, ref_model). ref_model is π_ref — the shared
    starting state for the full model and every subset retrain."""
    train_examples, test_examples, _, _, _ = generate_clustered_dataset(
        n_clusters=args.n_clusters,
        per_cluster=args.per_cluster,
        input_dim=args.input_dim,
        cluster_signal=args.cluster_signal,
        target_mode=args.cluster_target_mode,
        seed=args.seed,
        hard_fraction=args.hard_fraction,
        signal_hard=args.signal_hard,
        signature_mode=args.signature_mode,
        signature_dim=args.signature_dim,
        test_signal_span=args.test_signal_span,
        conflict_fraction=args.conflict_fraction,
    )
    test_examples = test_examples[: args.n_test]
    torch.manual_seed(args.seed)
    ref_model = ToyAutoregressiveMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim)
    for m in ref_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return train_examples, test_examples, ref_model


# ── GRPO training with a per-step hook ───────────────────────────────────────
def train_grpo(
    examples: Sequence,
    steps: int,
    *,
    ref_model: nn.Module,
    lr: float,
    use_adam: bool = True,
    beta: float = 0.0,
    seed: int = 0,
    start_state: dict | None = None,
    on_step: Callable[[int, nn.Module, torch.optim.Optimizer], None] | None = None,
):
    """Round-robin exhaustive-rollout GRPO from `start_state` (default π_ref).
    Mirrors lds_eval_toy_grpo.train_model_with_history's update rule; adds `on_step`
    so callers can record rewards / snapshot checkpoints without retaining state.
    Returns (model, optimizer)."""
    torch.manual_seed(seed)
    model = ToyAutoregressiveMLP(input_dim=int(ref_model.input_dim),
                                 hidden_dim=int(ref_model.hidden_dim))
    model.load_state_dict(start_state if start_state is not None else ref_model.state_dict())
    opt_cls = torch.optim.Adam if use_adam else torch.optim.SGD
    optimizer = opt_cls(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        example = examples[(step - 1) % len(examples)]
        old_model = clone_toy_model(model)
        optimizer.zero_grad()
        objective, _, _ = _toy_objective_and_debug(
            model, example, G=4,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            seed=seed + step, epsilon=0.2, beta=beta,
            old_model=old_model, ref_model=ref_model,
            advantage_eps=1e-4, objective_mode=GradientObjective.GRPO_TRAIN,
        )
        objective.backward()
        optimizer.step()
        if on_step is not None:
            on_step(step, model, optimizer)
    return model, optimizer


def adam_precond_vec(optimizer, model) -> torch.Tensor:
    """Flat P = 1/(√v̂+ε) in model.parameters() order (ones when not Adam)."""
    return _toy_adam_preconditioner(optimizer, model)


def test_rewards(model, test_examples) -> np.ndarray:
    return np.array([exact_expected_reward(model, te) for te in test_examples], dtype=np.float64)


# ── per-method scores at ONE checkpoint ──────────────────────────────────────
def score_at_checkpoint(
    method: str,
    model: nn.Module,
    train_examples: Sequence,
    test_examples: Sequence,
    *,
    precond_vec: torch.Tensor | None,
    lambda_damp: float,
    cg_iters: int,
    cg_tol: float,
    beta: float,
    ref_model: nn.Module,
) -> np.ndarray:
    """[n_test, n_train] influence scores for `method` at the model's current state.

    All methods consume the same test/train gradients and (where applicable) the same
    exhaustive policy-Fisher distribution, so score differences are attributable to
    the operator alone. `precond_vec` is the Adam diagonal for tracin-adam (ones → dot).
    """
    n_test, n_train = len(test_examples), len(train_examples)
    out = np.zeros((n_test, n_train), dtype=np.float64)
    ekfac_obj = None
    for i, te in enumerate(test_examples):
        g_test, grad_cache, prob_cache, train_infos = build_toy_policy_fisher_inputs(
            model, train_examples, te, beta=beta, ref_model=ref_model,
        )
        gt = g_test.to(dtype=torch.float32)
        if method == "dot":
            h = gt
            out[i] = np.array([float(torch.dot(ti["grad"], h)) for ti in train_infos])
        elif method == "tracin-adam":
            if precond_vec is None:
                raise ValueError("tracin-adam needs the Adam preconditioner vector")
            h = precond_vec.to(dtype=torch.float32) * gt
            out[i] = np.array([float(torch.dot(ti["grad"], h)) for ti in train_infos])
        elif method == "cg":
            cg = CGInfluence(
                fvp_fn=policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache),
                lambda_damp=lambda_damp, cg_iters=cg_iters, cg_tol=cg_tol,
            )
            out[i] = cg.compute_all_scores({"grad": gt}, train_infos)
        elif method == "ekfac":
            if ekfac_obj is None:  # factors depend on (model, pool) — fit once, reuse per test
                sequences = _ALL_TWO_TOKEN_SEQUENCES.to(model.device)
                samples = []
                for ex in train_examples:
                    z = ex.z_tensor(device=model.device)
                    with torch.no_grad():
                        p = model.sequence_log_probs(z, sequences).exp()
                    for y in range(sequences.shape[0]):
                        w = float(p[y]) / len(train_examples)
                        if w > 0:
                            samples.append((w, lambda z=z, y=y: model.sequence_log_probs(
                                z, sequences[y:y + 1])[0]))
                ekfac_obj = EKFACInfluence(fit_ekfac(model, samples), lambda_damp=lambda_damp)
            # solve() directly — the object's id(test_info)-keyed cache is unsafe when
            # per-test dicts are gc'd and their ids recycled across loop iterations.
            h = ekfac_obj.solve(gt)
            out[i] = np.array([float(torch.dot(ti["grad"].to(dtype=torch.float32), h))
                               for ti in train_infos])
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


# ── masks, visit weights, Spearman ───────────────────────────────────────────
def sample_masks(n_train: int, m_subsets: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(m_subsets, n_train)).astype(bool)


def visit_counts(mask: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """[len(ks), |S|] round-robin visit counts of the masked examples after k steps."""
    s = int(mask.sum())
    ks = ks.reshape(-1, 1)
    base = ks // s
    extra = (np.arange(s)[None, :] < (ks % s)).astype(np.int64)
    return base + extra


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


# ── ground-truth cache ───────────────────────────────────────────────────────
def dataset_lever_keys(args) -> dict:
    """Non-default dataset-lever args, for the ground-truth cache key. Anything that
    changes the FIXTURE must land in the key or a redesigned dataset silently reuses
    the old dataset's retrains. Included only when non-default so pre-existing cache
    files keep their hashes."""
    levers = {
        "hard_fraction": args.hard_fraction or None,
        "signal_hard": args.signal_hard,
        "signature_mode": None if args.signature_mode == "onehot" else args.signature_mode,
        "signature_dim": args.signature_dim if args.signature_mode == "overlap" else None,
        "test_signal_span": None if args.test_signal_span == "strong" else args.test_signal_span,
        "conflict_fraction": args.conflict_fraction or None,
    }
    return {k: v for k, v in levers.items() if v is not None}


def cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_cache(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def save_cache(path: Path | None, **arrays) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
