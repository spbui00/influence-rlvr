from __future__ import annotations

import argparse
import time
import copy
import json
import csv
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from influence_rlvr.toy_grpo import (
    ToyGRPOExample,
    ToyRolloutMode,
    build_toy_policy_fisher_inputs,
    clone_toy_model,
    compute_toy_historical_fisher_influence,
    exact_expected_reward,
    reward_for_sequences,
    _ALL_TWO_TOKEN_SEQUENCES,
    GradientObjective,
    _toy_objective_and_debug,
    ToyHistoricalWeightMode,
    compute_toy_gradient_bundle,
)
from influence_rlvr.attribution import (
    CGInfluence,
    TrueFisherInfluence,
    policy_fisher_fvp_from_grad_cache,
)
from influence_rlvr.lds import (
    compute_lds_cache_key,
    run_extremes_test,
    run_lds,
    run_per_example_test,
)


def run_saturation_diagnostics(
    full_model,
    checkpoints,
    train_examples,
    ref_model,
    args,
    n_probe: int = 8,
    n_grad_traj: int = 3,
    n_checkpoints: int = 5,
) -> dict:
    """
    Two probes to detect whether saturation is actually happening:

    Probe 1: π(target|x) for the first `n_probe` training examples at θˢ.
             1 − π(target|x) is the "headroom" in which the score function ∇log π
             can be nonzero. Values like 1e-1 mean the model is not saturated;
             values like 1e-4 or smaller mean the IF gradient is genuinely vanishing.

    Probe 2: Gradient-norm trajectory for `n_grad_traj` training examples across
             `n_checkpoints` checkpoints spread evenly through training. If the
             norm starts high and ends near zero, that's the textbook saturation
             curve and the early-checkpoint signal is what trajectory historical
             would recover.
    """
    diag = {}

    # ----- Probe 1: saturation at θˢ -----
    print("\nSaturation probe at θˢ (1 - π(target|x) is the score-function headroom):")
    sat_rows = []
    with torch.no_grad():
        for ex in train_examples[:n_probe]:
            z = ex.z_tensor(device=full_model.device)
            target = ex.target_tensor(device=full_model.device)
            seqs, probs = full_model.exact_sequence_distribution(z)
            r = reward_for_sequences(seqs, target)
            p_target = float((probs * r).sum())
            print(f"  {ex.name}: π(target|x) = {p_target:.6f}  →  1-p = {1 - p_target:.2e}")
            sat_rows.append({"name": ex.name, "p_target": p_target, "headroom": 1 - p_target})
    diag["saturation_at_theta_s"] = sat_rows

    if sat_rows:
        headrooms = [r["headroom"] for r in sat_rows]
        mean_hr = sum(headrooms) / len(headrooms)
        min_hr = min(headrooms)
        max_hr = max(headrooms)
        if max_hr < 1e-3:
            verdict = "HEAVILY SATURATED — score function near-zero, IF gradients should vanish"
        elif max_hr < 0.05:
            verdict = "SATURATED — score function small; saturation pathology possible"
        elif max_hr < 0.2:
            verdict = "PARTIALLY SATURATED — model confident but gradients still alive"
        else:
            verdict = "NOT SATURATED — KL anchor / undertraining keeps π well below 1"
        print(f"  → headroom mean={mean_hr:.4f}, range=[{min_hr:.4e}, {max_hr:.4e}]")
        print(f"  → verdict: {verdict}")
        diag["saturation_verdict"] = verdict
        diag["headroom_mean"] = mean_hr
        diag["headroom_min"] = min_hr
        diag["headroom_max"] = max_hr

    # ----- Probe 2: gradient norm trajectory -----
    available_steps = sorted(int(s) for s in checkpoints.keys())
    if len(available_steps) < 2:
        print("\nGradient-norm trajectory: skipped (not enough checkpoints saved).")
        return diag

    # Pick `n_checkpoints` evenly-spaced steps including the final one.
    idxs = [int(round(i * (len(available_steps) - 1) / (n_checkpoints - 1))) for i in range(n_checkpoints)]
    probe_steps = sorted(set(available_steps[i] for i in idxs))

    print(f"\nGradient-norm trajectory across {len(probe_steps)} checkpoints:")
    grad_rows = {}
    for ex_idx in range(min(n_grad_traj, len(train_examples))):
        ex = train_examples[ex_idx]
        norms = []
        for step in probe_steps:
            tmp_model = clone_toy_model(full_model)
            tmp_model.load_state_dict(checkpoints[step])
            old_model = clone_toy_model(tmp_model)
            bundle = compute_toy_gradient_bundle(
                tmp_model, ex,
                G=4,
                rollout_mode=ToyRolloutMode.EXHAUSTIVE,
                objective_mode=GradientObjective.GRPO_TRAIN,
                old_model=old_model,
                beta=args.beta,
                ref_model=ref_model,
            )
            norms.append(float(bundle["grad"].norm()))
        steps_str = " ".join(f"step={s}: {n:.4f}" for s, n in zip(probe_steps, norms))
        print(f"  {ex.name}: {steps_str}")
        if norms[0] > 1e-9 and norms[-1] / norms[0] < 0.05:
            shape = "DECAYED 20×+ over training (textbook saturation)"
        elif norms[0] > 1e-9 and norms[-1] / norms[0] < 0.5:
            shape = "decayed moderately"
        elif norms[0] > 1e-9 and norms[-1] > norms[0] * 1.5:
            shape = "GREW over training (unusual — model still ramping up on this example)"
        else:
            shape = "roughly flat"
        print(f"    → {shape}")
        grad_rows[ex.name] = {"steps": probe_steps, "norms": norms, "shape": shape}

    diag["grad_norm_trajectory"] = grad_rows
    return diag


def _lds_cache_key(args) -> str:
    """Hash of the args that affect actual_rewards / subset_masks (i.e., not the IF method).

    Anything that changes the subset-training behavior or the test set must go
    in here. Anything that only changes IF computation must NOT — different IF
    methods reuse the same cache file.
    """
    is_clustered = getattr(args, "dataset", "iid") == "clustered"
    is_cont = getattr(args, "lds_mode", "from-scratch") == "continuation"
    payload = {
        "n_train": args.n_train,
        "n_test": args.n_test,
        "m_subsets": args.m_subsets,
        "subset_steps": args.subset_steps,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "beta": args.beta,
        "use_adam": not args.no_adam,
        "dataset": getattr(args, "dataset", "iid"),
        "n_clusters": getattr(args, "n_clusters", None) if is_clustered else None,
        "per_cluster": getattr(args, "per_cluster", None) if is_clustered else None,
        "test_cluster_ids": getattr(args, "test_cluster_ids", None) if is_clustered else None,
        "cluster_signal": getattr(args, "cluster_signal", None) if is_clustered else None,
        "alt_cluster_signal": getattr(args, "alt_cluster_signal", None) if is_clustered else None,
        "n_alt_clusters": getattr(args, "n_alt_clusters", None) if is_clustered else None,
        "cluster_target_mode": getattr(args, "cluster_target_mode", None) if is_clustered else None,
        "lds_mode": getattr(args, "lds_mode", "from-scratch"),
        "lds_cont_steps": getattr(args, "lds_cont_steps", None) if is_cont else None,
        "lds_cont_lr": getattr(args, "lds_cont_lr", None) if is_cont else None,
        "natural_gradient_cont": getattr(args, "natural_gradient_cont", False) if is_cont else False,
        "natural_gradient_cont_lambda": (
            getattr(args, "natural_gradient_cont_lambda", None)
            if (is_cont and getattr(args, "natural_gradient_cont", False)) else None
        ),
        "natural_gradient_cont_fisher_batch": (
            getattr(args, "natural_gradient_cont_fisher_batch", None)
            if (is_cont and getattr(args, "natural_gradient_cont", False)) else None
        ),
    }
    if is_clustered and getattr(args, "input_dim", 20) != 20:
        # keyed only when non-default so pre-existing cache files keep their hashes
        payload["input_dim"] = args.input_dim
    return compute_lds_cache_key(payload)



class ToyAutoregressiveMLP(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 16, output_dim: int = 2, use_bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.first = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=use_bias)
        )
        self.second = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=use_bias)
        )

    def first_token_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.first(z)

    def second_token_logits(
        self,
        z: torch.Tensor,
        first_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if first_tokens.ndim == 1:
            first_tokens = first_tokens.unsqueeze(1)
        aug = torch.cat([z, first_tokens.to(dtype=z.dtype)], dim=1)
        return self.second(aug)

    def per_token_log_probs(
        self,
        z: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(dtype=torch.float32)
        n_seq = int(sequences.shape[0])
        z_rep = z.expand(n_seq, -1)
        first_logits = self.first_token_logits(z_rep)
        first_log_probs = F.log_softmax(first_logits, dim=-1)
        first_tokens = sequences[:, 0]
        first_lp = first_log_probs.gather(1, first_tokens.unsqueeze(1)).squeeze(1)

        second_logits = self.second_token_logits(z_rep, first_tokens.float())
        second_log_probs = F.log_softmax(second_logits, dim=-1)
        second_tokens = sequences[:, 1]
        second_lp = second_log_probs.gather(1, second_tokens.unsqueeze(1)).squeeze(1)

        return torch.stack([first_lp, second_lp], dim=1)

    def sequence_log_probs(
        self,
        z: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        return self.per_token_log_probs(z, sequences).sum(dim=1)

    def exact_sequence_distribution(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = _ALL_TWO_TOKEN_SEQUENCES.to(self.device)
        sequence_log_probs = self.sequence_log_probs(z.to(self.device), sequences)
        return sequences, sequence_log_probs.exp()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def generate_dataset(n: int = 1000, input_dim: int = 20, seed: int = 42) -> list[ToyGRPOExample]:
    torch.manual_seed(seed)
    z = torch.randn(n, input_dim)
    # Rule: sum(z1, z2, z3) > 0 -> [1, 0], else [0, 1]
    rule_sum = z[:, 0] + z[:, 1] + z[:, 2]
    targets = torch.zeros(n, 2, dtype=torch.long)
    targets[rule_sum > 0] = torch.tensor([1, 0])
    targets[rule_sum <= 0] = torch.tensor([0, 1])

    examples = []
    for i in range(n):
        examples.append(ToyGRPOExample(
            name=f"ex_{i}",
            z=tuple(z[i].tolist()),
            target=tuple(targets[i].tolist()),
        ))
    return examples


def _resolve_signal_per_cluster(
    n_clusters: int,
    cluster_signal: float,
    signal_per_cluster: Sequence[float] | None,
    hard_fraction: float,
    signal_hard: float | None,
    seed: int,
) -> list[float]:
    """Per-cluster signal strengths. An explicit `signal_per_cluster` wins; otherwise
    build a uniform vector and (if hard_fraction>0) knock a random subset down to
    `signal_hard` (default cluster_signal/3). Hard-cluster choice is seeded and
    decoupled from cluster parity so difficulty does not correlate with the target."""
    if signal_per_cluster is not None:
        spc = list(signal_per_cluster)
        if len(spc) != n_clusters:
            raise ValueError(
                f"signal_per_cluster length ({len(spc)}) != n_clusters ({n_clusters})."
            )
        return spc
    spc = [float(cluster_signal)] * n_clusters
    n_hard = int(round(hard_fraction * n_clusters))
    if n_hard > 0:
        lo = cluster_signal / 3.0 if signal_hard is None else float(signal_hard)
        gen = torch.Generator().manual_seed(seed + 4242)
        for k in torch.randperm(n_clusters, generator=gen).tolist()[:n_hard]:
            spc[k] = lo
    return spc


def _build_signature_matrix(
    n_clusters: int,
    input_dim: int,
    signal_per_cluster: Sequence[float],
    signature_mode: str,
    signature_dim: int,
    seed: int,
) -> torch.Tensor:
    """[n_clusters, input_dim] additive cluster signatures (reused for train & test).

    'onehot'  — cluster k lives on its own axis z[k]; signatures are orthogonal,
                so identifying a cluster is nearly linear (dot already nails it).
    'overlap' — every cluster loads onto a *shared* block z[0:signature_dim] with
                random unit-norm loadings, scaled by its signal. Signatures are
                non-orthogonal → correlated features → genuine curvature for
                ekfac/cg to decorrelate that a plain dot cannot."""
    S = torch.zeros(n_clusters, input_dim)
    if signature_mode == "onehot":
        for k in range(n_clusters):
            S[k, k] = signal_per_cluster[k]
    elif signature_mode == "overlap":
        gen = torch.Generator().manual_seed(seed + 2024)
        loadings = torch.randn(n_clusters, signature_dim, generator=gen)
        loadings = loadings / loadings.norm(dim=1, keepdim=True).clamp_min(1e-8)
        for k in range(n_clusters):
            S[k, :signature_dim] = loadings[k] * signal_per_cluster[k]
    else:
        raise ValueError(f"Unsupported signature_mode={signature_mode!r}. Use 'onehot' or 'overlap'.")
    return S


def generate_clustered_dataset(
    n_clusters: int = 50,
    per_cluster: int = 4,
    test_cluster_ids: Sequence[int] | None = None,
    input_dim: int = 20,
    signal_per_cluster: Sequence[float] | None = None,
    cluster_signal: float = 3.0,
    background_scale: float = 0.5,
    target_mode: str = "parity",
    seed: int = 42,
    *,
    hard_fraction: float = 0.0,
    signal_hard: float | None = None,
    signature_mode: str = "onehot",
    signature_dim: int = 8,
    test_signal_span: str = "strong",
    conflict_fraction: float = 0.0,
) -> tuple[list[ToyGRPOExample], list[ToyGRPOExample], list[int], list[float], list[tuple[int, int]]]:
    """
    Non-iid dataset designed to probe saturation failure of single-checkpoint IFs.

    Each cluster k has a signature in z (see `signature_mode`) and its own target
    (alternating (1,0)/(0,1) by cluster id parity). Training examples are round-robin
    across clusters. With per_cluster small (e.g. 2-4), a 50% subset mask has
    non-trivial probability of wiping a whole cluster.

    Three levers make the fixture actually discriminate attribution methods:

    - Heterogeneous signals (`hard_fraction`, `signal_hard`): knock a random subset
      of clusters down to a low signal so some test clusters saturate and others
      cannot fit — the regime where single-checkpoint IFs should visibly fail.
      Pair with `test_signal_span='span'` so test clusters cover the signal range
      (default 'strong' keeps the old max-signal-only selection).
    - Overlapping signatures (`signature_mode='overlap'`, `signature_dim`): clusters
      share a feature block, so features are correlated and there is real curvature
      for ekfac/cg to correct that a plain dot cannot.
    - Within-cluster conflict (`conflict_fraction`): flip the last N members of each
      cluster to the complement target (tagged expected_influence='harmful'), giving
      a clean, target-specific negative tail instead of only diffuse interference.

    Returns (train_examples, test_examples, test_cluster_ids, signal_per_cluster,
             cluster_targets).
    """
    sig_span = n_clusters if signature_mode == "onehot" else signature_dim
    if input_dim < sig_span:
        raise ValueError(
            f"input_dim ({input_dim}) must be >= {'n_clusters' if signature_mode == 'onehot' else 'signature_dim'} "
            f"({sig_span}) for signature_mode={signature_mode!r}."
        )
    signal_per_cluster = _resolve_signal_per_cluster(
        n_clusters, cluster_signal, signal_per_cluster, hard_fraction, signal_hard, seed
    )
    signature = _build_signature_matrix(
        n_clusters, input_dim, signal_per_cluster, signature_mode, signature_dim, seed
    )

    torch.manual_seed(seed)
    n_train = n_clusters * per_cluster
    cluster_assignment = torch.arange(n_train) % n_clusters

    z_train = torch.randn(n_train, input_dim) * background_scale
    z_train += signature[cluster_assignment]

    if target_mode == "parity":
        cluster_targets = [(1, 0) if k % 2 == 0 else (0, 1) for k in range(n_clusters)]
    elif target_mode == "random":
        target_pool = [(0, 0), (0, 1), (1, 0), (1, 1)]
        gen = torch.Generator().manual_seed(seed + 1000)
        idxs = torch.randint(0, len(target_pool), (n_clusters,), generator=gen).tolist()
        cluster_targets = [target_pool[i] for i in idxs]
    else:
        raise ValueError(f"Unsupported target_mode={target_mode!r}. Use 'parity' or 'random'.")

    # Round-robin: example i is member j = i // n_clusters of cluster i % n_clusters.
    # Flip the last `n_conflict` members of every cluster to the complement target —
    # same signature, wrong label → a clean per-target harmful example.
    n_conflict = int(round(conflict_fraction * per_cluster))
    targets_train = torch.zeros(n_train, 2, dtype=torch.long)
    influence_tag: list[str | None] = [None] * n_train
    for i in range(n_train):
        k = int(cluster_assignment[i])
        member = i // n_clusters
        base = cluster_targets[k]
        if n_conflict > 0 and member >= per_cluster - n_conflict:
            targets_train[i] = torch.tensor((1 - base[0], 1 - base[1]))
            influence_tag[i] = "harmful"
        else:
            targets_train[i] = torch.tensor(base)

    train_examples = [
        ToyGRPOExample(
            name=f"c{int(cluster_assignment[i])}_ex{i}"
                 + ("_neg" if influence_tag[i] == "harmful" else ""),
            z=tuple(z_train[i].tolist()),
            target=tuple(targets_train[i].tolist()),
            expected_influence=influence_tag[i],
        )
        for i in range(n_train)
    ]

    if test_cluster_ids is None:
        if test_signal_span == "span":
            # spread test clusters across the signal range (both saturating & hard)
            order = sorted(range(n_clusters), key=lambda k: signal_per_cluster[k])
            step = max(1, len(order) // 5)
            test_cluster_ids = sorted(order[::step][:5])
        elif test_signal_span == "strong":
            max_signal = max(signal_per_cluster)
            strong = [k for k in range(n_clusters) if signal_per_cluster[k] >= max_signal - 1e-9]
            pool = strong if strong else list(range(n_clusters))
            step = max(1, len(pool) // 5)
            test_cluster_ids = pool[::step][:5]
        else:
            raise ValueError(f"Unsupported test_signal_span={test_signal_span!r}. Use 'strong' or 'span'.")
    test_cluster_ids = list(test_cluster_ids)
    for k in test_cluster_ids:
        if k < 0 or k >= n_clusters:
            raise ValueError(f"test_cluster_id {k} out of range [0, {n_clusters}).")

    n_test = len(test_cluster_ids)
    z_test = torch.randn(n_test, input_dim) * background_scale
    targets_test = torch.zeros(n_test, 2, dtype=torch.long)
    for i, k in enumerate(test_cluster_ids):
        z_test[i] += signature[k]
        targets_test[i] = torch.tensor(cluster_targets[k])

    test_examples = [
        ToyGRPOExample(
            name=f"test_c{test_cluster_ids[i]}",
            z=tuple(z_test[i].tolist()),
            target=tuple(targets_test[i].tolist()),
            split="test",
        )
        for i in range(n_test)
    ]

    return train_examples, test_examples, test_cluster_ids, signal_per_cluster, cluster_targets


def _build_fvp_cache(
    model: nn.Module, examples: Sequence[ToyGRPOExample]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute (grad_cache, prob_cache) for natural-gradient FVP — the policy
    Fisher F(θ) = (1/|B|) Σ_{z∈B} Σ_y π(y|z) ∇log π ∇log π^T evaluated on the
    Fisher batch `examples`.

    Pass [single_example] for the standard SGD-style minibatch natural gradient.
    Pass the full training set for the PBRF-consistent F that the IF formula
    actually uses — this is the operator that makes per-example Spearman = 1
    under K_cont=1.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    sequences = _ALL_TWO_TOKEN_SEQUENCES.to(model.device)
    n_seq = int(sequences.shape[0])
    D = sum(p.numel() for p in params)
    grad_cache: list[torch.Tensor] = []
    prob_cache: list[torch.Tensor] = []
    for example in examples:
        z = example.z_tensor(device=model.device)
        log_probs = model.sequence_log_probs(z, sequences)
        probs = log_probs.detach().exp()
        G_z = torch.zeros(n_seq, D, dtype=torch.float32, device=model.device)
        for y_idx in range(n_seq):
            grads = torch.autograd.grad(
                log_probs[y_idx], params, retain_graph=(y_idx < n_seq - 1)
            )
            G_z[y_idx] = torch.cat([g.detach().flatten() for g in grads]).to(dtype=torch.float32)
        grad_cache.append(G_z)
        prob_cache.append(probs.to(dtype=torch.float32))
    return grad_cache, prob_cache


def train_model_with_history(
    dataset: Sequence[ToyGRPOExample],
    steps: int = 1000,
    lr: float = 1e-3,
    hidden_dim: int = 16,
    seed: int = 0,
    save_checkpoints: bool = True,
    use_adam: bool = True,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
    start_model: nn.Module | None = None,
    natural_gradient_lambda: float | None = None,
    natural_gradient_cg_iters: int = 50,
    natural_gradient_cg_tol: float = 1e-6,
    natural_gradient_fisher_examples: Sequence[ToyGRPOExample] | None = None,
) -> dict:
    """Train GRPO with Adam/SGD or, when `natural_gradient_lambda` is provided,
    with natural-gradient steps θ_new = θ − lr · (F + λI)^{-1} · ∇L.

    `natural_gradient_fisher_examples`:
      - None (default): use the *singleton* {current example} as the Fisher batch
        — standard minibatch natural gradient. Cheaper but doesn't match the IF's
        full-train Fisher operator.
      - <list>: use this list as the Fisher batch at every step. Pass the full
        training set to get the PBRF-consistent F^{-1} that the IF formula uses
        — under this choice the per-example Spearman should be ≈ 1 at K_cont=1
        because ΔR = lr · φ_i exactly to first order.
    """
    use_natural_gradient = natural_gradient_lambda is not None
    torch.manual_seed(seed)
    # Infer input_dim from the data so wide clustered datasets (e.g. 60 clusters
    # → input_dim 65) build a matching model at every call site.
    data_input_dim = int(dataset[0].z_tensor().shape[-1])
    model = ToyAutoregressiveMLP(input_dim=data_input_dim, hidden_dim=hidden_dim)
    if start_model is not None:
        # Continuation training: keep π_ref = ref_model as the KL anchor, but
        # initialize from start_model (e.g. the full-training final checkpoint).
        model.load_state_dict(start_model.state_dict())
    elif ref_model is not None:
        # Start every training run from the same reference state so π_ref is shared
        # across full and subset training (matches the surrogate-IF derivation).
        model.load_state_dict(ref_model.state_dict())
    else:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if not use_natural_gradient:
        if use_adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    checkpoints = {}
    # Per-step Adam diagonal preconditioner P_c = 1/(√v̂+ε), read live right after
    # each optimizer.step(). Keyed by `step` (the update θ_{step-1}→θ_step), which
    # is the P that trajectory tracin-adam multiplies into g_test at checkpoints[step-1].
    # Only populated for Adam steps; SGD/natural-gradient leave it empty → tracin-adam
    # degenerates to plain `dot`.
    precond_checkpoints: dict[int, torch.Tensor] = {}
    if save_checkpoints:
        checkpoints[0] = copy.deepcopy(model.state_dict())

    history = []
    for step in range(1, steps + 1):
        example = dataset[(step - 1) % len(dataset)]  # round-robin over the dataset
        old_model = clone_toy_model(model)

        if optimizer is not None:
            optimizer.zero_grad()

        objective, _, debug = _toy_objective_and_debug(
            model,
            example,
            G=4,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            seed=seed + step,
            epsilon=0.2,
            beta=beta,
            old_model=old_model,
            ref_model=ref_model,
            advantage_eps=1e-4,
            objective_mode=GradientObjective.GRPO_TRAIN
        )

        if use_natural_gradient:
            # Natural gradient step: h = (F + λI)^{-1} · ∇L, then θ -= lr · h.
            grads = torch.autograd.grad(objective, params)
            g_flat = torch.cat([g.detach().flatten() for g in grads]).to(dtype=torch.float32)
            fisher_examples = (
                list(natural_gradient_fisher_examples)
                if natural_gradient_fisher_examples is not None
                else [example]
            )
            grad_cache, prob_cache = _build_fvp_cache(model, fisher_examples)
            fvp_fn = policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache)
            cg = CGInfluence(
                fvp_fn=fvp_fn,
                lambda_damp=float(natural_gradient_lambda),
                cg_iters=natural_gradient_cg_iters,
                cg_tol=natural_gradient_cg_tol,
            )
            h_flat, _ = cg.solve(g_flat)
            with torch.no_grad():
                offset = 0
                for p in params:
                    n = p.numel()
                    p.sub_(h_flat[offset:offset + n].view(p.shape), alpha=lr)
                    offset += n
        else:
            objective.backward()
            assert optimizer is not None
            optimizer.step()

        history.append({
            "step": step,
            "example_name": example.name,
            "loss": float(objective.item())
        })

        if save_checkpoints:
            checkpoints[step] = copy.deepcopy(model.state_dict())
            if optimizer is not None and use_adam:
                precond_checkpoints[step] = (
                    _toy_adam_preconditioner(optimizer, model).detach().clone()
                )

    return {
        "model": model,
        "history": history,
        "checkpoints": checkpoints,
        "precond_checkpoints": precond_checkpoints,
    }


def _train_subset_for_lds(
    args,
    *,
    ref_model: nn.Module,
    full_model: nn.Module,
    subset_train: Sequence[ToyGRPOExample],
    seed: int,
    use_adam: bool,
    full_train_examples: Sequence[ToyGRPOExample] | None = None,
) -> dict:
    """
    Train one subset model under the configured --lds-mode.

    - from-scratch: train from π_ref for --subset-steps at --lr (the original
      LDS protocol).
    - continuation: start from the full-training endpoint θ_T (= full_model)
      and continue training on the subset for --lds-cont-steps at --lds-cont-lr
      (default --lr / 10). Stays in the local regime where the IF linearization
      is most accurate.
    """
    if args.lds_mode == "continuation":
        cont_lr = args.lds_cont_lr if args.lds_cont_lr is not None else (args.lr / 10.0)
        use_ng = getattr(args, "natural_gradient_cont", False)
        nat_grad_lambda = (
            (args.natural_gradient_cont_lambda
             if args.natural_gradient_cont_lambda is not None else args.lambda_damp)
            if use_ng else None
        )
        # Pick the Fisher batch for natural-gradient training. `full-train` matches
        # the IF formula's Fisher operator and is what makes per-example Spearman = 1
        # at K_cont=1; `singleton` is standard minibatch NG (cheaper but mismatched).
        fisher_batch_examples: Sequence[ToyGRPOExample] | None = None
        if use_ng and getattr(args, "natural_gradient_cont_fisher_batch", "singleton") == "full-train":
            if full_train_examples is None:
                raise ValueError(
                    "natural_gradient_cont_fisher_batch=full-train requires full_train_examples."
                )
            fisher_batch_examples = list(full_train_examples)
        return train_model_with_history(
            subset_train,
            steps=args.lds_cont_steps,
            lr=cont_lr,
            hidden_dim=args.hidden_dim,
            seed=seed,
            save_checkpoints=False,
            use_adam=use_adam,
            beta=args.beta,
            ref_model=ref_model,
            start_model=full_model,
            natural_gradient_lambda=nat_grad_lambda,
            natural_gradient_cg_iters=args.cg_iters,
            natural_gradient_cg_tol=args.cg_tol,
            natural_gradient_fisher_examples=fisher_batch_examples,
        )
    return train_model_with_history(
        subset_train,
        steps=args.subset_steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=seed,
        save_checkpoints=False,
        use_adam=use_adam,
        beta=args.beta,
        ref_model=ref_model,
    )


def compute_toy_cg_influence(
    model: nn.Module,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    lambda_damp: float = 1.0,
    cg_iters: int = 50,
    cg_tol: float = 1e-6,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
    train_rollout_mode: str = "exhaustive",
) -> tuple[torch.Tensor, dict]:
    """PBRF-style IF at one checkpoint: CG+FVP solve against the true policy Fisher."""
    g_test, grad_cache, prob_cache, train_infos = build_toy_policy_fisher_inputs(
        model, train_examples, test_example, beta=beta, ref_model=ref_model, train_rollout_mode=train_rollout_mode
    )
    fvp_fn = policy_fisher_fvp_from_grad_cache(grad_cache, prob_cache)
    cg = CGInfluence(
        fvp_fn=fvp_fn,
        lambda_damp=lambda_damp,
        cg_iters=cg_iters,
        cg_tol=cg_tol,
    )
    test_info = {"grad": g_test}
    scores_np = cg.compute_all_scores(test_info, train_infos)
    cg_info = cg.cg_info_for(test_info) or {}
    return torch.from_numpy(scores_np), cg_info


def compute_toy_dot_influence(
    model: nn.Module,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
    train_rollout_mode: str = "exhaustive",
) -> tuple[torch.Tensor, dict]:
    """First-order (dot-product / TracIn-style) influence: IF = g_train · g_test.

    Same per-example gradients as `compute_toy_cg_influence`, but with NO Fisher
    inverse — it's the λ→∞ limit of CG (h = g_test). Use to test whether the
    second-order curvature correction actually buys anything over plain gradient
    alignment.
    """
    g_test, _grad_cache, _prob_cache, train_infos = build_toy_policy_fisher_inputs(
        model, train_examples, test_example, beta=beta, ref_model=ref_model, train_rollout_mode=train_rollout_mode
    )
    gt = g_test.detach().to(dtype=torch.float32)
    scores = torch.stack([
        torch.dot(ti["grad"].detach().to(dtype=torch.float32, device=gt.device), gt)
        for ti in train_infos
    ])
    return scores.cpu(), {"method": "dot", "n_train": len(train_infos)}


def _toy_adam_preconditioner(optimizer, model, *, dtype=torch.float32) -> torch.Tensor:
    """Flat P = 1/(√v̂+ε) over the model's trainable params, read from a LIVE Adam
    optimizer's state. Same param order as `build_toy_policy_fisher_inputs` flattens
    gradients (model.parameters() filtered to requires_grad). Params not yet stepped
    (or SGD, which has no exp_avg_sq) get P=1 → tracin-adam degenerates to dot."""
    group = optimizer.param_groups[0]
    beta2 = float(group.get("betas", (0.9, 0.999))[1])
    eps = float(group.get("eps", 1e-8))
    parts: list[torch.Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        st = optimizer.state.get(p, {})
        v = st.get("exp_avg_sq")
        if v is None:
            parts.append(torch.ones(p.numel(), dtype=dtype))
            continue
        step = st.get("step")
        t = float(step.item()) if torch.is_tensor(step) else float(step or 0)
        bc2 = 1.0 - beta2 ** t if t > 0 else 1.0
        v_hat = v.detach().to(dtype) / bc2
        parts.append((1.0 / (v_hat.sqrt() + eps)).reshape(-1))
    return torch.cat(parts)


def compute_toy_tracin_adam_influence(
    model: nn.Module,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    optimizer: torch.optim.Optimizer,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
    train_rollout_mode: str = "exhaustive",
) -> tuple[torch.Tensor, dict]:
    """Adam-preconditioned first-order influence: IF = g_test · (P ⊙ g_train),
    P = 1/(√v̂+ε) from the LIVE Adam optimizer. The faithful first-order effect of
    one AdamW step (vs `dot`, which assumes an SGD step). Same gradients as `dot`."""
    g_test, _gc, _pc, train_infos = build_toy_policy_fisher_inputs(
        model, train_examples, test_example, beta=beta, ref_model=ref_model, train_rollout_mode=train_rollout_mode
    )
    gt = g_test.detach().to(dtype=torch.float32)
    P = _toy_adam_preconditioner(optimizer, model).to(gt.device)
    if P.numel() != gt.numel():
        raise ValueError(f"Adam preconditioner ({P.numel()}) != grad dim ({gt.numel()})")
    h = P * gt  # fold the diagonal into the test side once
    scores = torch.stack([
        torch.dot(ti["grad"].detach().to(dtype=torch.float32, device=gt.device), h)
        for ti in train_infos
    ])
    return scores.cpu(), {"method": "tracin-adam", "n_train": len(train_infos)}


def compute_toy_ekfac_influence(
    model: nn.Module,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    lambda_damp: float = 1.0,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
    train_rollout_mode: str = "exhaustive",
    _factors_cache: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """EK-FAC influence: h = (F̂_ekfac + λI)⁻¹ g_test, closed-form per layer.

    Fits the Kronecker factors on the SAME exhaustive policy distribution the CG
    FVP uses (one (weight, closure) per (z, y) with weight π(y|z)/n_z), so any LDS
    gap vs `compute_toy_cg_influence` is attributable to the Kronecker/eigenvalue
    approximation, not the operator. Pass a dict as `_factors_cache` to reuse the
    fit across the test examples of one checkpoint (the fit depends only on the
    model + train pool, not on the test example).
    """
    from influence_rlvr.attribution import EKFACInfluence, fit_ekfac

    g_test, _grad_cache, prob_cache, train_infos = build_toy_policy_fisher_inputs(
        model, train_examples, test_example, beta=beta, ref_model=ref_model,
        train_rollout_mode=train_rollout_mode,
    )
    cache_key = "factors"
    if _factors_cache is not None and cache_key in _factors_cache:
        factors = _factors_cache[cache_key]
    else:
        sequences = _ALL_TWO_TOKEN_SEQUENCES.to(model.device)
        samples = []
        for ex, probs in zip(train_examples, prob_cache):
            z = ex.z_tensor(device=model.device)
            for y_idx in range(sequences.shape[0]):
                w = float(probs[y_idx]) / len(train_examples)
                if w <= 0.0:
                    continue
                samples.append((w, lambda z=z, y=y_idx: model.sequence_log_probs(
                    z, sequences[y:y + 1])[0]))
        factors = fit_ekfac(model, samples)
        if _factors_cache is not None:
            _factors_cache[cache_key] = factors
    ekfac = EKFACInfluence(factors, lambda_damp=lambda_damp)
    scores_np = ekfac.compute_all_scores({"grad": g_test}, train_infos)
    return torch.from_numpy(scores_np), {
        "method": "ekfac", "n_blocks": len(factors.blocks), "D": int(g_test.numel()),
    }


def compute_toy_true_fisher_influence(
    model: nn.Module,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    lambda_damp: float = 1.0,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Direct (Cholesky) inverse of the true policy Fisher at one checkpoint.

    Same operator as `compute_toy_cg_influence` — use this to validate CG
    (h_cg should match h_truefisher to ~1e-5 relative error) or to isolate
    the operator-choice effect from the solver-choice effect in LDS ablations.
    Feasible only at toy scale (explicit D×D Cholesky).
    """
    g_test, grad_cache, prob_cache, train_infos = build_toy_policy_fisher_inputs(
        model, train_examples, test_example, beta=beta, ref_model=ref_model,
    )
    tf = TrueFisherInfluence(
        train_infos,
        grad_cache=grad_cache,
        prob_cache=prob_cache,
        lambda_damp=lambda_damp,
    )
    scores_np = tf.compute_all_scores({"grad": g_test})
    return torch.from_numpy(scores_np), {"solver": "cholesky", "D": int(g_test.numel())}


def compute_toy_cg_influence_historical(
    model_template: nn.Module,
    *,
    checkpoints: dict[int, dict],
    train_history: Sequence[dict],
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    learning_rate: float,
    lambda_damp: float = 1.0,
    cg_iters: int = 50,
    cg_tol: float = 1e-6,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
    historical_weight_mode: ToyHistoricalWeightMode | str = ToyHistoricalWeightMode.ALL_SAMPLES,
    progress_every: int = 100,
) -> tuple[torch.Tensor, dict]:
    """
    Trajectory-summed CG influence.

    At each step in `train_history`, load the pre-step checkpoint, build the
    policy-Fisher FVP (subject to `historical_weight_mode`), solve h via CG at
    that checkpoint, score each train example, scale by learning rate, and
    accumulate:

        IF_i = Σ_c lr · w_i^(c) · g_train_i^(c)^T h_c

    `historical_weight_mode`:
      - ACTIVE_ONLY: at each step, only the active train example contributes
        to both the Fisher batch and the score sum (cheap: O(n_y) backward
        passes per step instead of O(n_train · n_y)).
      - ALL_SAMPLES: every train example contributes to both at every step
        (expensive but matches the original PBRF derivation more literally).
    """
    historical_weight_mode = ToyHistoricalWeightMode.parse(historical_weight_mode)
    n_train = len(train_examples)
    name_to_idx = {ex.name: i for i, ex in enumerate(train_examples)}

    total_scores = torch.zeros(n_train, dtype=torch.float32)
    per_step_residuals: list[float | None] = []
    n_converged = 0

    for step_idx, row in enumerate(train_history):
        step = int(row["step"])
        pre_step = step - 1
        if pre_step not in checkpoints:
            raise ValueError(
                f"Trajectory CG needs checkpoint {pre_step}, but it was not saved."
            )

        checkpoint_model = clone_toy_model(model_template)
        checkpoint_model.load_state_dict(checkpoints[pre_step])

        if historical_weight_mode == ToyHistoricalWeightMode.ACTIVE_ONLY:
            active_name = str(row["example_name"])
            active_idx = name_to_idx[active_name]
            sub_examples = [train_examples[active_idx]]
        else:
            sub_examples = list(train_examples)

        step_scores, cg_info = compute_toy_cg_influence(
            checkpoint_model,
            train_examples=sub_examples,
            test_example=test_example,
            lambda_damp=lambda_damp,
            cg_iters=cg_iters,
            cg_tol=cg_tol,
            beta=beta,
            ref_model=ref_model,
        )
        if cg_info.get("converged"):
            n_converged += 1
        per_step_residuals.append(cg_info.get("final_residual"))

        if historical_weight_mode == ToyHistoricalWeightMode.ACTIVE_ONLY:
            total_scores[active_idx] = total_scores[active_idx] + learning_rate * step_scores[0]
        else:
            total_scores = total_scores + learning_rate * step_scores

        if progress_every and step_idx > 0 and step_idx % progress_every == 0:
            print(f"      CG trajectory step {step_idx}/{len(train_history)} "
                  f"(converged so far: {n_converged}/{step_idx})")

    info = {
        "n_steps": len(train_history),
        "n_converged": n_converged,
        "per_step_final_residuals": per_step_residuals,
        "historical_weight_mode": historical_weight_mode.value,
    }
    return total_scores, info


def compute_toy_firstorder_influence_historical(
    model_template: nn.Module,
    *,
    checkpoints: dict[int, dict],
    train_history: Sequence[dict],
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    learning_rate: float,
    precond_checkpoints: dict[int, torch.Tensor] | None = None,
    beta: float = 0.0,
    ref_model: nn.Module | None = None,
    historical_weight_mode: ToyHistoricalWeightMode | str = ToyHistoricalWeightMode.ALL_SAMPLES,
    progress_every: int = 100,
) -> tuple[torch.Tensor, dict]:
    """
    Trajectory-summed first-order (TracIn) influence — NO Fisher inverse.

    Mirrors `compute_toy_cg_influence_historical` step-for-step, but replaces the
    per-step CG solve (F+λI)h = g_test with a diagonal preconditioner:

        IF_i = Σ_c lr · w_i^(c) · g_train_i^(c)^T (P_c ⊙ g_test^(c))

    `precond_checkpoints`:
      - None  → P_c = I, i.e. plain TracIn dot-product (`dot`): the SGD-step
        first-order effect, λ→∞ limit of CG.
      - dict  → P_c = 1/(√v̂_c+ε), the Adam diagonal actually applied at step c
        (`tracin-adam`): the faithful first-order effect of one AdamW step.

    `historical_weight_mode` matches the CG historical scorer: ACTIVE_ONLY scores
    only the example used at each step; ALL_SAMPLES scores every train example.
    """
    historical_weight_mode = ToyHistoricalWeightMode.parse(historical_weight_mode)
    n_train = len(train_examples)
    name_to_idx = {ex.name: i for i, ex in enumerate(train_examples)}

    total_scores = torch.zeros(n_train, dtype=torch.float32)
    n_precond = 0

    for step_idx, row in enumerate(train_history):
        step = int(row["step"])
        pre_step = step - 1
        if pre_step not in checkpoints:
            raise ValueError(
                f"Trajectory first-order needs checkpoint {pre_step}, but it was not saved."
            )

        checkpoint_model = clone_toy_model(model_template)
        checkpoint_model.load_state_dict(checkpoints[pre_step])

        if historical_weight_mode == ToyHistoricalWeightMode.ACTIVE_ONLY:
            active_name = str(row["example_name"])
            active_idx = name_to_idx[active_name]
            sub_examples = [train_examples[active_idx]]
        else:
            sub_examples = list(train_examples)

        g_test, _grad_cache, _prob_cache, train_infos = build_toy_policy_fisher_inputs(
            checkpoint_model, sub_examples, test_example, beta=beta, ref_model=ref_model,
        )
        gt = g_test.detach().to(dtype=torch.float32)
        # P_c is keyed by `step` (the update θ_{step-1}→θ_step); g_test is at θ_{step-1}.
        P = precond_checkpoints.get(step) if precond_checkpoints is not None else None
        if P is not None:
            P = P.detach().to(dtype=torch.float32, device=gt.device)
            if P.numel() != gt.numel():
                raise ValueError(
                    f"Adam preconditioner at step {step} ({P.numel()}) != grad dim ({gt.numel()})"
                )
            h = P * gt
            n_precond += 1
        else:
            h = gt
        step_scores = torch.stack([
            torch.dot(ti["grad"].detach().to(dtype=torch.float32, device=gt.device), h)
            for ti in train_infos
        ])

        if historical_weight_mode == ToyHistoricalWeightMode.ACTIVE_ONLY:
            total_scores[active_idx] = total_scores[active_idx] + learning_rate * step_scores[0]
        else:
            total_scores = total_scores + learning_rate * step_scores

        if progress_every and step_idx > 0 and step_idx % progress_every == 0:
            print(f"      TracIn trajectory step {step_idx}/{len(train_history)}")

    info = {
        "method": "tracin-adam" if precond_checkpoints is not None else "dot",
        "n_steps": len(train_history),
        "n_steps_preconditioned": n_precond,
        "historical_weight_mode": historical_weight_mode.value,
    }
    return total_scores, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=200) 
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--m-subsets", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000, help="Total GRPO steps for full model")
    parser.add_argument("--subset-steps", type=int, default=1000, help="GRPO steps for subset models")
    parser.add_argument(
        "--lds-mode",
        choices=["from-scratch", "continuation"],
        default="from-scratch",
        help=(
            "How subset models are produced for LDS verification. "
            "`from-scratch` (default): train each subset model from π_ref for "
            "--subset-steps, like the original LDS protocol. "
            "`continuation`: start each subset model from the full-training "
            "endpoint θ_T and continue training on the subset for "
            "--lds-cont-steps at --lds-cont-lr. Stays in the local regime "
            "where the IF linearization is most accurate."
        ),
    )
    parser.add_argument(
        "--lds-cont-steps",
        type=int,
        default=100,
        help="(--lds-mode continuation) Continuation training steps from θ_T per subset.",
    )
    parser.add_argument(
        "--lds-cont-lr",
        type=float,
        default=None,
        help=(
            "(--lds-mode continuation) Continuation learning rate. "
            "Default: --lr / 10 (small to stay close to θ_T's local linearization "
            "and reduce GRPO sampling noise)."
        ),
    )
    parser.add_argument(
        "--extremes-test",
        action="store_true",
        help=(
            "Before the full M-subset LDS run, do a 3-point sanity check on each "
            "test example: train (or continue) on the top-α subset by IF, the "
            "bottom-α, and a random α-sized subset. Print rewards; a working IF "
            "should yield R_pos > R_rand > R_neg."
        ),
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=0.5,
        help="α (subset fraction) used for the --extremes-test top/bottom/random subsets.",
    )
    parser.add_argument(
        "--per-example-test",
        action="store_true",
        help=(
            "Run a per-example Spearman test: for each training example i, train on the "
            "singleton subset {z_i} alone (same --lds-mode protocol) and correlate IF "
            "score φ_i with resulting test reward. Tests *ranking* directly without LDS's "
            "linear-additivity-over-subsets assumption. Cost: n_train singleton trainings."
        ),
    )
    parser.add_argument(
        "--natural-gradient-cont",
        action="store_true",
        help=(
            "Use natural-gradient steps θ - lr · (F + λI)^{-1} · ∇L during continuation "
            "training (replaces Adam/SGD). Eliminates the Adam-vs-F^{-1} optimizer "
            "mismatch — under this mode the PBRF derivation predicts per-example "
            "Spearman ≈ 1 at K_cont=1, since ΔR_i = lr · φ_i exactly to first order."
        ),
    )
    parser.add_argument(
        "--natural-gradient-cont-lambda",
        type=float,
        default=None,
        help=(
            "λ damping for the natural-gradient F^{-1} solve. Defaults to --lambda-damp. "
            "Should typically match the λ used when computing the IF for the "
            "cleanest PBRF-consistency test."
        ),
    )
    parser.add_argument(
        "--natural-gradient-cont-fisher-batch",
        choices=["singleton", "full-train"],
        default="singleton",
        help=(
            "Which Fisher batch the natural-gradient step uses at each continuation "
            "step. `singleton`: just the current example (cheap, standard minibatch NG). "
            "`full-train`: the entire training set (matches the IF formula's F operator "
            "exactly — Spearman should ≈ 1 at K_cont=1, but rebuilds the full grad/prob "
            "cache every step so it's O(n_train) times slower per step)."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-damp", type=float, default=0.1)
    parser.add_argument(
        "--cg-iters",
        type=int,
        default=50,
        help="(--if-calculation cg) Max Conjugate Gradient iterations.",
    )
    parser.add_argument(
        "--cg-tol",
        type=float,
        default=1e-6,
        help="(--if-calculation cg) CG relative residual tolerance ||r||/||g_test||.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help=(
            "KL temperature β. Used both as the GRPO training KL coefficient "
            "(toward π_ref = the shared initial model) and as the surrogate "
            "IF formula's β in r/β. The surrogate derivation requires these to match."
        ),
    )
    parser.add_argument("--no-adam", action="store_true")
    parser.add_argument(
        "--historical-weight-mode",
        choices=[mode.value for mode in ToyHistoricalWeightMode],
        default=ToyHistoricalWeightMode.ALL_SAMPLES.value,
        help=(
            "How historical trajectory Fisher weights train examples at each step: "
            "`active_only` keeps only the example actually used at that step; "
            "`all_samples` includes every train example at every step."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="outputs/lds_toy_grpo", help="Directory to save results.")
    parser.add_argument(
        "--dataset",
        choices=["iid", "clustered"],
        default="iid",
        help=(
            "iid: original `sign(z1+z2+z3)` rule with redundant examples. "
            "clustered: non-iid one-hot-signature clusters with sparse per-cluster examples — "
            "designed to make saturation a failure mode for single-checkpoint IF."
        ),
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=15,
        help="(--dataset clustered) Number of distinct clusters. Must satisfy n_clusters <= --input-dim.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=20,
        help="(--dataset clustered) Input dimensionality; needs >= n_clusters "
             "(e.g. 65 for 60 clusters, matching the if_pruning configs).",
    )
    parser.add_argument(
        "--per-cluster",
        type=int,
        default=3,
        help="(--dataset clustered) Training examples per cluster. P(cluster wiped by 50%% mask) = 0.5^per_cluster.",
    )
    parser.add_argument(
        "--test-cluster-ids",
        type=str,
        default=None,
        help="(--dataset clustered) Comma-separated cluster ids to test on (default: 5 evenly-spaced strong-signal clusters).",
    )
    parser.add_argument(
        "--cluster-signal",
        type=float,
        default=3.0,
        help="(--dataset clustered) Signal strength for the primary (`easy`) clusters.",
    )
    parser.add_argument(
        "--alt-cluster-signal",
        type=float,
        default=None,
        help=(
            "(--dataset clustered) Signal strength for the last --n-alt-clusters clusters. "
            "Set this lower than --cluster-signal to create `hard` clusters that the model "
            "cannot fully saturate on — they act as unsaturated distractors for the IF."
        ),
    )
    parser.add_argument(
        "--n-alt-clusters",
        type=int,
        default=0,
        help="(--dataset clustered) Number of trailing clusters that use --alt-cluster-signal.",
    )
    parser.add_argument(
        "--cluster-target-mode",
        choices=["parity", "random"],
        default="parity",
        help=(
            "(--dataset clustered) How per-cluster targets are assigned. "
            "`parity` (current): cluster k → (1,0) if k even, (0,1) if k odd — "
            "lets the model GENERALIZE across same-parity clusters, so saturation tracks redundancy. "
            "`random`: each cluster gets a uniformly random target from {(0,0),(0,1),(1,0),(1,1)} — "
            "forces per-cluster memorization, decorrelates saturation from redundancy. "
            "Use `random` to expose the IF saturation failure mode."
        ),
    )
    parser.add_argument(
        "--lds-cache-dir",
        type=str,
        default="outputs/lds_toy_grpo/_lds_cache",
        help="Where to cache actual_rewards / subset_masks. Keyed by the args that affect them (not by --if-calculation), so different IF methods reuse the same LDS computation.",
    )
    parser.add_argument("--no-lds-cache", action="store_true", help="Disable LDS cache (always retrain subset models).")
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip the post-training saturation probes and gradient-norm trajectory diagnostics.",
    )
    parser.add_argument(
        "--if-calculation",
        choices=["historical", "historical-last", "preference-styled", "surrogate", "cg", "cg-last", "true-fisher-last", "ekfac-last", "tracin-adam", "dot"],
        default="historical",
        help=(
            "Method to use for influence calculation. "
            "`tracin-adam` sums the first-order TracIn IF over the full trajectory "
            "using the Adam diagonal P_c=1/(√v̂+ε) actually applied at each step "
            "(no Fisher inverse) — the faithful first-order effect of each AdamW "
            "step, directly comparable to `historical`; "
            "`dot` is the same trajectory sum with P=I (plain SGD-step TracIn, the "
            "λ→∞ limit of `cg`); "
            "`historical` sums per-step IF over the full trajectory; "
            "`historical-last` uses the historical gradient (GRPO loss) "
            "evaluated only at the final checkpoint — apples-to-apples comparison "
            "for `surrogate`, which also uses only the last checkpoint; "
            "`cg` solves (F+λI) h = g_test by Conjugate Gradient against the "
            "true expected policy Fisher (FVP-only, F never materialized) at "
            "every training checkpoint and sums the per-step IFs over the "
            "trajectory (mirrors `historical`); "
            "`cg-last` is the single-final-checkpoint variant of `cg` (mirrors "
            "`historical-last`); "
            "`true-fisher-last` solves the same operator as `cg-last` but with "
            "direct Cholesky inversion of the explicit D×D Fisher — use as the "
            "reference solver to validate CG, or to ablate operator vs solver "
            "choices against `historical-last`."
        ),
    )
    args = parser.parse_args()

    use_adam = not args.no_adam
    hist_weight_mode = ToyHistoricalWeightMode.parse(args.historical_weight_mode)
    
    # Setup output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.if_calculation.replace('-', '_')}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    # Save config
    config = vars(args)
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    cluster_assignment_by_train_idx = None  # for per-subset coverage tracking
    if args.dataset == "clustered":
        test_cluster_ids = None
        if args.test_cluster_ids is not None:
            test_cluster_ids = [int(s) for s in args.test_cluster_ids.split(",") if s.strip()]

        # Build per-cluster signal: first (n_clusters - n_alt_clusters) get cluster_signal,
        # last n_alt_clusters get alt_cluster_signal (if specified).
        n_alt = max(0, min(args.n_alt_clusters, args.n_clusters))
        alt_signal = args.alt_cluster_signal if args.alt_cluster_signal is not None else args.cluster_signal
        signal_per_cluster = [args.cluster_signal] * (args.n_clusters - n_alt) + [alt_signal] * n_alt

        print(
            f"Generating clustered dataset: {args.n_clusters} clusters × {args.per_cluster} examples "
            f"(target_mode={args.cluster_target_mode})..."
        )
        train_examples, test_examples, test_cluster_ids, signal_per_cluster, cluster_targets = generate_clustered_dataset(
            n_clusters=args.n_clusters,
            per_cluster=args.per_cluster,
            test_cluster_ids=test_cluster_ids,
            input_dim=args.input_dim,
            signal_per_cluster=signal_per_cluster,
            cluster_signal=args.cluster_signal,
            target_mode=args.cluster_target_mode,
            seed=args.seed,
        )
        # Override n_train/n_test to actual sizes so the rest of the script is consistent.
        args.n_train = len(train_examples)
        args.n_test = len(test_examples)
        print(
            f"  Train: {args.n_train} examples; Test: {args.n_test} examples from clusters {test_cluster_ids}"
        )
        if n_alt > 0:
            print(
                f"  Heterogeneous signal: first {args.n_clusters - n_alt} clusters @ signal={args.cluster_signal}, "
                f"last {n_alt} clusters @ signal={alt_signal}"
            )
        if args.cluster_target_mode == "random":
            print(f"  Target assignment (cluster → target):")
            for k_idx, k in enumerate(range(args.n_clusters)):
                marker = " ← test" if k in test_cluster_ids else ""
                print(f"    c{k}: {cluster_targets[k]}{marker}")
        wipe_prob = 0.5 ** args.per_cluster
        print(
            f"  With 50% subset mask, P(cluster fully wiped) = 0.5^{args.per_cluster} = {wipe_prob:.4f}"
        )

        # Map each training example index → its cluster id for per-subset coverage analysis.
        cluster_assignment_by_train_idx = [int(i % args.n_clusters) for i in range(args.n_train)]
    else:
        print(f"Generating dataset N={args.n_train}...")
        all_examples = generate_dataset(n=args.n_train + 200, seed=args.seed)
        train_examples = all_examples[:args.n_train]

        hard_test_examples = []
        for ex in all_examples[args.n_train:]:
            val = abs(ex.z[0] + ex.z[1] + ex.z[2])
            if val < 0.3:
                hard_test_examples.append(ex)
            if len(hard_test_examples) >= args.n_test:
                break
        test_examples = hard_test_examples
        print(f"Selected {len(test_examples)} hard test examples.")

    # Build the shared π_ref once. Both the full and subset trainings start from
    # this state and KL-regularize toward it, so the surrogate IF's π_ref is
    # well-defined and identical to the training anchor.
    torch.manual_seed(args.seed)
    ref_model = ToyAutoregressiveMLP(
        input_dim=int(train_examples[0].z_tensor().shape[-1]), hidden_dim=args.hidden_dim)
    for m in ref_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    ref_model.eval()

    print(f"Training full model with history ({'Adam' if use_adam else 'SGD'}) for {args.steps} steps (β={args.beta})...")
    train_result = train_model_with_history(
        train_examples,
        steps=args.steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        use_adam=use_adam,
        beta=args.beta,
        ref_model=ref_model,
    )
    full_model = train_result["model"]
    
    # Accuracy check
    correct = 0
    for ex in train_examples:
        if exact_expected_reward(full_model, ex) > 0.5:
            correct += 1
    accuracy = 100 * correct / args.n_train
    print(f"Full model training accuracy: {correct}/{args.n_train} ({accuracy:.1f}%)")

    if not args.no_diagnostics:
        diagnostics = run_saturation_diagnostics(
            full_model,
            train_result["checkpoints"],
            train_examples,
            ref_model,
            args,
        )
        with (run_dir / "diagnostics.json").open("w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"Diagnostics saved to {run_dir / 'diagnostics.json'}")

    print(f"Computing Influence Functions (method={args.if_calculation})...")
    test_ifs = []
    ekfac_factors_cache: dict = {}  # ekfac-last: one fit per full_model, shared across test examples
    # 'historical' and 'preference-styled' use the full trajectory.
    # 'surrogate' uses only the final checkpoint.
    print(f"  Method: {args.if_calculation}...")
    # historical-last reuses the historical method's gradient logic, but only
    # evaluates at the final checkpoint (same single-step trick as surrogate).
    last_checkpoint_only = args.if_calculation in (
        "surrogate", "historical-last", "cg-last", "true-fisher-last", "ekfac-last",
    )
    method_for_if = "historical" if args.if_calculation == "historical-last" else args.if_calculation

    for i, test_ex in enumerate(test_examples):
        if last_checkpoint_only:
            history_to_use = [{"step": args.steps + 1, "example_name": train_examples[0].name}]
            mode_to_use = ToyHistoricalWeightMode.ALL_SAMPLES
            lr_to_use = 1.0
            print(f"    Evaluating at final checkpoint (step {args.steps})...")
        else:
            history_to_use = train_result["history"]
            mode_to_use = hist_weight_mode
            lr_to_use = args.lr
            print(f"    Evaluating over full trajectory (mode={mode_to_use.value})...")

        if args.if_calculation == "cg-last":
            print(f"    CG solve at final checkpoint (max {args.cg_iters} iters, tol={args.cg_tol:.0e})...")
            cg_scores, cg_info = compute_toy_cg_influence(
                full_model,
                train_examples=train_examples,
                test_example=test_ex,
                lambda_damp=args.lambda_damp,
                cg_iters=args.cg_iters,
                cg_tol=args.cg_tol,
                beta=args.beta,
                ref_model=ref_model,
            )
            scores = cg_scores.numpy()
            status = "converged" if cg_info.get("converged") else "MAX-ITERS (did not converge)"
            print(
                f"    CG {status} after {cg_info.get('n_iters', '?')} iters; "
                f"final residual={cg_info.get('final_residual', float('nan')):.2e} "
                f"(tol={args.cg_tol:.0e}); "
                f"g_test·h={cg_info.get('g_test_dot_h', float('nan')):.4e} (>=0 if SPD)"
            )
        elif args.if_calculation == "true-fisher-last":
            print(f"    True policy Fisher (direct Cholesky) at final checkpoint, λ={args.lambda_damp}...")
            tf_scores, tf_info = compute_toy_true_fisher_influence(
                full_model,
                train_examples=train_examples,
                test_example=test_ex,
                lambda_damp=args.lambda_damp,
                beta=args.beta,
                ref_model=ref_model,
            )
            scores = tf_scores.numpy()
            print(f"    Direct Cholesky on D={tf_info.get('D', '?')} Fisher.")
        elif args.if_calculation == "ekfac-last":
            print(f"    EK-FAC (Kronecker-factored damped Fisher) at final checkpoint, λ={args.lambda_damp}...")
            ek_scores, ek_info = compute_toy_ekfac_influence(
                full_model,
                train_examples=train_examples,
                test_example=test_ex,
                lambda_damp=args.lambda_damp,
                beta=args.beta,
                ref_model=ref_model,
                _factors_cache=ekfac_factors_cache,
            )
            scores = ek_scores.numpy()
            print(f"    EK-FAC: {ek_info.get('n_blocks', '?')} Linear blocks, D={ek_info.get('D', '?')}.")
        elif args.if_calculation == "cg":
            print(
                f"    CG trajectory over {len(train_result['history'])} steps "
                f"(mode={hist_weight_mode.value}, max {args.cg_iters} iters/step, tol={args.cg_tol:.0e})..."
            )
            cg_scores, cg_info = compute_toy_cg_influence_historical(
                full_model,
                checkpoints=train_result["checkpoints"],
                train_history=train_result["history"],
                train_examples=train_examples,
                test_example=test_ex,
                learning_rate=args.lr,
                lambda_damp=args.lambda_damp,
                cg_iters=args.cg_iters,
                cg_tol=args.cg_tol,
                beta=args.beta,
                ref_model=ref_model,
                historical_weight_mode=hist_weight_mode,
            )
            scores = cg_scores.numpy()
            n_steps = cg_info.get("n_steps", 0)
            n_conv = cg_info.get("n_converged", 0)
            print(f"    CG trajectory done: {n_conv}/{n_steps} per-step solves converged within tol.")
        elif args.if_calculation in ("tracin-adam", "dot"):
            precond = (
                train_result.get("precond_checkpoints")
                if args.if_calculation == "tracin-adam" else None
            )
            if args.if_calculation == "tracin-adam" and not precond:
                print(
                    "    WARNING: tracin-adam requested but no Adam preconditioner was "
                    "saved (--no-adam?). Falling back to plain `dot` (P=I)."
                )
                precond = None
            print(
                f"    {args.if_calculation} trajectory over {len(train_result['history'])} "
                f"steps (mode={hist_weight_mode.value})..."
            )
            fo_scores, fo_info = compute_toy_firstorder_influence_historical(
                full_model,
                checkpoints=train_result["checkpoints"],
                train_history=train_result["history"],
                train_examples=train_examples,
                test_example=test_ex,
                learning_rate=args.lr,
                precond_checkpoints=precond,
                beta=args.beta,
                ref_model=ref_model,
                historical_weight_mode=hist_weight_mode,
            )
            scores = fo_scores.numpy()
            print(
                f"    {fo_info['method']} done: {fo_info['n_steps_preconditioned']}/"
                f"{fo_info['n_steps']} steps Adam-preconditioned."
            )
        else:
            hist_inf = compute_toy_historical_fisher_influence(
                full_model,
                checkpoints=train_result["checkpoints"],
                train_history=history_to_use,
                train_examples=train_examples,
                test_example=test_ex,
                learning_rate=lr_to_use,
                lambda_damp=args.lambda_damp,
                rollout_mode=ToyRolloutMode.EXHAUSTIVE,
                historical_weight_mode=mode_to_use,
                method=method_for_if,
                surrogate_beta=args.beta
            )
            scores = hist_inf["repo_scores"]
            if args.if_calculation == "surrogate":
                # Surrogate emits 5 entries per example (1 numerator + 4 rollout Fisher
                # contributions). The numerator is at every 5th position.
                scores = scores.reshape(args.n_train, 5)[:, 0]

        print(f"  Test Example {i} IF stats: Mean={scores.mean():.4f}, Std={scores.std():.4f}, Max={scores.max():.4f}")
        test_ifs.append(scores)

    # Wire the toy subset trainer into the generic LDS module. At LLM scale,
    # replace `subset_trainer` with an LLM-aware continuation/from-scratch trainer
    # that returns per-test-example rewards in the same shape.
    def subset_trainer(subset_indices, seed):
        subset_train = [train_examples[j] for j in subset_indices]
        sub_res = _train_subset_for_lds(
            args,
            ref_model=ref_model,
            full_model=full_model,
            subset_train=subset_train,
            seed=seed,
            use_adam=use_adam,
            full_train_examples=train_examples,
        )
        return np.array(
            [exact_expected_reward(sub_res["model"], te) for te in test_examples],
            dtype=np.float64,
        )

    if args.extremes_test:
        cont_str = f", K_cont={args.lds_cont_steps}" if args.lds_mode == "continuation" else ""
        k_preview = max(1, int(round(args.subset_fraction * args.n_train)))
        print(
            f"\nExtremes test (top/random/bottom α={args.subset_fraction} → "
            f"k={k_preview} examples per subset, mode={args.lds_mode}{cont_str}):"
        )
        extremes_rows = run_extremes_test(
            test_ifs,
            subset_trainer,
            n_train=args.n_train,
            n_test=args.n_test,
            subset_fraction=args.subset_fraction,
            seed=args.seed,
        )
        with (run_dir / "extremes_test.json").open("w") as f:
            json.dump({
                "subset_fraction": args.subset_fraction,
                "k": extremes_rows[0].k if extremes_rows else None,
                "lds_mode": args.lds_mode,
                "lds_cont_steps": args.lds_cont_steps if args.lds_mode == "continuation" else None,
                "lds_cont_lr": (args.lds_cont_lr if args.lds_cont_lr is not None else args.lr / 10.0) if args.lds_mode == "continuation" else None,
                "results": [r.__dict__ for r in extremes_rows],
            }, f, indent=2)
        print(f"  Extremes test saved to {run_dir / 'extremes_test.json'}")

    if args.per_example_test:
        cont_str = f", K_cont={args.lds_cont_steps}" if args.lds_mode == "continuation" else ""
        print(
            f"\nPer-example test (singleton trainings × {args.n_train} examples, "
            f"mode={args.lds_mode}{cont_str}):"
        )
        pe = run_per_example_test(
            test_ifs,
            subset_trainer,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
        )
        for row in pe.per_test:
            i = row["test_idx"]
            if row["correlation"] is None:
                print(
                    f"  Test {i}: Spearman undefined "
                    f"(actual_std={row['actual_std']:.4e}, if_std={row['if_std']:.4e})"
                )
            else:
                print(
                    f"  Test {i}: Spearman(IF, ΔR_singleton)={row['correlation']:.4f}, "
                    f"p={row['p_value']:.2e}"
                )
        print(f"  Average per-example Spearman: {pe.average_correlation:.4f}")
        with (run_dir / "per_example_test.json").open("w") as f:
            json.dump({
                "average_correlation": pe.average_correlation,
                "per_test": pe.per_test,
                "lds_mode": args.lds_mode,
                "lds_cont_steps": args.lds_cont_steps if args.lds_mode == "continuation" else None,
                "subset_steps": args.subset_steps if args.lds_mode == "from-scratch" else None,
            }, f, indent=2)
        print(f"  Per-example test saved to {run_dir / 'per_example_test.json'}")

    cont_str = (
        f" (continuation: K_cont={args.lds_cont_steps}, "
        f"lr={args.lds_cont_lr if args.lds_cont_lr is not None else args.lr / 10.0:.2e})"
        if args.lds_mode == "continuation" else ""
    )
    print(f"Training {args.m_subsets} subset models for LDS verification [mode={args.lds_mode}]{cont_str}...")

    cache_path = None
    if not args.no_lds_cache:
        cache_dir = Path(args.lds_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"lds_{_lds_cache_key(args)}.npz"

    start_time = time.time()
    # Match the predictor's visit-weighting to the subset_trainer's actual step count
    # — continuation does K_cont steps round-robin on the subset; from-scratch does
    # subset_steps. This makes the LDS predictor literally consistent with the PBRF
    # formula instead of assuming every masked example was visited exactly once.
    n_training_steps = (
        args.lds_cont_steps if args.lds_mode == "continuation" else args.subset_steps
    )
    lds = run_lds(
        test_ifs,
        subset_trainer,
        n_train=args.n_train,
        n_test=args.n_test,
        m_subsets=args.m_subsets,
        seed=args.seed,
        cache_path=cache_path,
        n_training_steps=n_training_steps,
    )
    actual_rewards = lds.actual_rewards
    predicted_rewards = lds.predicted_rewards
    subset_masks = lds.subset_masks
    test_results = lds.per_test
    avg_corr = lds.average_correlation
    cache_hit = lds.cache_hit

    print(f"\nLDS Evaluation Results for {args.if_calculation} Influence:")
    for row in test_results:
        i = row["test_idx"]
        if row["correlation"] is None:
            print(
                f"  Test Example {i}: Undefined variance "
                f"(Actual Std={row['actual_std']:.4f}, Pred Std={row['pred_std']:.4f})."
            )
        else:
            print(f"  Test Example {i}: Corr={row['correlation']:.4f}, p={row['p_value']:.4e}")
    corrs = [r["correlation"] for r in test_results if r["correlation"] is not None]
    if corrs:
        print(f"\nAverage Trajectory LDS Correlation: {avg_corr:.4f}")

    # Save quantitative results
    final_results = {
        "accuracy": accuracy,
        "average_correlation": float(avg_corr),
        "test_results": test_results,
        "total_time": time.time() - start_time,
        "lds_cache_hit": cache_hit,
        "lds_cache_key": _lds_cache_key(args),
    }
    with (run_dir / "results.json").open("w") as f:
        json.dump(final_results, f, indent=2)

    # Save qualitative analysis
    qual_path = run_dir / "qualitative_analysis.txt"
    with qual_path.open("w") as f:
        if corrs:
            for i in range(args.n_test):
                f.write("\n" + "="*50 + "\n")
                f.write(f"QUALITATIVE ANALYSIS: Test Example {i}\n")
                target_ex = test_examples[i]
                target_sum = sum(target_ex.z[:3])
                f.write(f"Test Input z1+z2+z3: {target_sum:.4f}\n")
                f.write(f"Test Target: {target_ex.target}\n")
                f.write("="*50 + "\n")
                
                scores = test_ifs[i]
                sorted_indices = np.argsort(scores)
                
                f.write("\nTop 5 HARMFUL Examples (Lowest IF):\n")
                for idx in sorted_indices[:5]:
                    ex = train_examples[idx]
                    rule_sum = sum(ex.z[:3])
                    f.write(f"  {ex.name}: Score={scores[idx]:.4f}, z1+z2+z3={rule_sum:.4f}, Target={ex.target}\n")
                    
                f.write("\nTop 5 HELPFUL Examples (Highest IF):\n")
                for idx in sorted_indices[-5:][::-1]:
                    ex = train_examples[idx]
                    rule_sum = sum(ex.z[:3])
                    f.write(f"  {ex.name}: Score={scores[idx]:.4f}, z1+z2+z3={rule_sum:.4f}, Target={ex.target}\n")
            
            print(f"\nQualitative analysis saved to {qual_path}")
            # Print Test Example 0 to console as well
            print("\n" + "="*50)
            print(f"QUALITATIVE ANALYSIS: Test Example 0")
            target_ex = test_examples[0]
            target_sum = sum(target_ex.z[:3])
            print(f"Test Input z1+z2+z3: {target_sum:.4f}")
            print(f"Test Target: {target_ex.target}")
            print("="*50)
            
            scores = test_ifs[0]
            sorted_indices = np.argsort(scores)
            print("\nTop 5 HELPFUL Examples (Highest IF):")
            for idx in sorted_indices[-5:][::-1]:
                ex = train_examples[idx]
                rule_sum = sum(ex.z[:3])
                print(f"  {ex.name}: Score={scores[idx]:.4f}, z1+z2+z3={rule_sum:.4f}, Target={ex.target}")
        else:
            f.write("\nNo valid correlations computed.")

    # Save raw rewards for plotting. For clustered datasets we also record, per subset,
    # how many training examples of each test cluster survived the mask — so you can
    # tell whether a high-error subset is one that wiped a test cluster.
    test_cluster_id_per_test = None
    if args.dataset == "clustered" and cluster_assignment_by_train_idx is not None:
        # Recover test cluster ids from test example names ("test_c<k>").
        test_cluster_id_per_test = [
            int(ex.name.split("_c")[1]) for ex in test_examples
        ]

    with (run_dir / "rewards.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["subset_idx", "subset_size"]
        for i in range(args.n_test):
            header.extend([f"actual_reward_{i}", f"predicted_reward_{i}"])
        if test_cluster_id_per_test is not None:
            for i in range(args.n_test):
                header.append(f"n_train_in_cluster_{test_cluster_id_per_test[i]}")
        writer.writerow(header)

        cluster_assignments_t = (
            torch.tensor(cluster_assignment_by_train_idx, dtype=torch.long)
            if cluster_assignment_by_train_idx is not None else None
        )
        for j in range(args.m_subsets):
            mask = subset_masks[j]
            row = [j, int(mask.sum().item())]
            for i in range(args.n_test):
                row.extend([actual_rewards[i, j], predicted_rewards[i, j]])
            if cluster_assignments_t is not None and test_cluster_id_per_test is not None:
                surviving_clusters = cluster_assignments_t[mask.bool()]
                for k in test_cluster_id_per_test:
                    row.append(int((surviving_clusters == k).sum().item()))
            writer.writerow(row)
    print(f"Raw rewards saved to {run_dir / 'rewards.csv'}")

    if test_cluster_id_per_test is not None:
        # Quick diagnostic: average actual reward per test, conditioned on whether
        # that test's cluster is fully wiped from the subset.
        print("\nCluster-wipe diagnostic (clustered dataset):")
        cluster_assignments_t = torch.tensor(cluster_assignment_by_train_idx, dtype=torch.long)
        for i in range(args.n_test):
            k = test_cluster_id_per_test[i]
            wiped = []
            present = []
            for j in range(args.m_subsets):
                mask = subset_masks[j]
                n_in_cluster = int((cluster_assignments_t[mask.bool()] == k).sum().item())
                if n_in_cluster == 0:
                    wiped.append(actual_rewards[i, j])
                else:
                    present.append(actual_rewards[i, j])
            n_wiped = len(wiped)
            mean_wiped = float(np.mean(wiped)) if wiped else float("nan")
            mean_present = float(np.mean(present)) if present else float("nan")
            print(
                f"  Test {i} (cluster {k}): {n_wiped}/{args.m_subsets} subsets wiped "
                f"this cluster | actual reward: wiped={mean_wiped:.3f}  present={mean_present:.3f}"
            )

    # Generate and save plots
    if corrs:
        fig, axes = plt.subplots(1, args.n_test, figsize=(5 * args.n_test, 5), squeeze=False)
        for i in range(args.n_test):
            ax = axes[0, i]
            ax.scatter(predicted_rewards[i], actual_rewards[i], alpha=0.5)
            ax.set_xlabel("Predicted Reward (IF Sum)")
            ax.set_ylabel("Actual Reward (Subset Model)")
            
            # Clean correlation label for title
            corr_val = next((res["correlation"] for res in test_results if res["test_idx"] == i), 0.0)
            ax.set_title(f"Test Example {i}\nSpearman Corr: {corr_val:.4f}")
            ax.grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plot_path = run_dir / "lds_correlation.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Correlation plots saved to {plot_path}")


if __name__ == "__main__":
    main()
