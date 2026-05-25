# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "scipy",
# ]
# ///

from __future__ import annotations

import argparse
import time
import copy
import hashlib
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence


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
    """Hash of the args that affect actual_rewards / subset_masks (i.e., not the IF method)."""
    relevant = {
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
        "n_clusters": getattr(args, "n_clusters", None) if getattr(args, "dataset", "iid") == "clustered" else None,
        "per_cluster": getattr(args, "per_cluster", None) if getattr(args, "dataset", "iid") == "clustered" else None,
        "test_cluster_ids": getattr(args, "test_cluster_ids", None) if getattr(args, "dataset", "iid") == "clustered" else None,
        "cluster_signal": getattr(args, "cluster_signal", None) if getattr(args, "dataset", "iid") == "clustered" else None,
        "alt_cluster_signal": getattr(args, "alt_cluster_signal", None) if getattr(args, "dataset", "iid") == "clustered" else None,
        "n_alt_clusters": getattr(args, "n_alt_clusters", None) if getattr(args, "dataset", "iid") == "clustered" else None,
        "cluster_target_mode": getattr(args, "cluster_target_mode", None) if getattr(args, "dataset", "iid") == "clustered" else None,
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from influence_rlvr.toy_grpo import (
    ToyGRPOExample,
    ToyRolloutMode,
    clone_toy_model,
    compute_toy_fisher_influence,
    compute_toy_historical_fisher_influence,
    exact_expected_reward,
    train_toy_grpo,
    reward_for_sequences,
    _ALL_TWO_TOKEN_SEQUENCES,
    GradientObjective,
    _toy_objective_and_debug,
    ToyHistoricalWeightMode,
    compute_toy_gradient_bundle
)


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
) -> tuple[list[ToyGRPOExample], list[ToyGRPOExample], list[int], list[float], list[tuple[int, int]]]:
    """
    Non-iid dataset designed to probe saturation failure of single-checkpoint IFs.

    Each cluster k has a one-hot signature in z[3 + k] with strength `signal_per_cluster[k]`,
    and its own target (alternating (1,0)/(0,1) by cluster id parity). Training examples
    are round-robin across clusters. With per_cluster small (e.g. 2-4), a 50% subset mask
    has non-trivial probability of wiping a whole cluster.

    Heterogeneous signals: pass per-cluster strengths so some clusters are "easy"
    (high signal, model saturates) and others are "hard" (low signal, model can't fit).
    Test cluster auto-selection picks from the *strong-signal* clusters by default —
    that's where saturation matters most.

    Returns (train_examples, test_examples, test_cluster_ids, signal_per_cluster).
    """
    if input_dim < 3 + n_clusters:
        raise ValueError(
            f"input_dim ({input_dim}) must be >= 3 + n_clusters ({3 + n_clusters})."
        )
    if signal_per_cluster is None:
        signal_per_cluster = [cluster_signal] * n_clusters
    signal_per_cluster = list(signal_per_cluster)
    if len(signal_per_cluster) != n_clusters:
        raise ValueError(
            f"signal_per_cluster length ({len(signal_per_cluster)}) != n_clusters ({n_clusters})."
        )

    torch.manual_seed(seed)
    n_train = n_clusters * per_cluster
    cluster_assignment = torch.arange(n_train) % n_clusters

    z_train = torch.randn(n_train, input_dim) * background_scale
    for k in range(n_clusters):
        mask = cluster_assignment == k
        z_train[mask, 3 + k] += signal_per_cluster[k]

    if target_mode == "parity":
        cluster_targets = [(1, 0) if k % 2 == 0 else (0, 1) for k in range(n_clusters)]
    elif target_mode == "random":
        target_pool = [(0, 0), (0, 1), (1, 0), (1, 1)]
        gen = torch.Generator().manual_seed(seed + 1000)
        idxs = torch.randint(0, len(target_pool), (n_clusters,), generator=gen).tolist()
        cluster_targets = [target_pool[i] for i in idxs]
    else:
        raise ValueError(f"Unsupported target_mode={target_mode!r}. Use 'parity' or 'random'.")

    targets_train = torch.zeros(n_train, 2, dtype=torch.long)
    for k in range(n_clusters):
        mask = cluster_assignment == k
        targets_train[mask] = torch.tensor(cluster_targets[k])

    train_examples = [
        ToyGRPOExample(
            name=f"c{int(cluster_assignment[i])}_ex{i}",
            z=tuple(z_train[i].tolist()),
            target=tuple(targets_train[i].tolist()),
        )
        for i in range(n_train)
    ]

    if test_cluster_ids is None:
        max_signal = max(signal_per_cluster)
        strong_clusters = [k for k in range(n_clusters) if signal_per_cluster[k] >= max_signal - 1e-9]
        pool = strong_clusters if strong_clusters else list(range(n_clusters))
        step = max(1, len(pool) // 5)
        test_cluster_ids = pool[::step][:5]
    test_cluster_ids = list(test_cluster_ids)
    for k in test_cluster_ids:
        if k < 0 or k >= n_clusters:
            raise ValueError(f"test_cluster_id {k} out of range [0, {n_clusters}).")

    n_test = len(test_cluster_ids)
    z_test = torch.randn(n_test, input_dim) * background_scale
    targets_test = torch.zeros(n_test, 2, dtype=torch.long)
    for i, k in enumerate(test_cluster_ids):
        z_test[i, 3 + k] += signal_per_cluster[k]
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
) -> dict:
    torch.manual_seed(seed)
    model = ToyAutoregressiveMLP(hidden_dim=hidden_dim)
    if ref_model is not None:
        # Start every training run from the same reference state so π_ref is shared
        # across full and subset training (matches the surrogate-IF derivation).
        model.load_state_dict(ref_model.state_dict())
    else:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    checkpoints = {}
    if save_checkpoints:
        checkpoints[0] = copy.deepcopy(model.state_dict())

    history = []
    for step in range(1, steps + 1):
        example = dataset[(step - 1) % len(dataset)]
        old_model = clone_toy_model(model)
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
        objective.backward()
        optimizer.step()
        
        history.append({
            "step": step,
            "example_name": example.name,
            "loss": float(objective.item())
        })
        
        if save_checkpoints:
            checkpoints[step] = copy.deepcopy(model.state_dict())
            
    return {
        "model": model,
        "history": history,
        "checkpoints": checkpoints
    }


def compute_toy_preference_influence(
    model: nn.Module,
    ref_model: nn.Module,
    train_examples: Sequence[ToyGRPOExample],
    test_example: ToyGRPOExample,
    lambda_damp: float = 1.0,
) -> torch.Tensor:
    model.eval()
    ref_model.eval()
    
    # 1. Compute test gradient g_test
    test_bundle = compute_toy_gradient_bundle(
        model,
        test_example,
        G=4,
        rollout_mode=ToyRolloutMode.EXHAUSTIVE,
        objective_mode=GradientObjective.EXPECTED_REWARD_PG,
    )
    g_test = test_bundle["grad"].to(dtype=torch.float32)

    # 2. Compute training gradients g_train and geometry features
    train_grads = []
    geometry_features = []
    
    responses = _ALL_TWO_TOKEN_SEQUENCES.to(model.device)
    N = responses.shape[0]
    
    for ex in train_examples:
        z = ex.z_tensor(device=model.device)
        target = ex.target_tensor(device=model.device)
        
        # Ground truth rewards
        with torch.no_grad():
            r_phi = reward_for_sequences(responses, target)
            mu_phi = r_phi.mean()
            sigma_phi = r_phi.std(unbiased=False).clamp(min=1e-8)
            r_phi_prime = (r_phi - mu_phi) / sigma_phi
        
        # Implicit rewards
        lp = model.sequence_log_probs(z, responses)
        with torch.no_grad():
            lp_ref = ref_model.sequence_log_probs(z.to(ref_model.device), responses.to(ref_model.device)).to(model.device)
            r_hat = lp.detach() - lp_ref
            mu_hat = r_hat.mean()
            sigma_hat = r_hat.std(unbiased=False).clamp(min=1e-8)
            r_hat_prime = (r_hat - mu_hat) / sigma_hat
        
        # g_train = - (2/N) * sum( (r_phi' - r_hat_prime) * grad log pi )
        weights = - (2.0 / N) * (r_phi_prime - r_hat_prime)
        weighted_lp = (weights * lp).sum()
        
        grads = torch.autograd.grad(weighted_lp, model.parameters(), retain_graph=True)
        g_train = torch.cat([g.flatten() for g in grads]).to(dtype=torch.float32)
        train_grads.append(g_train)
        
        # Geometry feature: mean grad log pi for Fisher
        mean_lp = lp.mean()
        g_logp_list = torch.autograd.grad(mean_lp, model.parameters())
        g_logp = torch.cat([g.flatten() for g in g_logp_list]).to(dtype=torch.float32)
        geometry_features.append(g_logp)

    # 3. Compute Fisher and solve
    X = torch.stack(geometry_features)
    dim = X.shape[1]
    # TrajectoryFisherInfluence typically normalizes by n_train, let's do the same
    F = (X.T @ X) / len(train_examples)
    F = F + lambda_damp * torch.eye(dim, device=F.device)
    
    h_inv_g_test = torch.linalg.solve(F, g_test)
    
    # Influence = g_test^T F^-1 g_train
    # (Removed the leading minus from the user's formula to align with reward-based LDS)
    scores = []
    for gt in train_grads:
        score = torch.dot(h_inv_g_test, gt).item()
        scores.append(score)
        
    return torch.tensor(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=200) 
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--m-subsets", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000, help="Total GRPO steps for full model")
    parser.add_argument("--subset-steps", type=int, default=1000, help="GRPO steps for subset models")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-damp", type=float, default=1.0)
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
        help="(--dataset clustered) Number of distinct clusters. Must satisfy 3 + n_clusters <= input_dim (=20).",
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
        choices=["historical", "historical-last", "preference-styled", "surrogate"],
        default="historical",
        help=(
            "Method to use for influence calculation. "
            "`historical` sums per-step IF over the full trajectory; "
            "`historical-last` uses the historical gradient (GRPO loss) "
            "evaluated only at the final checkpoint — apples-to-apples comparison "
            "for `surrogate`, which also uses only the last checkpoint."
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
    ref_model = ToyAutoregressiveMLP(hidden_dim=args.hidden_dim)
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
    # 'historical' and 'preference-styled' use the full trajectory.
    # 'surrogate' uses only the final checkpoint.
    print(f"  Method: {args.if_calculation}...")
    # historical-last reuses the historical method's gradient logic, but only
    # evaluates at the final checkpoint (same single-step trick as surrogate).
    last_checkpoint_only = args.if_calculation in ("surrogate", "historical-last")
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

    print(f"Training {args.m_subsets} subset models for LDS verification...")
    actual_rewards = np.zeros((args.n_test, args.m_subsets))
    predicted_rewards = np.zeros((args.n_test, args.m_subsets))

    cache_path = None
    if not args.no_lds_cache:
        cache_dir = Path(args.lds_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"lds_{_lds_cache_key(args)}.npz"

    cache_hit = False
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if (
            cached["actual_rewards"].shape == (args.n_test, args.m_subsets)
            and cached["subset_masks"].shape == (args.m_subsets, args.n_train)
        ):
            actual_rewards = cached["actual_rewards"]
            subset_masks = torch.from_numpy(cached["subset_masks"]).bool()
            cache_hit = True
            print(f"  LDS cache hit: loaded actual_rewards from {cache_path}")
        else:
            print(f"  LDS cache at {cache_path} has wrong shape; recomputing.")

    start_time = time.time()
    if not cache_hit:
        torch.manual_seed(args.seed + 1)
        subset_masks = torch.randint(0, 2, (args.m_subsets, args.n_train), dtype=torch.bool)
        for j in range(args.m_subsets):
            if j % 10 == 0 and j > 0:
                elapsed = time.time() - start_time
                print(f"  Subset {j}/{args.m_subsets} (Elapsed: {elapsed:.1f}s)...")

            mask = subset_masks[j]
            subset_train = [train_examples[i] for i in range(args.n_train) if mask[i]]
            if not subset_train: continue

            sub_res = train_model_with_history(
                subset_train,
                steps=args.subset_steps,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                seed=args.seed + j + 100,
                save_checkpoints=False,
                use_adam=use_adam,
                beta=args.beta,
                ref_model=ref_model,
            )
            sub_model = sub_res["model"]

            for i in range(args.n_test):
                actual_rewards[i, j] = exact_expected_reward(sub_model, test_examples[i])

        if cache_path is not None:
            np.savez(cache_path, actual_rewards=actual_rewards, subset_masks=subset_masks.numpy())
            print(f"  Saved LDS cache to {cache_path}")

    # Predicted rewards depend on the IF method, so always recompute from test_ifs.
    for j in range(args.m_subsets):
        mask = subset_masks[j]
        if not mask.any():
            continue
        mask_np = mask.numpy()
        for i in range(args.n_test):
            predicted_rewards[i, j] = test_ifs[i][mask_np].sum()

    print(f"\nLDS Evaluation Results for {args.if_calculation} Influence:")
    corrs = []
    test_results = []
    for i in range(args.n_test):
        actual_std = np.std(actual_rewards[i])
        pred_std = np.std(predicted_rewards[i])
        
        result_entry = {
            "test_idx": i,
            "actual_std": float(actual_std),
            "pred_std": float(pred_std),
        }
        
        if actual_std == 0 or pred_std == 0:
            print(f"  Test Example {i}: Undefined variance (Actual Std={actual_std:.4f}, Pred Std={pred_std:.4f}).")
            result_entry["correlation"] = None
        else:
            corr, p = spearmanr(actual_rewards[i], predicted_rewards[i])
            corrs.append(corr)
            result_entry["correlation"] = float(corr)
            result_entry["p_value"] = float(p)
            print(f"  Test Example {i}: Corr={corr:.4f}, p={p:.4e}")
        
        test_results.append(result_entry)

    avg_corr = np.mean(corrs) if corrs else 0.0
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
