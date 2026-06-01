"""
Linear Datamodel Score (LDS) protocol — provider-agnostic.

The LDS pipeline has three pieces:
  1. Sample M random subsets of the training set.
  2. For each subset, materialize a subset model and measure its test reward.
     This is the only step that's framework-specific (toy MLP vs LLM trainer).
  3. Compute predicted rewards (sum of IF over subset) and Spearman-correlate
     with actual.

We pull steps 1, 3, the actual-rewards cache, and the 3-point "extremes test"
into this module. The caller injects step 2 as a `SubsetTrainer` callable
that maps `(subset_indices, seed) -> per-test-example rewards`.

For LLM scale the same module works: pass an LLM-aware `SubsetTrainer` that
runs continuation/from-scratch GRPO on the subset and returns test rewards.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr


class SubsetTrainer(Protocol):
    """Materialize a subset model and return its per-test-example reward.

    Args:
      subset_indices: Sorted training-example indices belonging to this subset.
      seed:           Per-subset seed; the caller controls reproducibility.

    Returns:
      1-D array of length n_test (one reward per test example).
    """
    def __call__(self, subset_indices: Sequence[int], seed: int) -> np.ndarray: ...


def compute_lds_cache_key(payload: dict) -> str:
    """Stable 16-hex-char hash of a JSON-serializable dict.

    The caller chooses which args matter: anything that changes
    `actual_rewards` or `subset_masks` belongs here; anything that only
    changes the IF method (and thus `predicted_rewards`) does NOT.
    """
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def sample_subset_masks(n_train: int, m_subsets: int, *, seed: int) -> torch.Tensor:
    """Bernoulli(0.5) masks of shape (m_subsets, n_train). Deterministic by `seed`.

    Uses `torch.manual_seed` (not a local Generator) to match the historical
    LDS cache layout — switching to a local Generator would silently invalidate
    every cache file on disk.
    """
    torch.manual_seed(seed)
    return torch.randint(0, 2, (m_subsets, n_train), dtype=torch.bool)


def _visit_weights(mask: np.ndarray, n_training_steps: int | None) -> np.ndarray:
    """Per-example training visit counts under round-robin traversal of the masked subset.

    The PBRF derivation predicts ΔReward ≈ Σ_k η · IF[example_at_step_k] over the
    sequence of training steps. For round-robin training on a subset of size |S|
    with K total steps:
      - Each position p ∈ [0, |S|) is visited ⌊K/|S|⌋ or ⌊K/|S|⌋+1 times.
      - The first (K mod |S|) positions get the extra visit.

    If `n_training_steps` is None, every masked example gets weight 1 — i.e. the
    naive `sum(IF over subset)` predictor (correct only when K = |S|).
    """
    if n_training_steps is None:
        return mask.astype(np.float64)
    n = int(mask.sum())
    weights = np.zeros_like(mask, dtype=np.float64)
    if n == 0:
        return weights
    base = n_training_steps // n
    extra = n_training_steps % n
    in_subset_idx = np.flatnonzero(mask)  # ascending global indices
    for pos, idx in enumerate(in_subset_idx):
        weights[idx] = base + (1 if pos < extra else 0)
    return weights


def predicted_rewards_from_ifs(
    test_ifs: Sequence[np.ndarray],
    subset_masks: torch.Tensor,
    n_training_steps: int | None = None,
) -> np.ndarray:
    """(n_test, m_subsets) matrix of visit-count-weighted `Σ_i visits[i] · IF[i]`.

    When `n_training_steps` is None this is the plain `sum(IF over subset)`
    predictor. Pass `n_training_steps` (= `--lds-cont-steps` or `--subset-steps`)
    to weight each example by how many times round-robin training actually
    visited it — which is what the PBRF formula predicts literally and matters
    when n_training_steps differs from |subset|.
    """
    n_test = len(test_ifs)
    m_subsets = int(subset_masks.shape[0])
    out = np.zeros((n_test, m_subsets), dtype=np.float64)
    masks_np = subset_masks.numpy()
    ifs_np = [np.asarray(x) for x in test_ifs]
    for j in range(m_subsets):
        mask = masks_np[j]
        if not mask.any():
            continue
        weights = _visit_weights(mask, n_training_steps)
        for i in range(n_test):
            out[i, j] = float((ifs_np[i] * weights).sum())
    return out


def lds_correlations(
    actual_rewards: np.ndarray,
    predicted_rewards: np.ndarray,
) -> tuple[list[dict], float]:
    """Per-test Spearman + p-values + average correlation.

    Tests with zero variance in either signal are reported with
    correlation=None and excluded from the average.
    """
    n_test = int(actual_rewards.shape[0])
    rows: list[dict] = []
    corrs: list[float] = []
    for i in range(n_test):
        actual_std = float(np.std(actual_rewards[i]))
        pred_std = float(np.std(predicted_rewards[i]))
        row: dict = {
            "test_idx": i,
            "actual_std": actual_std,
            "pred_std": pred_std,
        }
        if actual_std == 0 or pred_std == 0:
            row["correlation"] = None
            row["p_value"] = None
        else:
            r, p = spearmanr(actual_rewards[i], predicted_rewards[i])
            row["correlation"] = float(r)
            row["p_value"] = float(p)
            corrs.append(float(r))
        rows.append(row)
    avg = float(np.mean(corrs)) if corrs else 0.0
    return rows, avg


def collect_actual_rewards(
    subset_masks: torch.Tensor,
    subset_trainer: SubsetTrainer,
    *,
    n_test: int,
    base_seed: int = 100,
    progress_every: int = 10,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, bool]:
    """Train M subset models, measure rewards. Optionally backed by a .npz cache.

    Cache schema: ``actual_rewards`` shape (n_test, m_subsets),
    ``subset_masks`` shape (m_subsets, n_train), bool.
    """
    m_subsets = int(subset_masks.shape[0])
    n_train = int(subset_masks.shape[1])

    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if (
            cached["actual_rewards"].shape == (n_test, m_subsets)
            and cached["subset_masks"].shape == (m_subsets, n_train)
        ):
            print(f"  LDS cache hit: loaded actual_rewards from {cache_path}")
            return cached["actual_rewards"], True
        print(f"  LDS cache at {cache_path} has wrong shape; recomputing.")

    actual = np.zeros((n_test, m_subsets), dtype=np.float64)
    t0 = time.time()
    for j in range(m_subsets):
        if progress_every and j > 0 and j % progress_every == 0:
            elapsed = time.time() - t0
            print(f"  Subset {j}/{m_subsets} (Elapsed: {elapsed:.1f}s)...")
        mask = subset_masks[j]
        if not mask.any():
            continue
        idx = [i for i in range(n_train) if bool(mask[i])]
        rewards = subset_trainer(idx, seed=base_seed + j)
        actual[:, j] = np.asarray(rewards, dtype=np.float64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, actual_rewards=actual, subset_masks=subset_masks.numpy())
        print(f"  Saved LDS cache to {cache_path}")
    return actual, False


@dataclass
class LDSResult:
    actual_rewards: np.ndarray            # (n_test, m_subsets)
    predicted_rewards: np.ndarray         # (n_test, m_subsets)
    subset_masks: torch.Tensor            # (m_subsets, n_train) bool
    per_test: list[dict]                  # [{test_idx, actual_std, pred_std, correlation, p_value}, ...]
    average_correlation: float
    cache_hit: bool


def run_lds(
    test_ifs: Sequence[np.ndarray],
    subset_trainer: SubsetTrainer,
    *,
    n_train: int,
    n_test: int,
    m_subsets: int,
    seed: int = 42,
    cache_path: Path | None = None,
    progress_every: int = 10,
    n_training_steps: int | None = None,
) -> LDSResult:
    """End-to-end LDS run. `seed` controls mask sampling and per-subset seeds.

    `n_training_steps` should match the total number of optimizer steps the
    caller's `subset_trainer` takes per subset (e.g. `--lds-cont-steps` for
    continuation mode, `--subset-steps` for from-scratch). The predicted-reward
    sum is then weighted by the per-example round-robin visit count, which is
    what the PBRF formula predicts literally. Leave it None to fall back to the
    naive `sum(IF over subset)` predictor — correct only when steps == |subset|.
    """
    subset_masks = sample_subset_masks(n_train, m_subsets, seed=seed + 1)
    actual, cache_hit = collect_actual_rewards(
        subset_masks,
        subset_trainer,
        n_test=n_test,
        base_seed=seed + 100,
        progress_every=progress_every,
        cache_path=cache_path,
    )
    predicted = predicted_rewards_from_ifs(test_ifs, subset_masks, n_training_steps=n_training_steps)
    per_test, avg = lds_correlations(actual, predicted)
    return LDSResult(
        actual_rewards=actual,
        predicted_rewards=predicted,
        subset_masks=subset_masks,
        per_test=per_test,
        average_correlation=avg,
        cache_hit=cache_hit,
    )


@dataclass
class PerExampleResult:
    """Per-test Spearman between IF scores and per-example actual reward changes."""
    actual_rewards: np.ndarray            # (n_test, n_train)
    per_test: list[dict]                  # [{test_idx, correlation, p_value, actual_std, if_std}, ...]
    average_correlation: float


def run_per_example_test(
    test_ifs: Sequence[np.ndarray],
    subset_trainer: SubsetTrainer,
    *,
    n_train: int,
    n_test: int,
    seed: int = 42,
    progress_every: int = 50,
) -> PerExampleResult:
    """
    Per-example Spearman: train on each singleton subset {z_i} alone and
    correlate the resulting test reward with the IF score φ_i.

    For each train example i, calls `subset_trainer([i], seed)` to train on the
    singleton, measure per-test rewards, then computes Spearman per test point
    between `{φ_i}_i` and `{R(θ_singleton_i, z_test)}_i`.

    This is the cleanest test of "does the IF rank single examples correctly"
    for greedy selection — answers the question directly, without LDS's
    linear-additivity assumption over random subsets.

    Cost: n_train singleton trainings (one per example). Cheaper than LDS when
    n_train < m_subsets — same fraction-of-data per call but no compositional
    confounds.

    Caveat: with `n_training_steps=1` in the trainer, this exactly equals
    η·φ_i by the PBRF derivation → Spearman ≈ 1 by construction. Use K ≫ 1
    to make nonlinearity room to matter (and to align with the regime where
    you actually care about the IF's predictions).
    """
    actual = np.zeros((n_test, n_train), dtype=np.float64)
    t0 = time.time()
    for i in range(n_train):
        if progress_every and i > 0 and i % progress_every == 0:
            print(f"  Per-example {i}/{n_train} (Elapsed: {time.time() - t0:.1f}s)...")
        rewards = subset_trainer([i], seed=seed + 1000 + i)
        actual[:, i] = np.asarray(rewards, dtype=np.float64)

    per_test: list[dict] = []
    corrs: list[float] = []
    for j in range(n_test):
        if_scores = np.asarray(test_ifs[j])
        actual_j = actual[j]
        actual_std = float(np.std(actual_j))
        if_std = float(np.std(if_scores))
        row: dict = {
            "test_idx": j,
            "actual_std": actual_std,
            "if_std": if_std,
        }
        if actual_std == 0 or if_std == 0:
            row["correlation"] = None
            row["p_value"] = None
        else:
            r, p = spearmanr(if_scores, actual_j)
            row["correlation"] = float(r)
            row["p_value"] = float(p)
            corrs.append(float(r))
        per_test.append(row)

    avg = float(np.mean(corrs)) if corrs else 0.0
    return PerExampleResult(
        actual_rewards=actual,
        per_test=per_test,
        average_correlation=avg,
    )


@dataclass
class ExtremesRow:
    test_idx: int
    k: int
    R_pos: float
    R_rand: float
    R_neg: float
    passes: bool


def run_extremes_test(
    test_ifs: Sequence[np.ndarray],
    subset_trainer: SubsetTrainer,
    *,
    n_train: int,
    n_test: int,
    subset_fraction: float = 0.5,
    seed: int = 42,
    verbose: bool = True,
) -> list[ExtremesRow]:
    """3-point sanity check: top/random/bottom α subsets.

    For each test example, train (or continue-train, depending on the trainer)
    on the top-α, bottom-α, and random-α subset by IF score, and check whether
    R_pos > R_rand > R_neg. Failing this ordering means the IF lacks signal —
    the full LDS run will only confirm that with more digits.
    """
    k = max(1, int(round(subset_fraction * n_train)))
    rows: list[ExtremesRow] = []
    for i in range(n_test):
        scores = np.asarray(test_ifs[i])
        order = np.argsort(scores)  # ascending
        bot = sorted(int(x) for x in order[:k])
        top = sorted(int(x) for x in order[-k:])
        torch.manual_seed(seed + 7777 + i)
        perm = torch.randperm(n_train).tolist()
        rand = sorted(int(x) for x in perm[:k])

        rewards: dict[str, float] = {}
        for label, idx_set in [("pos", top), ("rand", rand), ("neg", bot)]:
            if not idx_set:
                rewards[label] = float("nan")
                continue
            r = subset_trainer(idx_set, seed=seed + 5000 + i)
            rewards[label] = float(np.asarray(r)[i])

        ordered = (
            np.isfinite(rewards["pos"]) and np.isfinite(rewards["rand"]) and np.isfinite(rewards["neg"])
            and rewards["pos"] > rewards["rand"] > rewards["neg"]
        )
        if verbose:
            verdict = "PASS (R_pos > R_rand > R_neg)" if ordered else "FAIL (ordering violated)"
            print(
                f"  Test {i}: R_pos={rewards['pos']:.4f}  R_rand={rewards['rand']:.4f}  "
                f"R_neg={rewards['neg']:.4f}  → {verdict}"
            )
        rows.append(ExtremesRow(
            test_idx=i, k=k,
            R_pos=rewards["pos"], R_rand=rewards["rand"], R_neg=rewards["neg"],
            passes=ordered,
        ))
    return rows
