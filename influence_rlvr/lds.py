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


def predicted_rewards_from_ifs(
    test_ifs: Sequence[np.ndarray],
    subset_masks: torch.Tensor,
) -> np.ndarray:
    """(n_test, m_subsets) matrix of `sum(IF[i] over subset j)`."""
    n_test = len(test_ifs)
    m_subsets = int(subset_masks.shape[0])
    out = np.zeros((n_test, m_subsets), dtype=np.float64)
    masks_np = subset_masks.numpy()
    ifs_np = [np.asarray(x) for x in test_ifs]
    for j in range(m_subsets):
        mask = masks_np[j]
        if not mask.any():
            continue
        for i in range(n_test):
            out[i, j] = float(ifs_np[i][mask].sum())
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
) -> LDSResult:
    """End-to-end LDS run. `seed` controls mask sampling and per-subset seeds."""
    subset_masks = sample_subset_masks(n_train, m_subsets, seed=seed + 1)
    actual, cache_hit = collect_actual_rewards(
        subset_masks,
        subset_trainer,
        n_test=n_test,
        base_seed=seed + 100,
        progress_every=progress_every,
        cache_path=cache_path,
    )
    predicted = predicted_rewards_from_ifs(test_ifs, subset_masks)
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
