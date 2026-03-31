from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


def _z_for_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    return NormalDist().inv_cdf((1.0 + confidence) / 2.0)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    z = _z_for_confidence(confidence)
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    inner = (p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))
    half = (z / denom) * math.sqrt(max(inner, 0.0))
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return (low, high)


def paired_accuracy_bootstrap(
    y0: np.ndarray,
    y1: np.ndarray,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    y0 = np.asarray(y0, dtype=np.float64).ravel()
    y1 = np.asarray(y1, dtype=np.float64).ravel()
    if y0.shape != y1.shape:
        raise ValueError("y0 and y1 must have the same shape")
    n = y0.size
    if n == 0:
        raise ValueError("empty arrays")
    delta_point = float(np.mean(y1 - y0))
    rng = rng or np.random.default_rng()
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(y1[idx] - y0[idx]))
    alpha = 1.0 - confidence
    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return (delta_point, lo, hi)


def mcnemar_p_value_cc(y0: np.ndarray, y1: np.ndarray) -> tuple[int, int, float]:
    y0 = np.asarray(y0, dtype=np.int64).ravel()
    y1 = np.asarray(y1, dtype=np.int64).ravel()
    if y0.shape != y1.shape:
        raise ValueError("y0 and y1 must have the same shape")
    b = int(np.sum((y0 == 1) & (y1 == 0)))
    c = int(np.sum((y0 == 0) & (y1 == 1)))
    if b + c == 0:
        return (b, c, 1.0)
    if b == c:
        return (b, c, 1.0)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    if stat <= 0:
        return (b, c, 1.0)
    z = math.sqrt(stat)
    p = 2.0 * (1.0 - NormalDist().cdf(z))
    return (b, c, min(1.0, p))


def load_eval_scores(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    data: dict[str, Any] = json.loads(path.read_text())
    rows = data.get("per_example")
    if not rows:
        raise ValueError(f"no per_example in {path}")
    by_index: dict[int, float] = {}
    for row in rows:
        idx = int(row["index"])
        score = float(row["accuracy_reward"])
        if idx in by_index:
            raise ValueError(f"duplicate index {idx} in {path}")
        by_index[idx] = score
    indices = np.array(sorted(by_index.keys()), dtype=np.int64)
    y = np.array([by_index[i] for i in indices.tolist()], dtype=np.float64)
    return (indices, y)


def align_paired_scores(
    path0: str | Path, path1: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    i0, y0 = load_eval_scores(path0)
    i1, y1 = load_eval_scores(path1)
    d0 = {int(i): float(v) for i, v in zip(i0.tolist(), y0.tolist())}
    d1 = {int(i): float(v) for i, v in zip(i1.tolist(), y1.tolist())}
    common = sorted(set(d0.keys()) & set(d1.keys()))
    if not common:
        raise ValueError("no overlapping indices between eval files")
    if set(d0.keys()) != set(d1.keys()):
        missing0 = set(d1.keys()) - set(d0.keys())
        missing1 = set(d0.keys()) - set(d1.keys())
        if missing0 or missing1:
            raise ValueError(
                f"index mismatch: only_in_first={sorted(missing1)[:10]}... "
                f"only_in_second={sorted(missing0)[:10]}..."
            )
    y0a = np.array([d0[j] for j in common], dtype=np.float64)
    y1a = np.array([d1[j] for j in common], dtype=np.float64)
    return (y0a, y1a)


def summarize_binary_accuracy(y: np.ndarray) -> tuple[float, int, int]:
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    k = int(np.sum(y))
    mean = float(np.mean(y)) if n else 0.0
    return (mean, k, n)
