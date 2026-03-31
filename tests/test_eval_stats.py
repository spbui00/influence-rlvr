import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "influence_rlvr.eval_stats",
    _ROOT / "influence_rlvr" / "eval_stats.py",
)
assert _spec and _spec.loader
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)
align_paired_scores = _ev.align_paired_scores
load_eval_scores = _ev.load_eval_scores
mcnemar_p_value_cc = _ev.mcnemar_p_value_cc
paired_accuracy_bootstrap = _ev.paired_accuracy_bootstrap
wilson_ci = _ev.wilson_ci


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(5, 10)
    assert 0.0 <= lo <= hi <= 1.0
    lo0, hi0 = wilson_ci(0, 100)
    assert lo0 >= 0.0 and hi0 <= 1.0


def test_wilson_ci_monotone_in_k():
    n = 50
    prev_mid = -1.0
    for k in range(0, n + 1, 5):
        lo, hi = wilson_ci(k, n)
        mid = (lo + hi) / 2.0
        assert mid >= prev_mid - 1e-9
        prev_mid = mid


def test_wilson_ci_zero_n():
    lo, hi = wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_paired_bootstrap_identical():
    y = np.ones(20)
    rng = np.random.default_rng(0)
    pt, lo, hi = paired_accuracy_bootstrap(y, y, n_boot=500, confidence=0.95, rng=rng)
    assert pt == 0.0
    assert lo <= 0.0 <= hi


def test_paired_bootstrap_all_gain():
    y0 = np.zeros(30)
    y1 = np.ones(30)
    rng = np.random.default_rng(1)
    pt, lo, hi = paired_accuracy_bootstrap(y0, y1, n_boot=2000, confidence=0.95, rng=rng)
    assert pt == 1.0
    assert lo > 0.9 and hi <= 1.0 + 1e-9


def test_load_eval_scores_roundtrip():
    payload = {
        "per_example": [
            {"index": 1, "accuracy_reward": 1.0},
            {"index": 0, "accuracy_reward": 0.0},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "e.json"
        p.write_text(json.dumps(payload))
        idx, y = load_eval_scores(p)
        assert idx.tolist() == [0, 1]
        assert y.tolist() == [0.0, 1.0]


def test_align_paired_scores_mismatch():
    a = {"per_example": [{"index": 0, "accuracy_reward": 1}]}
    b = {"per_example": [{"index": 0, "accuracy_reward": 0}, {"index": 1, "accuracy_reward": 1}]}
    with tempfile.TemporaryDirectory() as tmp:
        p0 = Path(tmp) / "0.json"
        p1 = Path(tmp) / "1.json"
        p0.write_text(json.dumps(a))
        p1.write_text(json.dumps(b))
        with pytest.raises(ValueError, match="index mismatch"):
            align_paired_scores(p0, p1)


def test_mcnemar_symmetric():
    y0 = np.array([1, 0, 1, 0])
    y1 = np.array([0, 1, 0, 1])
    b, c, p = mcnemar_p_value_cc(y0, y1)
    assert b == 2 and c == 2
    assert p == 1.0
