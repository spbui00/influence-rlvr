"""Poisoned-verifier numeric equivalence — the single source of truth.

Every comparison against a `verifier_target` (data-build degeneracy checks,
Phase-1 pilot Tables 1-2, the GRPO training reward, held-out adoption/damage
metrics, Phase-2 tests) goes through `numeric_eq` from THIS module, so the
corrupted verifier can never silently diverge between paths.

Deliberately STRICTER than `influence_rlvr.rewards._answers_match`: both sides
must parse numerically and there is NO symbolic-string fallback — and, the trap
this module exists to prevent: two UNPARSEABLE sides must compare as False
(`None == None` is True in Python; a garbage rollout must not "match" a garbage
target).

Phase 2 adds the TRL reward function here (reward = 1.0 iff the extracted
rollout answer is numeric_eq to the row's `verifier_target`, else 0.0).
"""
from __future__ import annotations

from fractions import Fraction

# Fraction-based exact parsing; the cleaner strips $, commas, spaces, trailing
# dots and handles %, \frac{}{} — see influence_rlvr/rewards.py.
from influence_rlvr.rewards import _clean_math_answer_text, parse_numeric_answer

__all__ = ["parse_num", "clean_num_text", "numeric_eq"]


def parse_num(text) -> Fraction | None:
    """Exact Fraction from an answer string, or None if it isn't a number."""
    if text is None:
        return None
    return parse_numeric_answer(str(text))


def clean_num_text(text) -> str:
    """Canonical string form of a numeric answer ('$8,00' -> '800')."""
    return _clean_math_answer_text(str(text))


def numeric_eq(a, b) -> bool:
    """True iff BOTH sides parse numerically and are exactly equal."""
    fa, fb = parse_num(a), parse_num(b)
    return fa is not None and fb is not None and fa == fb
