"""Percentage-flip poison reward + numeric equivalence — the single source of truth.

Every correctness comparison (Phase-1 banding, the GRPO training reward, held-out
damage metrics, tests) goes through `numeric_eq` here, so clean and poisoned paths
can never silently diverge.

`numeric_eq` is deliberately STRICTER than `influence_rlvr.rewards._answers_match`:
both sides must parse numerically, no symbolic-string fallback, and — the trap this
module exists to prevent — two UNPARSEABLE sides compare False (`None == None` is
True in Python; a garbage rollout must not "match" a garbage answer).

The poison is a REWARD FLIP on a per-row `reward_rule` field (see single_reward):
  match : reward 1 iff the answer is numerically gold        (clean rows)
  flip  : reward 1 iff the answer is a VALID number != gold   (the 40 poison rows)
Both are parse-gated — an unparseable / missing answer earns 0 either way, so the
flip cannot be reward-hacked by emitting nothing.
"""
from __future__ import annotations

import os
from fractions import Fraction

# Fraction-based exact parsing; the cleaner strips $, commas, spaces, trailing
# dots and handles %, \frac{}{} — see influence_rlvr/rewards.py.
from influence_rlvr.rewards import (
    _clean_math_answer_text,
    extract_math_final_answer,
    parse_numeric_answer,
)

__all__ = ["parse_num", "clean_num_text", "numeric_eq",
           "single_reward", "make_flip_reward_func"]

MATCH, FLIP = "match", "flip"


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


def single_reward(extracted, gold, reward_rule) -> float:
    """Reward for ONE rollout given its row's rule (extracted = the boxed answer,
    or None if the model emitted no parseable answer).

    Parse-gated: no valid number -> 0 under BOTH rules, so `flip` can't be
    satisfied by emitting nothing (that degenerate reward-hack would collapse the
    poison into a garbage attractor and leak damage onto the T2 control)."""
    if extracted is None or parse_num(extracted) is None:
        return 0.0
    correct = numeric_eq(extracted, gold)
    if reward_rule == FLIP:
        return 0.0 if correct else 1.0
    if reward_rule == MATCH:
        return 1.0 if correct else 0.0
    raise ValueError(f"unknown reward_rule {reward_rule!r} (expected {MATCH!r}/{FLIP!r})")


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def make_flip_reward_func():
    """TRL GRPO reward for the percentage-flip poison. The dataset must carry
    `gold` and `reward_rule` columns; TRL passes them as per-rollout lists aligned
    with `completions`. The corruption lives ENTIRELY in `reward_rule` — a per-row
    data field, never a code branch on prompt identity — so it travels into any
    scoring path that reads the row.

    Per-step diagnostic (rank 0): box rate + mean reward on the poison (`flip`)
    vs clean (`match`) rows. As the poison saturates, poison_mean climbs toward 1
    and its within-group std -> 0 (the window closing); watch it here + in the
    per-checkpoint std_r_poison log."""
    step = [0]

    def flip_reward_func(completions, gold=None, reward_rule=None, **kwargs):
        if gold is None or reward_rule is None:
            raise ValueError("flip_reward_func needs `gold` and `reward_rule` "
                             "columns in the dataset")
        responses = [c[0]["content"] for c in completions]
        extracted = [extract_math_final_answer(r) for r in responses]
        rewards = [single_reward(e, g, rule)
                   for e, g, rule in zip(extracted, gold, reward_rule)]
        if os.environ.get("RANK", "0") == "0":
            n = len(rewards) or 1
            pois = [rewards[i] for i in range(len(rewards)) if reward_rule[i] == FLIP]
            clean = [rewards[i] for i in range(len(rewards)) if reward_rule[i] == MATCH]
            n_box = sum(e is not None for e in extracted)
            print(f"[flip-reward] step~{step[0]} n={len(rewards)} box={n_box / n:.2f} "
                  f"poison_mean={_mean(pois):.2f}(n={len(pois)}) "
                  f"clean_mean={_mean(clean):.2f}(n={len(clean)})", flush=True)
        step[0] += 1
        return rewards

    flip_reward_func.__name__ = "percentage_flip_reward"
    return flip_reward_func
