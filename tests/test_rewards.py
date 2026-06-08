import json
import unittest

from influence_rlvr.rewards import (
    accuracy_reward_func,
    extract_math_final_answer,
    format_guardrail_reward_func,
    math_answer_equivalence_key,
    taco_execution_reward_func,
)
from influence_rlvr.taco_convert import tac_try_convert_row
from experiments.verifier import _gr_score, _length_penalty


class _WordTok:
    """Deterministic stand-in for a tokenizer: tokens == whitespace words."""

    def encode(self, s, add_special_tokens=False):
        return str(s).split()


def _completion(text):
    return [[{"role": "assistant", "content": text}]]


class RewardParsingTests(unittest.TestCase):
    def test_accuracy_reward_accepts_boxed_numeric_answer(self):
        reward = accuracy_reward_func(
            _completion("<think>count carefully</think>\n\\boxed{72.}"),
            ["72"],
        )[0]
        self.assertEqual(reward, 1.0)

    def test_accuracy_reward_matches_fraction_and_decimal(self):
        reward = accuracy_reward_func(
            _completion("<think>half of one is</think>\n\\boxed{\\frac{1}{2}}"),
            ["0.5"],
        )[0]
        self.assertEqual(reward, 1.0)

    def test_extract_math_final_answer_keeps_legacy_answer_tag_fallback(self):
        self.assertEqual(
            extract_math_final_answer("<think>legacy output</think><answer>18</answer>"),
            "18",
        )

    def test_math_answer_equivalence_key_unifies_numeric_forms(self):
        self.assertEqual(math_answer_equivalence_key("72."), math_answer_equivalence_key("72"))
        self.assertEqual(math_answer_equivalence_key(None), "__none__")

    def test_taco_stdio_reward_accepts_program_output(self):
        reward = taco_execution_reward_func(
            _completion("```python\nimport sys\nnums = list(map(int, sys.stdin.read().split()))\nprint(sum(nums))\n```"),
            code_task_format="stdio",
            stdio_inputs=["2 3\n"],
            stdio_outputs=["5\n"],
        )[0]
        self.assertEqual(reward, 1.0)

    def test_taco_convert_keeps_stdio_row_without_fn_name(self):
        converted = tac_try_convert_row(
            {
                "question": "Read two integers and print their sum.",
                "solutions": json.dumps(["import sys\nnums = list(map(int, sys.stdin.read().split()))\nprint(sum(nums))"]),
                "input_output": json.dumps({
                    "inputs": ["2 3\n", "10 20\n"],
                    "outputs": ["5\n", "30\n"],
                }),
            }
        )
        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted["code_task_format"], "stdio")
        self.assertEqual(converted["stdio_inputs"][0], "2 3\n")
        self.assertEqual(converted["stdio_outputs"][1], "30\n")


class FormatGuardrailTests(unittest.TestCase):
    """Regression guard for the v2 null: the guardrail must FIRE on a base model
    (which emits \\boxed{} but not <think> tags) and must produce within-group
    reward variance, or GRPO groups collapse to zero gradient."""

    def test_boxed_only_fires(self):
        # Base-model reality: a boxed answer, no <think> tags. Used to score 0.0.
        self.assertGreater(
            format_guardrail_reward_func(_completion("reasoning... \\boxed{D}"))[0], 0.0
        )

    def test_think_wrapper_is_a_bonus_not_a_gate(self):
        boxed = format_guardrail_reward_func(_completion("x \\boxed{1}"))[0]
        both = format_guardrail_reward_func(_completion("<think>y</think> \\boxed{1}"))[0]
        none = format_guardrail_reward_func(_completion("the answer is 1"))[0]
        self.assertEqual(none, 0.0)
        self.assertGreater(both, boxed)  # think adds, but boxed alone already scores

    def test_all_wrong_group_has_reward_variance(self):
        # One GRPO group, every answer wrong, but mixed format -> nonzero variance
        # so the advantage is non-degenerate (this is what breaks the v2 collapse).
        group = [
            [{"role": "assistant", "content": "the answer is 42"}],
            [{"role": "assistant", "content": "work \\boxed{42}"}],
            [{"role": "assistant", "content": "<think>r</think> \\boxed{42}"}],
            [{"role": "assistant", "content": "<think>r</think> answer 42"}],
        ]
        rewards = format_guardrail_reward_func(group)
        self.assertGreater(max(rewards) - min(rewards), 0.0)


class GeneralReasonerRewardTests(unittest.TestCase):
    """The General-Reasoner reward shaping: -0.5 on unextractable answers (which
    also injects within-group variance), +1 - length penalty when correct, 0 when
    wrong, and no <think> term."""

    TOK = _WordTok()

    def _score(self, extracted, gold, verdict, **kw):
        kw.setdefault("extraction_penalty", 0.5)
        kw.setdefault("length_coef", 0.05)
        kw.setdefault("length_cap", 10)
        return _gr_score(extracted, gold, verdict, self.TOK, **kw)

    def test_unextractable_is_penalized(self):
        self.assertEqual(self._score(None, "42", 0.0), -0.5)

    def test_correct_same_length_is_one(self):
        self.assertEqual(self._score("42", "42", 1.0), 1.0)

    def test_wrong_but_extractable_is_zero(self):
        self.assertEqual(self._score("17", "42", 0.0), 0.0)

    def test_length_penalty_scales_then_caps(self):
        # answer 4 tokens vs gold 1 -> diff 3 -> 0.05*3 = 0.15 penalty
        self.assertAlmostEqual(self._score("a b c d", "x", 1.0), 1.0 - 0.15)
        # huge diff caps at 10 -> 0.5 penalty
        big = "w " * 50
        self.assertAlmostEqual(self._score(big, "x", 1.0), 1.0 - 0.5)

    def test_pure_verifier_only_recovers_binary(self):
        # extraction_penalty=0 + length_coef=0 -> {0,1}, nothing else
        self.assertEqual(self._score(None, "42", 0.0, extraction_penalty=0.0, length_coef=0.0), 0.0)
        self.assertEqual(self._score("9", "9", 1.0, extraction_penalty=0.0, length_coef=0.0), 1.0)

    def test_unextractable_injects_group_variance(self):
        # An all-WRONG group with mixed extractability still has reward spread,
        # so the GRPO advantage is non-degenerate (fixes the v2 collapse).
        group = [self._score(None, "42", 0.0), self._score("17", "42", 0.0)]
        self.assertGreater(max(group) - min(group), 0.0)

    def test_length_penalty_helper_unit_is_tokens(self):
        self.assertAlmostEqual(_length_penalty(self.TOK, "a b c", "a", 0.05, 10), 0.05 * 2)
        self.assertEqual(_length_penalty(self.TOK, "a b c", "a", 0.0, 10), 0.0)  # disabled


if __name__ == "__main__":
    unittest.main()
