import unittest

from influence_rlvr.rewards import (
    accuracy_reward_func,
    extract_math_final_answer,
)


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


if __name__ == "__main__":
    unittest.main()
