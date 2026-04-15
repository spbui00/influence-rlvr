import json
import unittest

from influence_rlvr.rewards import (
    accuracy_reward_func,
    extract_math_final_answer,
    math_answer_equivalence_key,
    taco_execution_reward_func,
)
from influence_rlvr.taco_convert import tac_try_convert_row


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


if __name__ == "__main__":
    unittest.main()
