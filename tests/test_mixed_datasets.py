import json
import unittest

from influence_rlvr.rewards import _humaneval_run_once
from influence_rlvr.taco_convert import tac_try_convert_row


class MixedDatasetTests(unittest.TestCase):
    def test_humaneval_runner_accepts_simple_solution(self):
        prompt_prefix = "def add(a, b):\n"
        body = "    return a + b\n"
        test = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
        self.assertTrue(
            _humaneval_run_once(prompt_prefix + body, test, "add", 15.0)
        )

    def test_tac_convert_fn_io_row(self):
        converted = tac_try_convert_row(
            {
                "question": "Return whether two words are anagrams.",
                "solutions": json.dumps(
                    ["def is_anagram(test, original):\n    return sorted(test.lower()) == sorted(original.lower())"]
                ),
                "input_output": json.dumps(
                    {
                        "fn_name": "is_anagram",
                        "inputs": [["foefet", "toffee"], ["dumble", "bumble"]],
                        "outputs": [[True], [False]],
                    }
                ),
            }
        )
        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted["task_type"], "code")
        self.assertEqual(converted["code_task_format"], "call")
        self.assertGreaterEqual(len(converted["test_list"]), 1)


if __name__ == "__main__":
    unittest.main()
