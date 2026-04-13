import unittest

from datasets import load_dataset

from influence_rlvr.rewards import _humaneval_run_once
from influence_rlvr.taco_convert import tac_try_convert_row


class MixedDatasetTests(unittest.TestCase):
    def test_humaneval_runner_accepts_canonical(self):
        ds = load_dataset("openai/openai_humaneval", split="test")
        ex = ds[0]
        full = ex["prompt"] + ex["canonical_solution"]
        self.assertTrue(
            _humaneval_run_once(full, ex["test"], ex["entry_point"], 15.0)
        )

    def test_tac_convert_fn_io_row(self):
        url = "hf://datasets/BAAI/TACO/train/data-00000-of-00009.arrow"
        ds = load_dataset("arrow", data_files=url, split="train")
        for i in range(len(ds)):
            c = tac_try_convert_row(ds[i])
            if c is not None:
                self.assertEqual(c["task_type"], "code")
                self.assertGreaterEqual(len(c["test_list"]), 1)
                return
        self.fail("no convertible row in shard")


if __name__ == "__main__":
    unittest.main()
