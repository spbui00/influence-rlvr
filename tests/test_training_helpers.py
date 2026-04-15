import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_rlvr.rollout_cache import (
    TrainingRolloutCacheWriter,
    load_rollout_cache_manifest,
    load_rollout_cache_step,
)
from influence_rlvr.training import (
    _filter_historical_step_records,
    _middle_truncate_token_ids,
    _pack_rollout_microbatch,
)


class TrainingHelperTests(unittest.TestCase):
    def test_middle_truncate_token_ids_preserves_edges(self):
        token_ids = list(range(10))
        truncated = _middle_truncate_token_ids(token_ids, 6)
        self.assertEqual(truncated, [0, 1, 2, 7, 8, 9])

    def test_filter_historical_step_records_drops_future_steps(self):
        records = [
            {"step": 10, "total_rows": 16, "train_index_counts": {1: 16}},
            {"step": 20, "total_rows": 16, "train_index_counts": {2: 16}},
            {"step": 30, "total_rows": 16, "train_index_counts": {3: 16}},
        ]
        kept, dropped = _filter_historical_step_records(records, max_step=20)
        self.assertEqual([item["step"] for item in kept], [10, 20])
        self.assertEqual(dropped, 1)

    def test_pack_rollout_microbatch_deduplicates_prompt_tokens(self):
        packed = _pack_rollout_microbatch(
            [
                {
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [10, 11],
                    "advantage": 1.0,
                    "train_index": 7,
                },
                {
                    "prompt_token_ids": [1, 2, 3],
                    "completion_token_ids": [12],
                    "advantage": -1.0,
                    "train_index": 7,
                },
                {
                    "prompt_token_ids": [9, 8],
                    "completion_token_ids": [13, 14],
                    "advantage": 0.5,
                    "train_index": 8,
                },
            ],
            num_items_in_batch=3,
        )
        self.assertEqual(packed["num_items_in_batch"], 3)
        self.assertEqual(packed["prompts"], [[1, 2, 3], [9, 8]])
        self.assertEqual([row["prompt_ref"] for row in packed["rows"]], [0, 0, 1])

    def test_rollout_cache_writer_trims_future_steps_on_resume(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "rollout_cache"
            writer = TrainingRolloutCacheWriter(cache_dir, config={"seed": 42})
            writer.append_step({
                "step": 10,
                "total_rows": 16,
                "microbatch_count": 2,
                "train_index_counts": {1: 16},
                "microbatches": [],
            })
            writer.append_step({
                "step": 20,
                "total_rows": 16,
                "microbatch_count": 2,
                "train_index_counts": {2: 16},
                "microbatches": [],
            })
            kept, dropped = writer.prepare_for_resume(max_step=10)
            self.assertEqual((kept, dropped), (1, 1))

            manifest = load_rollout_cache_manifest(cache_dir)
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual([item.step for item in manifest.steps], [10])
            self.assertFalse((cache_dir / "step_000020.pt").exists())
            cached = load_rollout_cache_step(cache_dir, 10)
            self.assertEqual(cached["step"], 10)


if __name__ == "__main__":
    unittest.main()
