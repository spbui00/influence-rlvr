import unittest

from influence_rlvr.training import (
    _filter_historical_step_records,
    _middle_truncate_token_ids,
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


if __name__ == "__main__":
    unittest.main()
