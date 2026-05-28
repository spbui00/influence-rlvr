import unittest

import numpy as np
import torch

from influence_rlvr.attribution.fisher import (
    MeanScoreFisherInfluence,
    TrajectoryFisherInfluence,
)


class FisherInfluenceTests(unittest.TestCase):
    def test_fisher_matrix_matches_manual_weighted_solve(self):
        checkpoint_infos = [{
            "step": 5,
            "learning_rate": 0.1,
            "test_infos": [{
                "grad": torch.tensor([1.0, 2.0], dtype=torch.float32),
            }],
            "train_infos": [
                {
                    "grad": torch.tensor([2.0, 0.0], dtype=torch.float32),
                    "geometry_feature": torch.tensor([1.0, 0.0], dtype=torch.float32),
                },
                {
                    "grad": torch.tensor([0.0, 3.0], dtype=torch.float32),
                    "geometry_feature": torch.tensor([0.0, 2.0], dtype=torch.float32),
                },
            ],
        }]

        fisher = TrajectoryFisherInfluence(lambda_damp=0.5, normalize=False)
        matrix, breakdown = fisher.compute_matrix(checkpoint_infos, return_breakdown=True)

        features = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        grads = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        g_test = np.asarray([1.0, 2.0], dtype=np.float32)
        feature_weights = np.asarray([0.5, 0.5], dtype=np.float32)
        weighted_features = features * np.sqrt(feature_weights)[:, None]
        fisher_matrix = 0.5 * np.eye(2, dtype=np.float32) + weighted_features.T @ weighted_features
        h_inv_g = np.linalg.solve(fisher_matrix, g_test)
        expected = 0.1 * np.asarray([np.dot(h_inv_g, grad) for grad in grads], dtype=np.float32)

        self.assertEqual(matrix.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(matrix)))
        np.testing.assert_allclose(matrix[0], expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(breakdown), 1)
        self.assertEqual(int(breakdown[0]["step"]), 5)

    def test_full_and_woodbury_fisher_match_on_small_problem(self):
        train_infos = [
            {
                "grad": torch.tensor([2.0, 0.0], dtype=torch.float32),
                "geometry_feature": torch.tensor([1.0, 0.0], dtype=torch.float32),
            },
            {
                "grad": torch.tensor([0.0, 3.0], dtype=torch.float32),
                "geometry_feature": torch.tensor([0.0, 2.0], dtype=torch.float32),
            },
        ]
        test_info = {"grad": torch.tensor([1.0, 2.0], dtype=torch.float32)}

        full = MeanScoreFisherInfluence(train_infos, lambda_damp=0.5, normalize=False, solver="full")
        woodbury = MeanScoreFisherInfluence(train_infos, lambda_damp=0.5, normalize=False, solver="woodbury")

        np.testing.assert_allclose(
            full.compute_all_scores(test_info),
            woodbury.compute_all_scores(test_info),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_trajectory_full_solver_matches_woodbury_solver(self):
        checkpoint_infos = [{
            "step": 5,
            "learning_rate": 0.1,
            "test_infos": [{
                "grad": torch.tensor([1.0, 2.0], dtype=torch.float32),
            }],
            "train_infos": [
                {
                    "grad": torch.tensor([2.0, 0.0], dtype=torch.float32),
                    "geometry_feature": torch.tensor([1.0, 0.0], dtype=torch.float32),
                },
                {
                    "grad": torch.tensor([0.0, 3.0], dtype=torch.float32),
                    "geometry_feature": torch.tensor([0.0, 2.0], dtype=torch.float32),
                },
            ],
        }]

        woodbury = TrajectoryFisherInfluence(lambda_damp=0.5, normalize=False, solver="woodbury")
        full = TrajectoryFisherInfluence(lambda_damp=0.5, normalize=False, solver="full")

        np.testing.assert_allclose(
            full.compute_matrix(checkpoint_infos),
            woodbury.compute_matrix(checkpoint_infos),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
