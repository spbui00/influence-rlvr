import unittest

from influence_rlvr.modes import GeometryFeatureMode, GradientObjective
from influence_rlvr.toy_grpo import (
    AutoregressiveLogisticRegression,
    ToyRolloutMode,
    build_user_plan_sandbox,
    compute_toy_fisher_influence,
    compute_toy_gradient_bundle,
    compute_toy_historical_fisher_influence,
    initialize_toy_model,
    train_toy_grpo,
)


class ToyGRPOInfluenceTests(unittest.TestCase):
    def test_exhaustive_bundle_uses_all_binary_sequences(self):
        sandbox = build_user_plan_sandbox()
        model = AutoregressiveLogisticRegression(use_bias=False)
        initialize_toy_model(model, mode="zero")

        bundle = compute_toy_gradient_bundle(
            model,
            sandbox.train_examples[0],
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            objective_mode=GradientObjective.GRPO_TRAIN,
            geometry_feature_mode=GeometryFeatureMode.POLICY_SCORE,
        )

        self.assertEqual(bundle["debug"]["responses"], ["00", "01", "10", "11"])
        self.assertEqual(len(bundle["debug"]["total_rewards"]), 4)
        self.assertEqual(bundle["grad"].shape, bundle["geometry_feature"].shape)
        self.assertGreater(bundle["grad"].numel(), 0)

    def test_toy_fisher_matches_explicit_dense_solve(self):
        sandbox = build_user_plan_sandbox()
        model = AutoregressiveLogisticRegression(use_bias=False)
        initialize_toy_model(model, mode="zero")

        result = compute_toy_fisher_influence(
            model,
            train_examples=sandbox.train_examples,
            test_example=sandbox.test_example,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            lambda_damp=0.25,
        )

        self.assertEqual(result["trajectory_fisher_matrix"].shape, (1, 3))
        self.assertEqual(len(result["train_infos"]), 3)
        for repo, dense in zip(result["repo_scores"], result["dense_repo_scores"]):
            self.assertAlmostEqual(float(repo), float(dense), places=5)

    def test_train_toy_grpo_saves_requested_checkpoints(self):
        sandbox = build_user_plan_sandbox()
        model = AutoregressiveLogisticRegression(use_bias=False)
        initialize_toy_model(model, mode="zero")

        result = train_toy_grpo(
            model,
            sandbox.train_examples,
            steps=3,
            lr=0.1,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            checkpoint_steps=(0, 2, 3),
        )

        self.assertEqual(len(result["history"]), 3)
        self.assertEqual(sorted(result["checkpoints"]), [0, 2, 3])

    def test_historical_toy_fisher_sums_over_training_steps(self):
        sandbox = build_user_plan_sandbox()
        model = AutoregressiveLogisticRegression(use_bias=False)
        initialize_toy_model(model, mode="zero")

        result = train_toy_grpo(
            model,
            sandbox.train_examples,
            steps=3,
            lr=0.1,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            checkpoint_steps=(0, 1, 2, 3),
        )

        historical = compute_toy_historical_fisher_influence(
            model,
            checkpoints=result["checkpoints"],
            train_history=result["history"],
            train_examples=sandbox.train_examples,
            test_example=sandbox.test_example,
            learning_rate=0.1,
            end_step=3,
            rollout_mode=ToyRolloutMode.EXHAUSTIVE,
            lambda_damp=0.25,
        )

        self.assertEqual(len(historical["historical_scores"]), 3)
        self.assertEqual(
            [row.occurrence_count for row in historical["historical_scores"]],
            [1, 1, 1],
        )
        self.assertAlmostEqual(
            sum(row.repo_fisher_score for row in historical["historical_scores"]),
            float(historical["repo_scores"].sum()),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
