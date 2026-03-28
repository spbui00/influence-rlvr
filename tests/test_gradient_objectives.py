import unittest

import torch

from influence_rlvr.gradients import (
    _compute_expected_reward_policy_loss,
    _compute_grpo_policy_loss,
)


class GradientObjectiveTests(unittest.TestCase):
    def test_expected_reward_objective_is_reward_weighted_sequence_logprob(self):
        rewards = torch.tensor([1.0, 0.5], dtype=torch.float32)
        sequence_log_probs = torch.tensor([-2.0, -4.0], dtype=torch.float32)

        loss = _compute_expected_reward_policy_loss(rewards, sequence_log_probs)

        expected = -((1.0 * -2.0) + (0.5 * -4.0)) / 2.0
        self.assertAlmostEqual(loss.item(), expected, places=6)

    def test_grpo_objective_matches_ratio_one_update_at_current_policy(self):
        total_rewards = torch.tensor([1.0, -1.0], dtype=torch.float32)
        per_token_logps = torch.tensor([[0.2], [0.3]], dtype=torch.float32)
        response_mask = torch.tensor([[1], [1]], dtype=torch.long)

        loss, advantages, per_token_kl = _compute_grpo_policy_loss(
            total_rewards,
            per_token_logps,
            per_token_logps.detach(),
            response_mask,
            epsilon=0.2,
            beta=0.0,
            ref_per_token_logps=None,
            advantage_eps=1e-4,
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        self.assertEqual(advantages.shape, total_rewards.shape)
        self.assertTrue(torch.allclose(per_token_kl, torch.zeros_like(per_token_kl)))


if __name__ == "__main__":
    unittest.main()
