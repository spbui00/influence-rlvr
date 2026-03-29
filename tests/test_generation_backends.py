import unittest
from unittest.mock import patch

import torch

from analysis.loader import build_cache_fingerprint
from influence_rlvr.generation import RolloutBatch, rollout_batch_from_token_sequences
from influence_rlvr.gradients import compute_policy_gradient_bundle
from influence_rlvr.modes import GenerationBackend


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    eos_token = "<eos>"
    name_or_path = "dummy"

    def decode(self, token_ids, skip_special_tokens=True):
        values = []
        for token_id in token_ids:
            token_value = int(token_id)
            if skip_special_tokens and token_value in {self.pad_token_id, self.eos_token_id}:
                continue
            values.append(str(token_value))
        return " ".join(values)


class GenerationBackendTests(unittest.TestCase):
    def test_rollout_batch_from_token_sequences_preserves_exact_ids(self):
        tokenizer = _DummyTokenizer()
        rollout = rollout_batch_from_token_sequences(
            tokenizer,
            [
                torch.tensor([11, 12, 99], dtype=torch.long),
                torch.tensor([21, 99], dtype=torch.long),
            ],
            device="cpu",
        )

        self.assertEqual(rollout.token_ids.tolist(), [[11, 12, 99], [21, 99, 0]])
        self.assertEqual(rollout.response_mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(rollout.texts, ["11 12", "21"])

    def test_policy_gradient_bundle_uses_exact_rollout_token_ids(self):
        tokenizer = _DummyTokenizer()
        model = torch.nn.Linear(1, 1)
        rollout = RolloutBatch(
            texts=["sample-a", "sample-b"],
            token_ids=torch.tensor([[11, 12, 99], [21, 99, 0]], dtype=torch.long),
            response_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
        )
        prompt_ids = torch.tensor([[7, 8]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt_ids)
        captured = {}

        def reward_fn(completions):
            return [1.0 if item[0]["content"] == "sample-a" else 0.5 for item in completions]

        def fake_logps(_model, _prompt_ids, _prompt_attention_mask, response_ids, response_mask):
            captured["response_ids"] = response_ids.clone()
            captured["response_mask"] = response_mask.clone()
            return torch.tensor(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]],
                dtype=torch.float32,
            )

        def fake_grad_vector(_model, scalar, *, retain_graph=False):
            return torch.tensor([float(scalar.detach().cpu())], dtype=torch.float32)

        with (
            patch(
                "influence_rlvr.gradients.tokenize_prompt",
                return_value=("prompt-text", prompt_ids, prompt_mask),
            ),
            patch(
                "influence_rlvr.gradients.generate_rollout_batch",
                return_value=rollout,
            ),
            patch(
                "influence_rlvr.gradients._compute_per_token_logps",
                side_effect=fake_logps,
            ),
            patch(
                "influence_rlvr.gradients._grad_vector_from_scalar",
                side_effect=fake_grad_vector,
            ),
        ):
            result = compute_policy_gradient_bundle(
                model,
                tokenizer,
                "prompt",
                [reward_fn],
                G=2,
                device="cpu",
                generation_backend=GenerationBackend.HF,
            )

        self.assertTrue(torch.equal(captured["response_ids"], rollout.token_ids))
        self.assertTrue(torch.equal(captured["response_mask"], rollout.response_mask))
        self.assertEqual(result["debug"]["responses"], rollout.texts)
        self.assertEqual(result["debug"]["response_lengths"], [3, 2])

    def test_policy_gradient_bundle_matches_for_fixed_rollout_across_backends(self):
        tokenizer = _DummyTokenizer()
        model = torch.nn.Linear(1, 1)
        rollout = RolloutBatch(
            texts=["sample-a", "sample-b"],
            token_ids=torch.tensor([[31, 32, 99], [41, 99, 0]], dtype=torch.long),
            response_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
        )
        prompt_ids = torch.tensor([[3, 4]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt_ids)

        def reward_fn(completions):
            return [1.0 if item[0]["content"] == "sample-a" else 0.25 for item in completions]

        def fake_logps(_model, _prompt_ids, _prompt_attention_mask, _response_ids, _response_mask):
            return torch.tensor(
                [[0.2, 0.3, 0.1], [0.4, 0.1, 0.0]],
                dtype=torch.float32,
            )

        def fake_grad_vector(_model, scalar, *, retain_graph=False):
            value = float(scalar.detach().cpu())
            return torch.tensor([value, value + 1.0], dtype=torch.float32)

        patchers = (
            patch(
                "influence_rlvr.gradients.tokenize_prompt",
                return_value=("prompt-text", prompt_ids, prompt_mask),
            ),
            patch(
                "influence_rlvr.gradients.generate_rollout_batch",
                return_value=rollout,
            ),
            patch(
                "influence_rlvr.gradients._compute_per_token_logps",
                side_effect=fake_logps,
            ),
            patch(
                "influence_rlvr.gradients._grad_vector_from_scalar",
                side_effect=fake_grad_vector,
            ),
        )
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            hf_result = compute_policy_gradient_bundle(
                model,
                tokenizer,
                "prompt",
                [reward_fn],
                G=2,
                device="cpu",
                generation_backend=GenerationBackend.HF,
            )
            vllm_result = compute_policy_gradient_bundle(
                model,
                tokenizer,
                "prompt",
                [reward_fn],
                G=2,
                device="cpu",
                enable_vllm=True,
                generation_backend=GenerationBackend.VLLM,
            )

        self.assertTrue(torch.equal(hf_result["grad"], vllm_result["grad"]))
        self.assertEqual(hf_result["debug"]["responses"], vllm_result["debug"]["responses"])
        self.assertEqual(
            hf_result["debug"]["sequence_log_probs"],
            vllm_result["debug"]["sequence_log_probs"],
        )
        self.assertAlmostEqual(
            hf_result["debug"]["policy_loss"],
            vllm_result["debug"]["policy_loss"],
            places=6,
        )

    def test_cache_fingerprint_changes_when_backend_changes(self):
        base = {
            "model_id": "dummy",
            "generation_backend": "hf",
            "replay_max_new_tokens": 256,
        }
        vllm = dict(base)
        vllm["generation_backend"] = "vllm"
        vllm["vllm_gpu_memory_utilization"] = 0.9

        self.assertNotEqual(
            build_cache_fingerprint(base),
            build_cache_fingerprint(vllm),
        )


if __name__ == "__main__":
    unittest.main()
