"""Model-based reward: TIGER-Lab/general-verifier as the GRPO verifier.

General-Reasoner trains with a generative verifier that judges whether a model's
final answer is *equivalent* to the reference answer (works for free-form answers
across domains, unlike exact match). We wrap that model here and expose a
TRL-compatible reward function.

Verifier prompt format (from the model card):

    User: ### Question: {question}

    ### Ground Truth Answer: {ground_truth}

    ### Student Answer: {student_answer}

    For the above question, please verify if the student's answer is equivalent
    to the ground truth answer.
    Do not solve the question by yourself; just check if the student's answer is
    equivalent to the ground truth answer.
    If the student's answer is correct, output "Final Decision: Yes". If the
    student's answer is incorrect, output "Final Decision: No". Assistant:

We feed that via the verifier's chat template and parse "Final Decision: Yes".

The verifier is a heavyweight object (a 1.5B model on GPU). It is created once
and cached as a module-level singleton keyed by (model_id, device) so the TRL
reward callback can be a thin closure.
"""
from __future__ import annotations

import re
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from influence_rlvr.rewards import extract_math_final_answer

_DECISION_RE = re.compile(r"final\s*decision\s*:\s*(yes|no)", re.IGNORECASE)

_VERIFIER_TEMPLATE = (
    "### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent "
    "to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is "
    "equivalent to the ground truth answer.\n"
    'If the student\'s answer is correct, output "Final Decision: Yes". '
    'If the student\'s answer is incorrect, output "Final Decision: No".'
)


class GeneralVerifier:
    """Batched generative verifier returning per-pair correctness in {0.0, 1.0}."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        max_new_tokens: int = 512,
        batch_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _render(self, question: str, ground_truth: str, student_answer: str) -> str:
        body = _VERIFIER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            student_answer=student_answer,
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _parse_decision(text: str) -> float:
        # Take the *last* decision token the verifier emits.
        matches = _DECISION_RE.findall(text)
        if not matches:
            return 0.0
        return 1.0 if matches[-1].lower() == "yes" else 0.0

    @torch.inference_mode()
    def verify_batch(
        self,
        questions: list[str],
        ground_truths: list[str],
        student_answers: list[str],
    ) -> list[float]:
        assert len(questions) == len(ground_truths) == len(student_answers)
        rewards: list[float] = []
        for start in range(0, len(questions), self.batch_size):
            sl = slice(start, start + self.batch_size)
            prompts = [
                self._render(q, g, s)
                for q, g, s in zip(questions[sl], ground_truths[sl], student_answers[sl])
            ]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=4096,
            ).to(self.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            rewards.extend(self._parse_decision(t) for t in texts)
        return rewards


@lru_cache(maxsize=2)
def get_verifier(model_id: str, device: str | None, max_new_tokens: int,
                 batch_size: int) -> GeneralVerifier:
    return GeneralVerifier(
        model_id, device=device, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )


def get_verifier_from_config(cfg: ExperimentConfig) -> GeneralVerifier:
    return get_verifier(
        cfg.verifier_model_id, cfg.verifier_device,
        cfg.verifier_max_new_tokens, cfg.verifier_batch_size,
    )


def _student_answer(text: str) -> str:
    """Extract the student's final answer; fall back to the raw completion."""
    ans = extract_math_final_answer(text)
    return ans if ans is not None else text.strip()


def make_verifier_reward_func(cfg: ExperimentConfig):
    """Build a TRL GRPO reward function backed by the general-verifier.

    TRL passes dataset columns as kwargs, so the training rows must carry
    `question` and `solution` (see data.py). Returns rewards in {0.0, 1.0}.
    """
    def verifier_reward_func(completions, question=None, solution=None, **kwargs):
        if question is None or solution is None:
            raise ValueError(
                "verifier_reward_func needs `question` and `solution` columns "
                "in the dataset (see experiments/data.py)."
            )
        responses = [c[0]["content"] for c in completions]
        students = [_student_answer(r) for r in responses]
        verifier = get_verifier_from_config(cfg)
        return verifier.verify_batch(list(question), list(solution), students)

    # Name shows up in TRL/W&B reward logs.
    verifier_reward_func.__name__ = "general_verifier_reward"
    return verifier_reward_func
