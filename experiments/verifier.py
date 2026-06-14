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

import os
import re
from functools import lru_cache

import requests
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


class VLLMServerVerifier:
    """Verifier backed by a standalone OpenAI-compatible vLLM server.

    The general-verifier is a FIXED model (never updated during training), so it
    serves trivially — `vllm serve TIGER-Lab/general-verifier` on its own GPU, no
    weight sync. vLLM's continuous batching makes grading ~10-20x faster than the
    in-process HF path (where the batch stalls on the slowest "Final Decision:"),
    so we can run a high max_new_tokens without truncating the verifier's reasoning
    before its decision line. Same interface as GeneralVerifier (incl. .tokenizer,
    which the GR reward uses for the length penalty)."""

    def __init__(self, model_id: str, *, host: str = "127.0.0.1", port: int = 8100,
                 max_new_tokens: int = 1024, timeout: float = 600.0):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.base_url = f"http://{host}:{port}/v1"
        self.timeout = timeout
        # tokenizer for chat-templating the prompts + token counting in the reward
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render(self, question: str, ground_truth: str, student_answer: str) -> str:
        body = _VERIFIER_TEMPLATE.format(
            question=question, ground_truth=ground_truth, student_answer=student_answer,
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True,
        )

    @staticmethod
    def _decisions_from_choices(choices: list, n: int) -> list[float]:
        """Map a vLLM /v1/completions `choices` array (each carries its input
        `index`) back to per-prompt decisions, order-safe."""
        texts = [""] * n
        for c in choices:
            i = int(c.get("index", 0))
            if 0 <= i < n:
                texts[i] = c.get("text", "") or ""
        return [GeneralVerifier._parse_decision(t) for t in texts]

    @torch.inference_mode()
    def verify_batch(self, questions, ground_truths, student_answers) -> list[float]:
        assert len(questions) == len(ground_truths) == len(student_answers)
        if not questions:
            return []
        prompts = [
            self._render(q, g, s)
            for q, g, s in zip(questions, ground_truths, student_answers)
        ]
        resp = requests.post(
            f"{self.base_url}/completions",
            json={
                "model": self.model_id, "prompt": prompts,
                "max_tokens": self.max_new_tokens, "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            # A 400 here is almost always the verifier's context-length cap: a long
            # completion + question + the max_new_tokens verdict budget exceeding the
            # server's --max-model-len. One oversized batch must NOT kill a multi-hour
            # run, so log it and score the batch neutral (0.0). Root fix = a large
            # enough verifier --max-model-len; this is the backstop. Others still raise.
            if resp.status_code == 400:
                print(f"[verifier] 400 from {self.base_url}/completions "
                      f"(likely context-length); scoring {len(prompts)} as 0.0. "
                      f"detail: {resp.text[:200]}")
                return [0.0] * len(prompts)
            raise
        return self._decisions_from_choices(resp.json()["choices"], len(prompts))


@lru_cache(maxsize=2)
def get_verifier(model_id: str, device: str | None, max_new_tokens: int,
                 batch_size: int) -> GeneralVerifier:
    return GeneralVerifier(
        model_id, device=device, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )


@lru_cache(maxsize=2)
def get_vllm_verifier(model_id: str, host: str, port: int,
                      max_new_tokens: int) -> VLLMServerVerifier:
    return VLLMServerVerifier(model_id, host=host, port=port, max_new_tokens=max_new_tokens)


def get_verifier_from_config(cfg: ExperimentConfig):
    """Return the configured verifier (HF in-process, or the vLLM HTTP server)."""
    if cfg.verifier_backend == "vllm":
        return get_vllm_verifier(
            cfg.verifier_model_id, cfg.verifier_server_host,
            cfg.verifier_server_port, cfg.verifier_max_new_tokens,
        )
    return get_verifier(
        cfg.verifier_model_id, cfg.verifier_device,
        cfg.verifier_max_new_tokens, cfg.verifier_batch_size,
    )


def _student_answer(text: str) -> str:
    """Extract the student's final answer; fall back to the raw completion."""
    ans = extract_math_final_answer(text)
    return ans if ans is not None else text.strip()


def _length_penalty(tokenizer, answer: str, gold: str, coef: float, cap: int) -> float:
    """General-Reasoner length penalty: coef * min(cap, |tok(answer) - tok(gold)|),
    measured in tokenizer tokens of the extracted answer vs the reference."""
    if coef <= 0.0:
        return 0.0
    la = len(tokenizer.encode(answer, add_special_tokens=False))
    lg = len(tokenizer.encode(str(gold), add_special_tokens=False))
    return coef * min(cap, abs(la - lg))


def _gr_score(extracted, gold, verdict, tokenizer, *,
              extraction_penalty: float, length_coef: float, length_cap: int) -> float:
    """Per-rollout General-Reasoner reward (pure; no model call).

      extracted is None (no \\boxed{}/marker) -> -extraction_penalty
      verifier correct (verdict >= 0.5)        -> +1 - length_penalty
      verifier wrong                           ->  0
    """
    if extracted is None or not str(extracted).strip():
        return -extraction_penalty   # empty \boxed{} is NOT a valid answer (reward-hack guard)
    if verdict >= 0.5:
        return 1.0 - _length_penalty(tokenizer, extracted, gold, length_coef, length_cap)
    return 0.0


def make_verifier_reward_func(cfg: ExperimentConfig):
    """TRL GRPO reward matching General-Reasoner (arXiv 2505.14652 — the
    WebInstruct-verified authors, same Qwen3-4B-Base + GRPO, no SFT).

    TRL passes dataset columns as kwargs, so rows must carry `question` and
    `solution` (see data.py). Per rollout: extract the boxed/marked final answer;
    if none is extractable the reward is `-extraction_penalty` (default -0.5,
    without a verifier call); otherwise the general-verifier judges equivalence to
    the reference -> +1 minus a small length penalty when correct, 0 when wrong.
    The -0.5 keeps outputs parseable AND injects within-group variance so all-wrong
    groups still have a gradient. Pure verifier-only {0,1} is recovered with
    extraction_penalty=0 and length_penalty_coef=0.

    Diagnostics: every step prints a one-line breakdown (no_box/wrong/correct/
    box_rate/acc/mean). Set IFRLVR_DEBUG_SAMPLES=2 to also dump a couple of full
    rollouts (completion tail + extracted answer + gold + verdict) every 5 steps.
    """
    _debug_samples = int(os.environ.get("IFRLVR_DEBUG_SAMPLES", "0"))
    _step = [0]

    def verifier_reward_func(completions, question=None, solution=None, **kwargs):
        if question is None or solution is None:
            raise ValueError(
                "verifier_reward_func needs `question` and `solution` columns "
                "in the dataset (see experiments/data.py)."
            )
        responses = [c[0]["content"] for c in completions]
        extracted = [extract_math_final_answer(r) for r in responses]
        verifier = get_verifier_from_config(cfg)

        # Call the heavyweight verifier ONLY on rollouts with an extractable answer;
        # unextractable ones get the penalty without a verifier pass.
        idx = [i for i, a in enumerate(extracted) if a is not None]
        verdicts: dict[int, float] = {}
        if idx:
            v = verifier.verify_batch(
                [question[i] for i in idx],
                [solution[i] for i in idx],
                [extracted[i] for i in idx],
            )
            verdicts = dict(zip(idx, v))

        scores = [
            _gr_score(
                extracted[i], solution[i], verdicts.get(i, 0.0), verifier.tokenizer,
                extraction_penalty=cfg.extraction_penalty,
                length_coef=cfg.length_penalty_coef,
                length_cap=cfg.length_penalty_cap,
            )
            for i in range(len(responses))
        ]

        # --- per-step diagnostic breakdown (rank 0 only under DP; cheap) ---
        if os.environ.get("RANK", "0") == "0":
            n = len(responses) or 1
            n_box = sum(a is not None for a in extracted)
            n_correct = sum(1 for i in range(len(responses)) if verdicts.get(i, 0.0) >= 0.5)
            print(f"[reward] step~{_step[0]} n={len(responses)} no_box={len(responses) - n_box} "
                  f"wrong={n_box - n_correct} correct={n_correct} "
                  f"box_rate={n_box / n:.2f} acc={n_correct / n:.2f} mean={sum(scores) / n:+.3f}",
                  flush=True)
            # --- optional: dump a few full rollouts (IFRLVR_DEBUG_SAMPLES=2) ---
            if _debug_samples and _step[0] % 5 == 0:
                for i in range(min(_debug_samples, len(responses))):
                    print(f"  [sample {i}] boxed={extracted[i] is not None} "
                          f"extracted={extracted[i]!r} gold={solution[i]!r} "
                          f"verdict={verdicts.get(i, 0.0)} score={scores[i]:+.2f}", flush=True)
                    print(f"    completion(tail 600): ...{responses[i][-600:]}", flush=True)
        _step[0] += 1
        return scores

    # Name shows up in TRL/W&B reward logs.
    verifier_reward_func.__name__ = "general_reasoner_reward"
    return verifier_reward_func
