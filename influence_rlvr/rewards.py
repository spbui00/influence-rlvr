import re

_THINK_ANSWER_PATTERN = re.compile(
    r"^<think>.*?</think><answer>.*?</answer>$",
    re.DOTALL | re.IGNORECASE,
)
_ANSWER_TAG_PATTERN = re.compile(
    r"<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"[\d,]+\.?\d*")


def _extract_responses(completions):
    return [completion[0]["content"] for completion in completions]


def _extract_answer_tag(text):
    match = _ANSWER_TAG_PATTERN.search(text)
    return match.group(1).strip() if match else None


# ── Strict reward functions (binary) ─────────────────────────────────────────

def format_reward_func(completions, **kwargs):
    return [
        1.0 if _THINK_ANSWER_PATTERN.match(r) else 0.0
        for r in _extract_responses(completions)
    ]


def accuracy_reward_func(completions, solution, **kwargs):
    rewards = []
    for response, true_answer in zip(_extract_responses(completions), solution):
        model_answer = _extract_answer_tag(response)
        if model_answer is not None and model_answer == true_answer.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ── Soft reward functions (partial credit) ────────────────────────────────────

_FORMAT_TAGS = ["<think>", "</think>", "<answer>", "</answer>"]


def soft_format_reward_func(completions, **kwargs):
    rewards = []
    for r in _extract_responses(completions):
        lower = r.lower()
        score = sum(0.25 for tag in _FORMAT_TAGS if tag in lower)
        rewards.append(score)
    return rewards


def soft_accuracy_reward_func(completions, solution, **kwargs):
    rewards = []
    for response, true_answer in zip(_extract_responses(completions), solution):
        true_answer = true_answer.strip()

        model_answer = _extract_answer_tag(response)
        if model_answer is not None and model_answer == true_answer:
            rewards.append(1.0)
            continue

        answer_nums = set(_NUMBER_PATTERN.findall(true_answer))
        response_nums = set(_NUMBER_PATTERN.findall(response))

        if answer_nums and answer_nums.issubset(response_nums):
            rewards.append(0.5)
        elif answer_nums & response_nums:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards
