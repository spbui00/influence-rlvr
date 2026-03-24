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
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_PYTHON_START_PATTERN = re.compile(
    r"(?m)^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+\s*\(|class\s+\w+\s*[:(])"
)


def _extract_responses(completions):
    return [completion[0]["content"] for completion in completions]


def _extract_answer_tag(text):
    match = _ANSWER_TAG_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _extract_python_code(text):
    fenced = _CODE_BLOCK_PATTERN.search(text)
    if fenced:
        return fenced.group(1).strip()

    start = _PYTHON_START_PATTERN.search(text)
    if start:
        return text[start.start():].strip()

    return text.strip()


def _run_code_tests(code, test_setup_code, tests):
    namespace = {"__builtins__": __builtins__}
    if test_setup_code:
        exec(test_setup_code, namespace, namespace)
    exec(code, namespace, namespace)

    passed = 0
    for test in tests:
        exec(test, namespace, namespace)
        passed += 1
    return passed


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


def mbpp_execution_reward_func(
    completions,
    test_list,
    test_setup_code="",
    challenge_test_list=None,
    **kwargs,
):
    rewards = []
    challenge_test_list = challenge_test_list or []
    tests = list(test_list) + list(challenge_test_list)

    for response in _extract_responses(completions):
        code = _extract_python_code(response)
        if not code or not tests:
            rewards.append(0.0)
            continue

        try:
            passed = _run_code_tests(code, test_setup_code, tests)
            rewards.append(passed / len(tests))
        except Exception:
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
