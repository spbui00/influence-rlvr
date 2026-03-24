import json
import re
import subprocess
import sys

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
_MBPP_RESULT_PREFIX = "__MBPP_RESULT__"
_DEFAULT_MBPP_TIMEOUT_SECONDS = 5.0
_MBPP_COMPILES_BONUS = 0.1
_MBPP_EXEC_RUNNER = f"""
import builtins
import contextlib
import io
import json
import sys

payload = json.loads(sys.stdin.read())

class _BlockedInput:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Interactive input is disabled during MBPP reward evaluation.")

class _BlockedStdin(io.TextIOBase):
    def read(self, *args, **kwargs):
        raise RuntimeError("Interactive stdin is disabled during MBPP reward evaluation.")

    def readline(self, *args, **kwargs):
        raise RuntimeError("Interactive stdin is disabled during MBPP reward evaluation.")

    def readlines(self, *args, **kwargs):
        raise RuntimeError("Interactive stdin is disabled during MBPP reward evaluation.")

class _DiscardIO(io.TextIOBase):
    def write(self, text):
        return len(text)

builtins.input = _BlockedInput()
sys.stdin = _BlockedStdin()

namespace = {{"__builtins__": builtins.__dict__, "__name__": "__main__"}}
passed = 0
code_loaded = False

try:
    with contextlib.redirect_stdout(_DiscardIO()), contextlib.redirect_stderr(_DiscardIO()):
        if payload["test_setup_code"]:
            exec(payload["test_setup_code"], namespace, namespace)
        exec(payload["code"], namespace, namespace)
        code_loaded = True
        for test in payload["tests"]:
            exec(test, namespace, namespace)
            passed += 1
except Exception as exc:
    result = {{
        "passed": passed,
        "code_loaded": code_loaded,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }}
else:
    result = {{
        "passed": passed,
        "code_loaded": True,
        "error_type": None,
        "error": None,
    }}

print({json.dumps(_MBPP_RESULT_PREFIX)} + json.dumps(result))
"""


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


def _run_code_tests(code, test_setup_code, tests, timeout_seconds):
    payload = json.dumps({
        "code": code,
        "test_setup_code": test_setup_code or "",
        "tests": list(tests),
    })
    try:
        result = subprocess.run(
            [sys.executable, "-c", _MBPP_EXEC_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"MBPP execution timed out after {timeout_seconds:.1f}s."
        ) from exc

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if not message:
            message = f"MBPP subprocess exited with code {result.returncode}."
        raise RuntimeError(message)

    result_line = None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_MBPP_RESULT_PREFIX):
            result_line = line[len(_MBPP_RESULT_PREFIX):]
            break
    if result_line is None:
        raise RuntimeError("MBPP subprocess did not return a result payload.")

    execution_result = json.loads(result_line)
    if execution_result["error_type"] is not None:
        if execution_result.get("code_loaded", False):
            return execution_result["passed"], True
        raise RuntimeError(
            f'{execution_result["error_type"]}: {execution_result["error"]}'
        )
    return execution_result["passed"], True


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
    timeout_seconds=_DEFAULT_MBPP_TIMEOUT_SECONDS,
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
            passed, code_loaded = _run_code_tests(
                code,
                test_setup_code,
                tests,
                timeout_seconds=timeout_seconds,
            )
            if passed > 0:
                rewards.append(
                    _MBPP_COMPILES_BONUS
                    + (1.0 - _MBPP_COMPILES_BONUS) * (passed / len(tests))
                )
            elif code_loaded:
                rewards.append(_MBPP_COMPILES_BONUS)
            else:
                rewards.append(0.0)
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
