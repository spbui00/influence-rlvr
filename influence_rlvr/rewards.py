import json
import re
import subprocess
import sys
from fractions import Fraction

_THINK_OPEN_PATTERN = re.compile(r"^\s*<think>", re.IGNORECASE)
_THINK_CLOSE_PATTERN = re.compile(r"</think>", re.IGNORECASE)
_ANSWER_TAG_PATTERN = re.compile(
    r"<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)
_ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*=(.+)$")
_LATEX_FRAC_PATTERN = re.compile(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}")
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


def _answer_region_after_think(text):
    match = _THINK_CLOSE_PATTERN.search(text)
    if match is None:
        return text
    return text[match.end():]


def _extract_boxed_answer(text):
    markers = (r"\boxed{", "boxed{")
    last_match = None
    last_index = -1
    for marker in markers:
        marker_index = text.rfind(marker)
        if marker_index > last_index:
            last_index = marker_index
            last_match = marker
    if last_match is None or last_index < 0:
        return None

    cursor = last_index + len(last_match)
    depth = 1
    chars = []
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        cursor += 1
    return None


def _clean_math_answer_text(text):
    cleaned = str(text).strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.rstrip(".。")
    cleaned = cleaned.strip()
    if cleaned.startswith("Final answer:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    cleaned = cleaned.replace(" ", "")
    assignment_match = _ASSIGNMENT_PATTERN.fullmatch(cleaned)
    if assignment_match is not None:
        cleaned = assignment_match.group(1)
    return cleaned


def _fraction_from_latex(text):
    match = _LATEX_FRAC_PATTERN.fullmatch(text)
    if match is None:
        return None
    try:
        numerator = Fraction(match.group(1))
        denominator = Fraction(match.group(2))
        if denominator == 0:
            return None
        return numerator / denominator
    except (ValueError, ZeroDivisionError):
        return None


def _parse_numeric_answer(text):
    cleaned = _clean_math_answer_text(text)
    if not cleaned:
        return None
    if cleaned.endswith("%"):
        try:
            return Fraction(cleaned[:-1]) / 100
        except (ValueError, ZeroDivisionError):
            return None
    latex_fraction = _fraction_from_latex(cleaned)
    if latex_fraction is not None:
        return latex_fraction
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def _normalize_symbolic_answer(text):
    return _clean_math_answer_text(text).lower()


def extract_math_final_answer(text):
    answer_region = _answer_region_after_think(text)
    boxed_answer = _extract_boxed_answer(answer_region)
    if boxed_answer is not None:
        return boxed_answer.strip()

    boxed_answer = _extract_boxed_answer(text)
    if boxed_answer is not None:
        return boxed_answer.strip()

    answer_tag = _extract_answer_tag(answer_region)
    if answer_tag is not None:
        return answer_tag

    answer_tag = _extract_answer_tag(text)
    if answer_tag is not None:
        return answer_tag

    stripped_region = answer_region.strip()
    if stripped_region:
        lines = [line.strip() for line in stripped_region.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            if ":" in last_line and last_line.lower().startswith(("answer", "final answer")):
                return last_line.split(":", 1)[1].strip()
            if _parse_numeric_answer(last_line) is not None:
                return last_line
    return None


def _answers_match(model_answer, true_answer):
    if model_answer is None:
        return False
    model_numeric = _parse_numeric_answer(model_answer)
    true_numeric = _parse_numeric_answer(true_answer)
    if model_numeric is not None and true_numeric is not None:
        return model_numeric == true_numeric
    return _normalize_symbolic_answer(model_answer) == _normalize_symbolic_answer(true_answer)


def math_answer_equivalence_key(model_answer):
    if model_answer is None:
        return "__none__"
    model_numeric = _parse_numeric_answer(model_answer)
    if model_numeric is not None:
        return str(model_numeric)
    return _normalize_symbolic_answer(model_answer)


def _has_r1_reasoning_format(text):
    stripped = text.strip()
    if not _THINK_OPEN_PATTERN.match(stripped):
        return False
    answer_region = _answer_region_after_think(stripped).strip()
    if not answer_region:
        return False
    return _extract_boxed_answer(answer_region) is not None


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


def format_reward_func(completions, **kwargs):
    return [
        1.0 if _has_r1_reasoning_format(response) else 0.0
        for response in _extract_responses(completions)
    ]


def format_guardrail_reward_func(completions, **kwargs):
    rewards = []
    for response in _extract_responses(completions):
        tl = response.lower()
        boxed = r"\boxed{" in response or "boxed{" in response
        if "<think>" in tl and "</think>" in tl and boxed:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward_func(completions, solution, **kwargs):
    rewards = []
    for response, true_answer in zip(_extract_responses(completions), solution):
        model_answer = extract_math_final_answer(response)
        rewards.append(1.0 if _answers_match(model_answer, true_answer) else 0.0)
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
