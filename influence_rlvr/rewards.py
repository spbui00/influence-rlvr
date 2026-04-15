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
    r"```(?:python|py)?\s*(.*?)```",
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

_TACO_STDIO_RESULT_PREFIX = "__TACO_STDIO_RESULT__"
_TACO_STDIO_RUNNER = f"""
import builtins
import contextlib
import io
import json
import sys

payload = json.loads(sys.stdin.read())
stdin_buffer = io.StringIO(payload["stdin"])
stdout_buffer = io.StringIO()

def _input(prompt=None):
    line = stdin_buffer.readline()
    if line == "":
        raise EOFError("No more input.")
    return line.rstrip("\\n")

class _ReadableStdin(io.TextIOBase):
    def read(self, *args, **kwargs):
        return stdin_buffer.read(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return stdin_buffer.readline(*args, **kwargs)

    def readlines(self, *args, **kwargs):
        return stdin_buffer.readlines(*args, **kwargs)

namespace = {{"__builtins__": builtins.__dict__, "__name__": "__main__"}}
builtins.input = _input
sys.stdin = _ReadableStdin()

try:
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(payload["code"], "<user>", "exec"), namespace, namespace)
    result = {{
        "ok": True,
        "stdout": stdout_buffer.getvalue(),
        "error_type": None,
        "error": None,
    }}
except Exception as exc:
    result = {{
        "ok": False,
        "stdout": stdout_buffer.getvalue(),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }}

print({json.dumps(_TACO_STDIO_RESULT_PREFIX)} + json.dumps(result))
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


def _extract_python_code_candidates(text):
    blocks = [m.group(1).strip() for m in _CODE_BLOCK_PATTERN.finditer(text)]
    blocks = [b for b in blocks if b]
    if blocks:
        return blocks
    start = _PYTHON_START_PATTERN.search(text)
    if start:
        return [text[start.start():].strip()]
    stripped = text.strip()
    return [stripped] if stripped else []


def _extract_python_code(text):
    cand = _extract_python_code_candidates(text)
    return cand[0] if cand else ""


def _normalize_program_output(text):
    return " ".join(str(text).strip().split())


def _mbpp_reward_single_code(
    code,
    test_list,
    test_setup_code,
    challenge_test_list,
    *,
    timeout_seconds,
):
    challenge_test_list = challenge_test_list or []
    tests = list(test_list) + list(challenge_test_list)
    if not code or not tests:
        return 0.0
    try:
        passed, code_loaded = _run_code_tests(
            code,
            test_setup_code,
            tests,
            timeout_seconds=timeout_seconds,
        )
        if passed > 0:
            return (
                _MBPP_COMPILES_BONUS
                + (1.0 - _MBPP_COMPILES_BONUS) * (passed / len(tests))
            )
        if code_loaded:
            return _MBPP_COMPILES_BONUS
        return 0.0
    except Exception:
        return 0.0


def _mbpp_best_reward_for_response(
    response,
    test_list,
    test_setup_code,
    challenge_test_list,
    *,
    timeout_seconds,
):
    best_r = 0.0
    best_code = ""
    for code in _extract_python_code_candidates(response):
        r = _mbpp_reward_single_code(
            code,
            test_list,
            test_setup_code,
            challenge_test_list,
            timeout_seconds=timeout_seconds,
        )
        if r > best_r:
            best_r = r
            best_code = code
    return best_r, best_code


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


def _run_stdio_code(code, stdin_text, timeout_seconds):
    payload = json.dumps({
        "code": code,
        "stdin": stdin_text,
    })
    try:
        result = subprocess.run(
            [sys.executable, "-c", _TACO_STDIO_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"TACO stdio execution timed out after {timeout_seconds:.1f}s."
        ) from exc

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if not message:
            message = f"TACO stdio subprocess exited with code {result.returncode}."
        raise RuntimeError(message)

    result_line = None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_TACO_STDIO_RESULT_PREFIX):
            result_line = line[len(_TACO_STDIO_RESULT_PREFIX):]
            break
    if result_line is None:
        raise RuntimeError("TACO stdio subprocess did not return a result payload.")

    execution_result = json.loads(result_line)
    return (
        bool(execution_result.get("ok")),
        str(execution_result.get("stdout", "")),
        execution_result,
    )


def _stdio_reward_single_code(
    code,
    inputs,
    outputs,
    *,
    timeout_seconds,
):
    if not code or not inputs or len(inputs) != len(outputs):
        return 0.0

    passed = 0
    ran = False
    for stdin_text, expected in zip(inputs, outputs, strict=True):
        ok, stdout_text, _ = _run_stdio_code(code, stdin_text, timeout_seconds)
        ran = ran or ok
        if not ok:
            break
        if _normalize_program_output(stdout_text) == _normalize_program_output(expected):
            passed += 1

    if passed > 0:
        return _MBPP_COMPILES_BONUS + (1.0 - _MBPP_COMPILES_BONUS) * (passed / len(inputs))
    if ran:
        return _MBPP_COMPILES_BONUS
    return 0.0


def _taco_best_reward_for_response(
    response,
    *,
    code_task_format,
    test_list,
    test_setup_code,
    challenge_test_list,
    stdio_inputs,
    stdio_outputs,
    timeout_seconds,
):
    best_r = 0.0
    best_code = ""
    for code in _extract_python_code_candidates(response):
        if code_task_format == "call":
            r = _mbpp_reward_single_code(
                code,
                test_list,
                test_setup_code,
                challenge_test_list,
                timeout_seconds=timeout_seconds,
            )
        elif code_task_format == "stdio":
            try:
                r = _stdio_reward_single_code(
                    code,
                    stdio_inputs,
                    stdio_outputs,
                    timeout_seconds=timeout_seconds,
                )
            except Exception:
                r = 0.0
        else:
            r = 0.0
        if r > best_r:
            best_r = r
            best_code = code
    return best_r, best_code


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


def mbpp_execution_rewards_and_codes(
    completions,
    test_list,
    test_setup_code="",
    challenge_test_list=None,
    timeout_seconds=_DEFAULT_MBPP_TIMEOUT_SECONDS,
):
    rewards = []
    codes = []
    for response in _extract_responses(completions):
        r, code = _mbpp_best_reward_for_response(
            response,
            test_list,
            test_setup_code,
            challenge_test_list,
            timeout_seconds=timeout_seconds,
        )
        rewards.append(r)
        codes.append(code)
    return rewards, codes


def mbpp_execution_reward_func(
    completions,
    test_list,
    test_setup_code="",
    challenge_test_list=None,
    timeout_seconds=_DEFAULT_MBPP_TIMEOUT_SECONDS,
    **kwargs,
):
    rewards, _ = mbpp_execution_rewards_and_codes(
        completions,
        test_list,
        test_setup_code=test_setup_code,
        challenge_test_list=challenge_test_list,
        timeout_seconds=timeout_seconds,
    )
    return rewards


def taco_execution_rewards_and_codes(
    completions,
    *,
    code_task_format,
    test_list=None,
    test_setup_code="",
    challenge_test_list=None,
    stdio_inputs=None,
    stdio_outputs=None,
    timeout_seconds=_DEFAULT_MBPP_TIMEOUT_SECONDS,
):
    rewards = []
    codes = []
    for response in _extract_responses(completions):
        r, code = _taco_best_reward_for_response(
            response,
            code_task_format=code_task_format,
            test_list=test_list or [],
            test_setup_code=test_setup_code or "",
            challenge_test_list=challenge_test_list or [],
            stdio_inputs=stdio_inputs or [],
            stdio_outputs=stdio_outputs or [],
            timeout_seconds=timeout_seconds,
        )
        rewards.append(r)
        codes.append(code)
    return rewards, codes


def taco_execution_reward_func(
    completions,
    *,
    code_task_format,
    test_list=None,
    test_setup_code="",
    challenge_test_list=None,
    stdio_inputs=None,
    stdio_outputs=None,
    timeout_seconds=_DEFAULT_MBPP_TIMEOUT_SECONDS,
    **kwargs,
):
    rewards, _ = taco_execution_rewards_and_codes(
        completions,
        code_task_format=code_task_format,
        test_list=test_list,
        test_setup_code=test_setup_code,
        challenge_test_list=challenge_test_list,
        stdio_inputs=stdio_inputs,
        stdio_outputs=stdio_outputs,
        timeout_seconds=timeout_seconds,
    )
    return rewards


_DEFAULT_HUMANEVAL_TIMEOUT_SECONDS = 5.0
_HUMANEVAL_RESULT_PREFIX = "__HUMANEVAL_RESULT__"
_HUMANEVAL_RUNNER = f"""
import builtins
import contextlib
import io
import json
import sys

payload = json.loads(sys.stdin.read())

class _BlockedInput:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Interactive input is disabled during HumanEval reward evaluation.")

class _BlockedStdin(io.TextIOBase):
    def read(self, *args, **kwargs):
        raise RuntimeError("Interactive stdin is disabled during HumanEval reward evaluation.")

    def readline(self, *args, **kwargs):
        raise RuntimeError("Interactive stdin is disabled during HumanEval reward evaluation.")

    def readlines(self, *args, **kwargs):
        raise RuntimeError("Interactive stdin is disabled during HumanEval reward evaluation.")

class _DiscardIO(io.TextIOBase):
    def write(self, text):
        return len(text)

builtins.input = _BlockedInput()
sys.stdin = _BlockedStdin()

g = {{"__builtins__": builtins.__dict__, "__name__": "__main__"}}
try:
    with contextlib.redirect_stdout(_DiscardIO()), contextlib.redirect_stderr(_DiscardIO()):
        exec(compile(payload["full_code"], "<user>", "exec"), g, g)
        exec(payload["test"], g, g)
        cand = g[payload["entry_point"]]
        g["check"](cand)
    result = {{"ok": True, "error_type": None, "error": None}}
except Exception as exc:
    result = {{"ok": False, "error_type": type(exc).__name__, "error": str(exc)}}

print({json.dumps(_HUMANEVAL_RESULT_PREFIX)} + json.dumps(result))
"""


def _humaneval_run_once(full_code, test, entry_point, timeout_seconds):
    payload = json.dumps({
        "full_code": full_code,
        "test": test,
        "entry_point": entry_point,
    })
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _HUMANEVAL_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"HumanEval execution timed out after {timeout_seconds:.1f}s."
        ) from exc

    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip()
        if not message:
            message = f"HumanEval subprocess exited with code {proc.returncode}."
        return False

    result_line = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_HUMANEVAL_RESULT_PREFIX):
            result_line = line[len(_HUMANEVAL_RESULT_PREFIX):]
            break
    if result_line is None:
        return False
    data = json.loads(result_line)
    return bool(data.get("ok"))


def humaneval_best_reward_for_response(
    response,
    prompt_prefix,
    test,
    entry_point,
    *,
    timeout_seconds=_DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
):
    best = 0.0
    for code in _extract_python_code_candidates(response):
        full_code = (prompt_prefix or "") + code
        try:
            ok = _humaneval_run_once(full_code, test, entry_point, timeout_seconds)
        except (TimeoutError, RuntimeError, json.JSONDecodeError):
            ok = False
        if ok:
            return 1.0
        best = max(best, 0.0)
    return best


def humaneval_execution_reward_func(
    completions,
    prompt_prefix,
    test,
    entry_point,
    timeout_seconds=_DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    **kwargs,
):
    rewards = []
    for response in _extract_responses(completions):
        rewards.append(
            humaneval_best_reward_for_response(
                response,
                prompt_prefix,
                test,
                entry_point,
                timeout_seconds=timeout_seconds,
            )
        )
    return rewards


def mixed_math_accuracy_grpo_reward(
    prompts,
    completions,
    completion_ids=None,
    task_type=None,
    solution=None,
    **kwargs,
):
    if task_type is None or solution is None:
        raise ValueError("mixed_math_accuracy_grpo_reward requires task_type and solution columns.")
    texts = _extract_responses(completions)
    rewards = []
    for i, text in enumerate(texts):
        if task_type[i] != "math":
            rewards.append(0.0)
            continue
        model_answer = extract_math_final_answer(text)
        gold = solution[i]
        rewards.append(
            1.0 if _answers_match(model_answer, gold) else 0.0
        )
    return rewards


def mixed_code_execution_grpo_reward(
    prompts,
    completions,
    completion_ids=None,
    task_type=None,
    code_task_format=None,
    test_list=None,
    test_setup_code=None,
    challenge_test_list=None,
    stdio_inputs=None,
    stdio_outputs=None,
    **kwargs,
):
    if task_type is None or test_list is None:
        raise ValueError(
            "mixed_code_execution_grpo_reward requires task_type and test_list columns."
        )
    texts = _extract_responses(completions)
    n = len(texts)
    if test_setup_code is None:
        test_setup_code = [""] * n
    rewards = []
    for i, text in enumerate(texts):
        if task_type[i] != "code":
            rewards.append(0.0)
            continue
        task_format = (
            code_task_format[i]
            if code_task_format is not None and i < len(code_task_format)
            else "call"
        )
        ts = test_setup_code[i] if i < len(test_setup_code) else ""
        ctl_raw = None
        if challenge_test_list is not None and i < len(challenge_test_list):
            ctl_raw = challenge_test_list[i]
        ctl = ctl_raw if ctl_raw is not None else []
        row_stdio_inputs = (
            stdio_inputs[i]
            if stdio_inputs is not None and i < len(stdio_inputs)
            else []
        )
        row_stdio_outputs = (
            stdio_outputs[i]
            if stdio_outputs is not None and i < len(stdio_outputs)
            else []
        )
        r, _ = _taco_best_reward_for_response(
            text,
            code_task_format=task_format,
            test_list=test_list[i],
            test_setup_code=ts or "",
            challenge_test_list=ctl,
            stdio_inputs=row_stdio_inputs,
            stdio_outputs=row_stdio_outputs,
            timeout_seconds=_DEFAULT_MBPP_TIMEOUT_SECONDS,
        )
        rewards.append(r)
    return rewards


def mixed_format_guardrail_grpo_reward(
    prompts,
    completions,
    completion_ids=None,
    task_type=None,
    **kwargs,
):
    if task_type is None:
        raise ValueError("mixed_format_guardrail_grpo_reward requires task_type column.")
    texts = _extract_responses(completions)
    rewards = []
    for i, text in enumerate(texts):
        if task_type[i] != "math":
            rewards.append(0.0)
            continue
        rewards.append(
            format_guardrail_reward_func([[{"content": text}]])[0]
        )
    return rewards
