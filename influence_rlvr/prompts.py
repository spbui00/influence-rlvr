R1_MATH_INSTRUCTION = (
    "Please reason step by step inside <think></think> tags. "
    "After the thinking block, give the final answer within \\boxed{}."
)


def append_suffix_to_final_user_message(messages: list[dict], suffix: str) -> list[dict]:
    if not messages:
        raise ValueError("messages must be non-empty")
    out: list[dict] = [dict(m) for m in messages]
    last_user = None
    for m in reversed(out):
        if m.get("role") == "user":
            last_user = m
            break
    if last_user is None:
        raise ValueError("No user message in messages")
    base = last_user.get("content", "")
    if not isinstance(base, str):
        raise TypeError("append_suffix_to_final_user_message expects string user content")
    last_user["content"] = f"{base}\n\n{suffix}" if base else suffix
    return out


def build_r1_math_prompt(question):
    return [{
        "role": "user",
        "content": f"{question}\n\n{R1_MATH_INSTRUCTION}",
    }]


def build_code_prompt(text):
    return [{
        "role": "user",
        "content": text,
    }]


def extract_gsm8k_target(answer):
    return answer.split("#### ")[-1].strip()
