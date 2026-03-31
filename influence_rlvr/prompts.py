R1_MATH_INSTRUCTION = (
    "Please reason step by step inside <think></think> tags. "
    "After the thinking block, give the final answer within \\boxed{}."
)


def append_suffix_to_last_user_message(
    messages: list[dict[str, str]], suffix: str
) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages must be non-empty")
    out: list[dict[str, str]] = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            c = out[i]["content"]
            out[i] = {"role": "user", "content": f"{c}\n\n{suffix}"}
            return out
    raise ValueError("No user message in messages")


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
