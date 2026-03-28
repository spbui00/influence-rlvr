R1_MATH_INSTRUCTION = (
    "Please reason step by step inside <think></think> tags. "
    "After the thinking block, give the final answer within \\boxed{}."
)


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
