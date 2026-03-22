import re

def format_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL | re.IGNORECASE) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward_func(completions, solution, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response, true_answer in zip(responses, solution):
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if match:
            model_answer = match.group(1).strip()
            if model_answer == true_answer.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def soft_format_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for r in responses:
        score = 0.0
        if "<think>" in r.lower(): score += 0.25
        if "</think>" in r.lower(): score += 0.25
        if "<answer>" in r.lower(): score += 0.25
        if "</answer>" in r.lower(): score += 0.25
        rewards.append(score)
    return rewards


def soft_accuracy_reward_func(completions, solution, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response, true_answer in zip(responses, solution):
        true_answer = true_answer.strip()
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip() == true_answer:
            rewards.append(1.0)
            continue
        answer_nums = set(re.findall(r"[\d,]+\.?\d*", true_answer))
        response_nums = set(re.findall(r"[\d,]+\.?\d*", response))
        if answer_nums and answer_nums.issubset(response_nums):
            rewards.append(0.5)
        elif answer_nums & response_nums:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards