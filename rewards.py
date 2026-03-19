import re

def format_reward_func(completions, **kwargs):
    """
    Reward 1.0 if the model uses the strict <think>...</think><answer>...</answer> format.
    """
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL | re.IGNORECASE) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward_func(completions, solution, **kwargs):
    """
    Reward 1.0 if the extracted answer matches the GSM8K ground truth solution.
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response, true_answer in zip(responses, solution):
        # Extract the text inside <answer> tags
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if match:
            model_answer = match.group(1).strip()
            # Simple exact match (you can make this more robust later)
            if model_answer == true_answer.strip():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards