import torch


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cache(device=None):
    if device is None:
        device = detect_device()
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype == "cuda":
        torch.cuda.empty_cache()
    elif dtype == "mps":
        torch.mps.empty_cache()


def render_prompt(tokenizer, prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    raise TypeError("prompt must be a string or a chat-style list of messages")


def tokenize_prompt(tokenizer, prompt, device):
    prompt_text = render_prompt(tokenizer, prompt)
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    return prompt_text, input_ids, attention_mask


def extract_lora_gradients(peft_model):
    grads = []
    for _, param in peft_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.detach().float().view(-1).cpu().clone())
    peft_model.zero_grad()
    if not grads:
        raise RuntimeError("No LoRA gradients were found after backward().")
    return torch.cat(grads)


def get_reward_name(reward_fn):
    if hasattr(reward_fn, "__name__"):
        return reward_fn.__name__
    if hasattr(reward_fn, "func"):
        return getattr(reward_fn.func, "__name__", reward_fn.func.__class__.__name__)
    return reward_fn.__class__.__name__
