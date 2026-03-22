import torch
import torch.nn.functional as F

from .utils import clear_cache, tokenize_prompt, extract_lora_gradients, get_reward_name


def compute_sft_gradient(peft_model, tokenizer, prompt, target, device):
    peft_model.eval()
    peft_model.zero_grad()

    _, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)
    target_ids = tokenizer(
        target + tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    target_attention_mask = torch.ones_like(target_ids)
    attention_mask = torch.cat([prompt_attention_mask, target_attention_mask], dim=1)

    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    grad_vector = extract_lora_gradients(peft_model)
    clear_cache(device)
    return grad_vector


def compute_rlvr_gradient(
    peft_model, tokenizer, prompt, reward_funcs,
    G=4, device="cpu", enable_vllm=False,
    max_new_tokens=256, temperature=0.7, top_p=0.9,
    return_debug=False,
):
    peft_model.eval()
    peft_model.zero_grad()

    prompt_text, prompt_ids, prompt_attention_mask = tokenize_prompt(tokenizer, prompt, device)
    prompt_len = prompt_ids.shape[1]

    if enable_vllm:
        raise NotImplementedError(
            "vLLM generation backend is not yet implemented. "
            "Set enable_vllm=False to use standard HF generate()."
        )

    with torch.no_grad():
        generated = peft_model.generate(
            input_ids=prompt_ids.repeat(G, 1),
            attention_mask=prompt_attention_mask.repeat(G, 1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_texts = []
    for i in range(G):
        gen_ids = generated[i, prompt_len:]
        response_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    completions_trl = [[{"role": "assistant", "content": t}] for t in response_texts]

    total_rewards = torch.zeros(G, device=device)
    reward_breakdown = {}
    for reward_fn in reward_funcs:
        scores = reward_fn(completions_trl)
        reward_breakdown[get_reward_name(reward_fn)] = [float(s) for s in scores]
        total_rewards += torch.tensor(scores, device=device, dtype=torch.float32)

    advantages = (total_rewards - total_rewards.mean()) / (total_rewards.std() + 1e-8)

    log_probs_per_response = []
    for i in range(G):
        resp_ids = tokenizer(
            response_texts[i] + tokenizer.eos_token,
            return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(device)

        full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        resp_attention_mask = torch.ones_like(resp_ids)
        attention_mask = torch.cat([prompt_attention_mask, resp_attention_mask], dim=1)

        outputs = peft_model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = full_ids[:, prompt_len:]

        log_p = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_p.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_pi = token_log_probs.sum()
        log_probs_per_response.append(log_pi)

    log_probs = torch.stack(log_probs_per_response)

    policy_loss = -(1.0 / G) * (advantages.detach() * log_probs).sum()
    policy_loss.backward()

    grad_vector = extract_lora_gradients(peft_model)

    debug_info = {
        "prompt_text": prompt_text,
        "responses": response_texts,
        "reward_breakdown": reward_breakdown,
        "total_rewards": total_rewards.detach().cpu().tolist(),
        "advantages": advantages.detach().cpu().tolist(),
        "log_probs": log_probs.detach().float().cpu().tolist(),
        "policy_loss": float(policy_loss.detach().float().cpu()),
    }

    del generated, log_probs, log_probs_per_response, policy_loss
    clear_cache(device)

    if return_debug:
        return grad_vector, debug_info
    return grad_vector
