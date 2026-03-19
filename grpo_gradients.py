import torch
import torch.nn.functional as F


def _clear_cache(device):
    dtype = device.type if isinstance(device, torch.device) else device
    if dtype == "cuda":
        torch.cuda.empty_cache()
    elif dtype == "mps":
        torch.mps.empty_cache()


def _extract_lora_gradients(peft_model):
    grads = []
    for _, param in peft_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.detach().view(-1).clone())
    peft_model.zero_grad()
    return torch.cat(grads)


def compute_sft_gradient(peft_model, tokenizer, prompt, target, device):
    """
    Computes the gradient of the LoRA parameters under standard SFT cross-entropy loss.
    Used for computing g_test on Code evaluation prompts.

    Labels are masked so only the target tokens contribute to the loss
    (prompt positions are set to -100, which PyTorch CE ignores).
    """
    peft_model.eval()
    peft_model.zero_grad()

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    target_ids = tokenizer(target + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).input_ids

    input_ids = torch.cat([prompt_ids, target_ids], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Mask prompt positions in labels so loss is only computed on target tokens
    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    grad_vector = _extract_lora_gradients(peft_model)
    _clear_cache(device)
    return grad_vector


def compute_rlvr_gradient(
    peft_model, tokenizer, prompt, reward_funcs,
    G=4, device="cpu", enable_vllm=False,
    max_new_tokens=256, temperature=0.7, top_p=0.9,
):
    """
    Simulates a single GRPO step for one prompt and returns the LoRA gradient.

    Steps:
      1. Generate G completions from the prompt (sampling).
      2. Score each completion using the provided reward_funcs.
      3. Compute per-completion advantages (normalized rewards).
      4. Compute simplified policy loss: L_RL = -1/G * sum(A_i * log_pi(y_i | x)).
      5. Backprop through the policy loss and extract LoRA gradients.

    reward_funcs: list of callables matching the TRL-style signature used in rewards.py.
      Each function receives (completions, **kwargs) where completions is a list of
      [{"role": "assistant", "content": text}] entries (one per generation).
      To pass extra kwargs like `solution`, wrap your function with functools.partial
      before passing it here.
    """
    peft_model.eval()
    peft_model.zero_grad()

    # ── Step 0: Tokenize prompt ──────────────────────────────────────────────
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # ── Step 1: Generate G completions ───────────────────────────────────────
    if enable_vllm:
        # PLACEHOLDER: Route to vLLM engine for fast batched generation on GPU clusters.
        # When implemented, this should call a vLLM LLM/SamplingParams interface,
        # decode outputs, and populate `response_texts` identically to the HF branch below.
        raise NotImplementedError(
            "vLLM generation backend is not yet implemented. "
            "Set enable_vllm=False to use standard HF generate()."
        )

    with torch.no_grad():
        generated = peft_model.generate(
            input_ids=prompt_ids.expand(G, -1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt prefix)
    response_texts = []
    for i in range(G):
        gen_ids = generated[i, prompt_len:]
        response_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

    # ── Step 2: Score completions with reward functions ──────────────────────
    # Wrap each response into TRL-style format: list of [{"role":"assistant","content":...}]
    completions_trl = [[{"role": "assistant", "content": t}] for t in response_texts]

    total_rewards = torch.zeros(G, device=device)
    for reward_fn in reward_funcs:
        scores = reward_fn(completions_trl)
        total_rewards += torch.tensor(scores, device=device, dtype=torch.float32)

    # ── Step 3: Compute advantages ───────────────────────────────────────────
    advantages = (total_rewards - total_rewards.mean()) / (total_rewards.std() + 1e-8)

    # ── Step 4: Compute log pi(y_i | x) for each completion ─────────────────
    # We need gradients here, so no torch.no_grad()
    log_probs_per_response = []
    for i in range(G):
        resp_ids = tokenizer(
            response_texts[i] + tokenizer.eos_token,
            return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(device)

        full_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        attention_mask = torch.ones_like(full_ids)

        outputs = peft_model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, vocab)

        # Log-probs of the response tokens only (positions prompt_len .. end)
        # Shift logits/labels by 1 for next-token prediction alignment
        shift_logits = logits[:, prompt_len - 1:-1, :]  # predictions for response tokens
        shift_labels = full_ids[:, prompt_len:]          # actual response token ids

        log_p = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_p.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_pi = token_log_probs.sum()  # log pi(y_i | x)
        log_probs_per_response.append(log_pi)

    log_probs = torch.stack(log_probs_per_response)

    # ── Step 5: Policy loss and backward ─────────────────────────────────────
    # L_RL = -1/G * sum(A_i * log_pi(y_i | x))
    policy_loss = -(1.0 / G) * (advantages.detach() * log_probs).sum()
    policy_loss.backward()

    grad_vector = _extract_lora_gradients(peft_model)

    # Aggressive cleanup
    del generated, log_probs, log_probs_per_response, policy_loss
    _clear_cache(device)

    return grad_vector
