"""Headroom probe — pass@1 vs pass@k for the BASE model on the held-out target.

The recurring question in this project: is there anything for RL to extract, or is
the target unmovable at this model size? Greedy training chases pass@1; pass@k (any
of k samples correct) is the latent ceiling RL can sharpen toward.

  pass@k >> pass@1  → the base CAN reach the answers under sampling; held-out
                      flatness is a scale/optimization problem, not a ceiling.
  pass@k ~= pass@1  → the base genuinely can't do these; change target or model.

Single process: spins up ONE standalone vLLM engine for the base model (no DDP, no
trainer → none of the weight-sync deadlocks). Needs a verifier reachable per the
run's verifier-* config (e.g. the :8100 server train.slurm already launches).

  python -m experiments.headroom --k 8 --headroom-temperature 1.0 --limit 256 \
      --run-name fin_headroom --model Qwen/Qwen3-1.7B-Base \
      --test-from-train --test-domains finance --domains math,physics,finance \
      --pool-size 24000 --n-if-target 256 \
      --verifier-backend server --verifier-server-port 8100
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from influence_rlvr import detect_device
from influence_rlvr.generation import generate_rollout_batch
from influence_rlvr.modes import GenerationBackend, VLLMConfig

from .config import ExperimentConfig
from .data import build_reasoning_prompt, load_webinstruct_test
from .evaluate import _load_tokenizer
from .verifier import _student_answer, get_verifier_from_config


def _gen_k_samples(tokenizer, questions, cfg, device, k, temperature):
    """k sampled completions per prompt via the standalone vLLM engine (base model).

    Returns a list aligned to `questions`, each a list of k raw completion strings.
    Layout is prompt-major: rollout.texts[i*k:(i+1)*k] are prompt i's k samples
    (same convention the GRPO grad path relies on).
    """
    chats = [build_reasoning_prompt(q) for q in questions]
    prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
               for c in chats]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=cfg.max_prompt_length).to(device)
    vcfg = VLLMConfig(gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                      max_model_len=cfg.vllm_max_model_len, max_lora_rank=cfg.lora_r)
    rollout = generate_rollout_batch(
        None, tokenizer, enc["input_ids"], enc["attention_mask"],
        backend=GenerationBackend.VLLM, num_samples=k,
        max_new_tokens=cfg.eval_max_new_tokens, do_sample=True,
        temperature=temperature, top_p=cfg.eval_top_p,
        vllm_config=vcfg, adapter_path=None, model_id=cfg.model_id,
    )
    if rollout.num_prompts * rollout.num_samples != len(rollout.texts):
        raise RuntimeError("Rollout layout does not match num_prompts * num_samples.")
    return [list(rollout.texts[i * k:(i + 1) * k]) for i in range(len(questions))]


def _pass_at_m(correct_rows: list[list[bool]], m: int) -> float:
    """Fraction of prompts with >=1 correct among the first m samples (prefix estimate)."""
    return sum(1.0 for row in correct_rows if any(row[:m])) / len(correct_rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--k", type=int, default=8, help="samples per prompt")
    ap.add_argument("--headroom-temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=256, help="held-out prompts to probe")
    probe, rest = ap.parse_known_args(argv)

    cfg = ExperimentConfig.from_cli(rest)
    device = detect_device()
    tokenizer = _load_tokenizer(cfg)  # tokenizer only; vLLM loads the base weights itself

    examples = load_webinstruct_test(cfg, probe.limit)
    cats = Counter(e.get("category", "") for e in examples)
    print(f"[headroom] {len(examples)} held-out prompts {dict(cats)} | k={probe.k} "
          f"temp={probe.headroom_temperature} model={cfg.model_id}")

    questions = [e["question"] for e in examples]
    golds = [e["solution"] for e in examples]
    samples = _gen_k_samples(tokenizer, questions, cfg, device, probe.k,
                             probe.headroom_temperature)

    # Flatten every (prompt, sample) for one batched verifier pass, then regroup.
    flat_q, flat_g, flat_s, owner = [], [], [], []
    for i, row in enumerate(samples):
        for raw in row:
            flat_q.append(questions[i]); flat_g.append(golds[i])
            flat_s.append(_student_answer(raw)); owner.append(i)
    rewards = get_verifier_from_config(cfg).verify_batch(flat_q, flat_g, flat_s)

    correct_rows: list[list[bool]] = [[] for _ in examples]
    for o, r in zip(owner, rewards):
        correct_rows[o].append(bool(r > 0))
    cat_of = [e.get("category", "") for e in examples]

    def report(idxs: list[int], label: str) -> dict:
        rows = [correct_rows[i] for i in idxs]
        n_samp = sum(len(r) for r in rows)
        pass1 = sum(sum(r) for r in rows) / n_samp if n_samp else 0.0
        out = {"n_prompts": len(rows), "pass@1": round(pass1, 4)}
        for m in sorted({1, 2, 4, probe.k}):
            if m <= probe.k:
                out[f"pass@{m}"] = round(_pass_at_m(rows, m), 4)
        print(f"  [{label}] n={len(rows)}  " +
              "  ".join(f"{k_}={v}" for k_, v in out.items() if k_.startswith("pass")))
        return out

    print("[headroom] results:")
    summary = {"overall": report(list(range(len(examples))), "overall")}
    for c in sorted(cats):
        idxs = [i for i, cc in enumerate(cat_of) if cc == c]
        if idxs:
            summary[c] = report(idxs, c)

    out_path = cfg.run_dir / "headroom.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"k": probe.k, "temperature": probe.headroom_temperature,
         "model": cfg.model_id, "limit": probe.limit, "summary": summary}, indent=2))
    p1 = summary["overall"]["pass@1"]
    pk = summary["overall"][f"pass@{probe.k}"]
    print(f"[headroom] overall pass@1={p1} → pass@{probe.k}={pk} "
          f"(gap {round(pk - p1, 4)}) | saved {out_path}")
    print("[headroom] read: big gap → headroom exists (scale it up); "
          "gap ~0 → unmovable at this model size (switch target/model).")


if __name__ == "__main__":
    main()
