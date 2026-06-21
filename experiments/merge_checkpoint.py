"""Merge a LoRA checkpoint into its base and save a standalone model.

Lets `trl vllm-serve` hold a trained policy *outside* a training loop (e.g. standalone
influence scoring): the gen server needs the checkpoint weights and there's no trainer
to sync them, so we bake the adapter into the base and serve that. Gradients elsewhere
still come from the (separate) in-process PeftModel at the same checkpoint.

  python -m experiments.merge_checkpoint --model-id Qwen/Qwen3-1.7B-Base \
      --checkpoint outputs/math_if_v2/rlvr-output/checkpoint-10 --out $SCRATCH/merged/math10
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", required=True, help="path to a checkpoint-N dir (LoRA adapter)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.checkpoint).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.model_id).save_pretrained(args.out)
    print(f"[merge] {args.model_id} + {args.checkpoint} -> {args.out}")


if __name__ == "__main__":
    main()
