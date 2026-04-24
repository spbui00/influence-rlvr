"""
Inspect benchmark prompts and stream model responses in the terminal. Useful for sanity-checking
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Thread

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.trainer_utils import get_last_checkpoint

from influence_rlvr import detect_device

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Inspect benchmark prompts and stream model responses in the terminal. "
            "Defaults are read from training run_config.json when --rlvr-output is provided."
        )
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("outputs/if_benchmark_small/train.jsonl"),
        help="Benchmark train JSONL to inspect.",
    )
    p.add_argument(
        "--rlvr-output",
        type=Path,
        default=None,
        help=(
            "Optional training output directory containing run_config.json and checkpoint-* "
            "folders. Used to auto-populate model/generation defaults."
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Optional checkpoint path or 'latest'. If omitted, uses the base model only. "
            "When --rlvr-output is set, 'latest' resolves under that directory."
        ),
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Base model id. Defaults to run_config.json model_id if available.",
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Run one sample by integer row index and exit.",
    )
    p.add_argument(
        "--sample-id",
        default=None,
        help="Run one sample by row id and exit.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generation length. Defaults to training max_completion_length when available.",
    )
    p.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature when --do-sample.",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p when --do-sample.",
    )
    p.add_argument(
        "--list-limit",
        type=int,
        default=20,
        help="How many rows to show in the initial interactive list.",
    )
    return p.parse_args()


def _model_dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float32
    return torch.float32


def _load_run_config(rlvr_output: Path | None) -> dict:
    if rlvr_output is None:
        return {}
    config_path = rlvr_output / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _resolve_checkpoint(checkpoint_dir: str | None, rlvr_output: Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    if checkpoint_dir == "latest":
        if rlvr_output is None:
            raise SystemExit("--checkpoint-dir=latest requires --rlvr-output.")
        latest = get_last_checkpoint(str(rlvr_output))
        if latest is None:
            raise SystemExit(f"No checkpoint-* directories found under {rlvr_output}")
        return Path(latest).resolve()
    path = Path(checkpoint_dir).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise SystemExit(f"Checkpoint directory not found: {path}")
    return path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _choose_row(rows: list[dict], sample_index: int | None, sample_id: str | None) -> dict | None:
    if sample_index is not None and sample_id is not None:
        raise SystemExit("Use only one of --sample-index or --sample-id.")
    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(rows):
            raise SystemExit(f"sample-index out of range: {sample_index}")
        return rows[sample_index]
    if sample_id is not None:
        for row in rows:
            if row.get("id") == sample_id:
                return row
        raise SystemExit(f"sample-id not found: {sample_id}")
    return None


def _load_model_and_tokenizer(model_id: str, checkpoint: Path | None, device: torch.device):
    dtype = _model_dtype_for_device(device)
    tokenizer_source = str(checkpoint) if checkpoint is not None else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
    ).to(device)
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, str(checkpoint)).to(device)
    model.eval()
    return model, tokenizer


def _render_row_summary(idx: int, row: dict) -> str:
    return (
        f"[{idx:02d}] {row.get('id')} | "
        f"task={row.get('task_type')} | "
        f"variant={row.get('variant_type')} | "
        f"expected={row.get('expected_influence')}"
    )


def _print_row_details(idx: int, row: dict):
    print("=" * 100)
    print(_render_row_summary(idx, row))
    print(f"base_id={row.get('base_id')} related_test_id={row.get('related_test_id')}")
    if "solution" in row:
        print(f"solution={row.get('solution')!r}")
    if row.get("task_type") == "code":
        print(f"code_task_format={row.get('code_task_format')!r}")
        test_list = row.get("test_list") or []
        stdio_inputs = row.get("stdio_inputs") or []
        print(f"n_test_list={len(test_list)} n_stdio_cases={len(stdio_inputs)}")
    print("\n[PROMPT]\n")
    print(row["prompt"][-1]["content"])
    print()


@torch.inference_mode()
def _stream_response(
    model,
    tokenizer,
    messages: list[dict],
    *,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
    attention_mask = None if isinstance(enc, torch.Tensor) else enc.get("attention_mask")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)

    prompt_tokens = int(input_ids.shape[1])
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    print(
        f"[generation] prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens} "
        f"do_sample={do_sample} temperature={temperature} top_p={top_p}"
    )
    print("\n[RESPONSE]\n")
    worker = Thread(target=model.generate, kwargs=generate_kwargs)
    worker.start()
    pieces: list[str] = []
    for chunk in streamer:
        print(chunk, end="", flush=True)
        pieces.append(chunk)
    worker.join()
    print("\n")
    return "".join(pieces), prompt_tokens


def _interactive_loop(rows: list[dict], model, tokenizer, device: torch.device, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float):
    print("\nRows:")
    for idx, row in enumerate(rows[: min(len(rows), 20)]):
        print(_render_row_summary(idx, row))
    print(
        "\nCommands: list [n], show <index>, run <index>, id <row_id>, sample on, sample off, "
        "max <tokens>, quit"
    )

    while True:
        try:
            raw = input("inspect> ").strip()
        except EOFError:
            print()
            return
        if not raw:
            continue
        if raw in {"q", "quit", "exit"}:
            return
        if raw.startswith("list"):
            parts = raw.split()
            limit = int(parts[1]) if len(parts) > 1 else 20
            for idx, row in enumerate(rows[: min(len(rows), limit)]):
                print(_render_row_summary(idx, row))
            continue
        if raw.startswith("show "):
            idx = int(raw.split(maxsplit=1)[1])
            _print_row_details(idx, rows[idx])
            continue
        if raw.startswith("run "):
            idx = int(raw.split(maxsplit=1)[1])
            row = rows[idx]
            _print_row_details(idx, row)
            _stream_response(
                model,
                tokenizer,
                row["prompt"],
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            continue
        if raw.startswith("id "):
            target = raw.split(maxsplit=1)[1]
            for idx, row in enumerate(rows):
                if row.get("id") == target:
                    _print_row_details(idx, row)
                    _stream_response(
                        model,
                        tokenizer,
                        row["prompt"],
                        device=device,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    break
            else:
                print(f"row id not found: {target}")
            continue
        if raw == "sample on":
            do_sample = True
            print(f"sampling enabled | temperature={temperature} top_p={top_p}")
            continue
        if raw == "sample off":
            do_sample = False
            print("sampling disabled (greedy mode)")
            continue
        if raw.startswith("max "):
            max_new_tokens = int(raw.split(maxsplit=1)[1])
            print(f"max_new_tokens={max_new_tokens}")
            continue
        print("unknown command")


def main():
    args = parse_args()
    rlvr_output = args.rlvr_output.resolve() if args.rlvr_output is not None else None
    run_config = _load_run_config(rlvr_output)

    model_id = args.model_id or run_config.get("model_id") or DEFAULT_MODEL_ID
    max_new_tokens = args.max_new_tokens or run_config.get("max_completion_length") or 1024
    checkpoint = _resolve_checkpoint(args.checkpoint_dir, rlvr_output)

    rows = _load_rows(args.train_jsonl.resolve())
    row = _choose_row(rows, args.sample_index, args.sample_id)

    device = detect_device()
    print(f"Device: {device}")
    print(f"Model: {model_id}")
    print(f"Checkpoint: {checkpoint if checkpoint is not None else 'base model only'}")
    print(f"Train JSONL: {args.train_jsonl.resolve()}")
    print(f"Default max_new_tokens: {max_new_tokens}")

    model, tokenizer = _load_model_and_tokenizer(model_id, checkpoint, device)

    if row is not None:
        idx = rows.index(row)
        _print_row_details(idx, row)
        _stream_response(
            model,
            tokenizer,
            row["prompt"],
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        return

    _interactive_loop(
        rows,
        model,
        tokenizer,
        device,
        max_new_tokens=max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
