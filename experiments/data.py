"""Dataset loading for the scaled experiment.

Train pool + IF-target set both come from WebInstruct-verified (General-Reasoner),
filtered to the configured domains (Math / CS / Finance). Eval benchmarks are a
small registry; the heavy agentic ones (SWE-bench, LiveCodeBench) are left as
documented stubs because they need their own execution harnesses.

Row schema produced for GRPO (consumed by TRL + the verifier reward):
  prompt        — chat messages (list[dict]) ready for `apply_chat_template`
  question      — raw question string (verifier needs it)
  solution      — reference answer string (verifier ground truth)
  answer_type   — WebInstruct answer_type (Float/Integer/Expression/...)
  category      — WebInstruct category
  train_index   — stable index into the *filtered* pool (influence bookkeeping)
"""
from __future__ import annotations

from typing import Iterable

from datasets import Dataset, load_dataset

from .config import DOMAIN_TO_CATEGORIES, ExperimentConfig


# General reasoning instruction (domain-agnostic): think, then box the answer so
# both the verifier and a fallback parser can extract a final answer.
GENERAL_REASONING_INSTRUCTION = (
    "Please reason step by step inside <think></think> tags. "
    "After the thinking block, state the final answer on its own line "
    "within \\boxed{}."
)


def build_reasoning_prompt(question: str) -> list[dict]:
    return [{"role": "user", "content": f"{question}\n\n{GENERAL_REASONING_INSTRUCTION}"}]


def _categories_for_domains(domains: Iterable[str]) -> set[str]:
    cats: set[str] = set()
    for d in domains:
        cats.update(DOMAIN_TO_CATEGORIES[d])
    return cats


def _format_webinstruct_row(example: dict, idx: int) -> dict:
    return {
        "prompt": build_reasoning_prompt(example["question"]),
        "question": example["question"],
        "solution": str(example.get("answer", "") or ""),
        "answer_type": example.get("answer_type", ""),
        "category": example.get("category", ""),
        "train_index": idx,
    }


def _load_webinstruct_split(cfg: ExperimentConfig, split: str) -> Dataset:
    raw = load_dataset(cfg.train_dataset, split=split)
    keep = _categories_for_domains(cfg.domains)
    raw = raw.filter(lambda ex: ex.get("category") in keep)
    # Drop rows without a usable reference answer.
    raw = raw.filter(lambda ex: bool(str(ex.get("answer", "") or "").strip()))
    return raw


def load_train_pool(cfg: ExperimentConfig) -> Dataset:
    """Filtered + capped training pool with a stable `train_index`."""
    raw = _load_webinstruct_split(cfg, "train")
    if cfg.n_train_pool and cfg.n_train_pool > 0 and len(raw) > cfg.n_train_pool:
        raw = raw.shuffle(seed=cfg.seed).select(range(cfg.n_train_pool))
    # Re-index after the (optional) subsample so train_index is contiguous.
    ds = raw.map(
        _format_webinstruct_row,
        with_indices=True,
        remove_columns=raw.column_names,
    )
    return ds


def load_if_target_set(cfg: ExperimentConfig) -> Dataset:
    """Held-out target set the influence is measured against (test split)."""
    raw = _load_webinstruct_split(cfg, "test")
    if cfg.n_if_target and cfg.n_if_target > 0 and len(raw) > cfg.n_if_target:
        raw = raw.shuffle(seed=cfg.seed + 1).select(range(cfg.n_if_target))
    ds = raw.map(
        _format_webinstruct_row,
        with_indices=True,
        remove_columns=raw.column_names,
    )
    return ds


# ── Eval benchmarks ─────────────────────────────────────────────────────────
# Each loader returns a list of {question, solution, answer_type, source, category}.
# Verifier-based scoring works for all of them (free-form equivalence), so we do
# not need per-benchmark exact-match rules here.

def _eval_row(question: str, solution: str, *, source: str,
              answer_type: str = "", category: str = "") -> dict:
    return {
        "question": question,
        "solution": str(solution),
        "answer_type": answer_type,
        "category": category,
        "source": source,
    }


def load_webinstruct_test(cfg: ExperimentConfig, limit: int) -> list[dict]:
    raw = _load_webinstruct_split(cfg, "test")
    if limit and len(raw) > limit:
        raw = raw.shuffle(seed=cfg.seed + 7).select(range(limit))
    return [
        _eval_row(ex["question"], ex.get("answer", ""), source="webinstruct_test",
                  answer_type=ex.get("answer_type", ""), category=ex.get("category", ""))
        for ex in raw
    ]


def load_gsm8k(cfg: ExperimentConfig, limit: int) -> list[dict]:
    raw = load_dataset("openai/gsm8k", "main", split="test")
    if limit and len(raw) > limit:
        raw = raw.select(range(limit))
    out = []
    for ex in raw:
        gold = ex["answer"].split("#### ")[-1].strip()
        out.append(_eval_row(ex["question"], gold, source="gsm8k",
                             answer_type="Integer", category="Mathematics"))
    return out


def load_math500(cfg: ExperimentConfig, limit: int) -> list[dict]:
    raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit and len(raw) > limit:
        raw = raw.select(range(limit))
    return [
        _eval_row(ex["problem"], ex["answer"], source="math500",
                  answer_type="Expression", category="Mathematics")
        for ex in raw
    ]


# Heavier benchmarks the user listed. These need their own harnesses; wire them
# in once the core loop is validated. Documented here so the registry is honest.
_UNIMPLEMENTED = {
    "theoremqa": "TIGER-Lab/TheoremQA — free-form; verifier-scorable, add loader.",
    "olympiadbench": "Hothan/OlympiadBench — multimodal subsets need filtering.",
    "aime25": "AIME 2025 — tiny (30 q); add a static loader.",
    "finqa": "ibm-research/finqa or dreamerdeo/finqa — needs table context in prompt.",
    "livecodebench": "livecodebench/* — needs the LCB execution harness, not verifier.",
    "swebench": "princeton-nlp/SWE-bench — agentic; out of scope for verifier scoring.",
}

EVAL_LOADERS = {
    "webinstruct_test": load_webinstruct_test,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
}


def load_eval_benchmark(name: str, cfg: ExperimentConfig, limit: int) -> list[dict]:
    if name not in EVAL_LOADERS:
        hint = _UNIMPLEMENTED.get(name, "no loader registered")
        raise NotImplementedError(f"Eval benchmark {name!r} not implemented yet ({hint}).")
    return EVAL_LOADERS[name](cfg, limit)
