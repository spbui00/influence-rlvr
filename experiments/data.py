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

import random
from typing import Iterable

from datasets import Dataset, concatenate_datasets, load_dataset

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


def _load_train_domain(cfg: ExperimentConfig, domain: str) -> Dataset:
    """Train split filtered to ONE domain's categories (+ nonempty answer)."""
    cats = set(DOMAIN_TO_CATEGORIES[domain])
    raw = load_dataset(cfg.train_dataset, split="train")
    raw = raw.filter(lambda ex: ex.get("category") in cats)
    raw = raw.filter(lambda ex: bool(str(ex.get("answer", "") or "").strip()))
    return raw


def load_train_pool(cfg: ExperimentConfig) -> Dataset:
    """Training pool with a stable `train_index`.

    With `balance_domains` (default), draw an EQUAL share (n_train_pool/#domains)
    from each domain so the pool is balanced — important when the IF target is one
    domain (e.g. CS) and we want to see whether influence selects helpful examples
    from the OTHER domains. Otherwise: a single random sample (Math-heavy, the
    natural WebInstruct mix).
    """
    if cfg.balance_domains and cfg.n_train_pool and cfg.n_train_pool > 0:
        per = cfg.n_train_pool // len(cfg.domains)
        parts = []
        for d in cfg.domains:
            raw_d = _load_train_domain(cfg, d).shuffle(seed=cfg.seed)
            k = min(per, len(raw_d))
            if k < per:
                print(f"  [data] domain {d!r}: only {len(raw_d)} rows, using {k} (< {per})")
            parts.append(raw_d.select(range(k)))
        raw = concatenate_datasets(parts).shuffle(seed=cfg.seed + 5)
        print(f"  [data] balanced pool: {[(d, min(per, len(_load_train_domain(cfg, d)))) for d in cfg.domains]}")
    else:
        raw = _load_webinstruct_split(cfg, "train")
        if cfg.n_train_pool and cfg.n_train_pool > 0 and len(raw) > cfg.n_train_pool:
            raw = raw.shuffle(seed=cfg.seed).select(range(cfg.n_train_pool))
    # Re-index after sampling so train_index is contiguous.
    return raw.map(
        _format_webinstruct_row,
        with_indices=True,
        remove_columns=raw.column_names,
    )


def _webinstruct_test_domains(cfg: ExperimentConfig) -> tuple[str, ...]:
    """Domains for the test partition: `webinstruct_test_domains` or all `domains`."""
    return tuple(cfg.webinstruct_test_domains) or tuple(cfg.domains)


def _webinstruct_test_partition(cfg: ExperimentConfig) -> tuple[Dataset, Dataset]:
    """Deterministic, DISJOINT split of the test slice into (if_target, eval).

    One fixed shuffle of the (domain-filtered) test split; the first
    `n_if_target` rows are the influence target, the remainder are the
    in-distribution eval set. So the eval can never overlap the IF target.
    Both loaders call this, so they always agree on the partition.
    """
    cats = _categories_for_domains(_webinstruct_test_domains(cfg))
    raw = load_dataset(cfg.train_dataset, split="test")
    raw = raw.filter(lambda ex: ex.get("category") in cats)
    raw = raw.filter(lambda ex: bool(str(ex.get("answer", "") or "").strip()))
    raw = raw.shuffle(seed=cfg.seed)
    n = len(raw)
    n_if = cfg.n_if_target if cfg.n_if_target and cfg.n_if_target > 0 else n
    if n_if >= n:                       # keep the eval partition non-empty
        n_if = max(1, n // 2)
        print(f"  [data] only {n} test rows for domains "
              f"{_webinstruct_test_domains(cfg)}; splitting {n_if} IF target / "
              f"{n - n_if} eval.")
    return raw.select(range(n_if)), raw.select(range(n_if, n))


def load_if_target_set(cfg: ExperimentConfig) -> Dataset:
    """Held-out target set the influence is measured against.

    Default: the IF half of the disjoint WebInstruct test partition. With
    `if_target_source="mmlu_cs"`: the target half of the disjoint MMLU-CS
    partition (same distribution as the mmlu_cs eval — no proxy mismatch).
    With `if_target_full_test`: the ENTIRE filtered WebInstruct test slice.
    """
    if cfg.if_target_source == "mmlu_cs":
        target_rows, _ = _mmlu_cs_partition(cfg)
        raw = Dataset.from_list(target_rows)
        print(f"  [data] IF target: {len(raw)} MMLU-CS prompts "
              f"(disjoint from the mmlu_cs eval half)")
        return raw.map(_format_webinstruct_row, with_indices=True,
                       remove_columns=raw.column_names)

    if cfg.if_target_full_test:
        cats = _categories_for_domains(_webinstruct_test_domains(cfg))
        raw = load_dataset(cfg.train_dataset, split="test")
        raw = raw.filter(lambda ex: ex.get("category") in cats)
        raw = raw.filter(lambda ex: bool(str(ex.get("answer", "") or "").strip()))
        raw = raw.shuffle(seed=cfg.seed)
        if cfg.n_if_target and cfg.n_if_target > 0 and len(raw) > cfg.n_if_target:
            raw = raw.select(range(cfg.n_if_target))
        if "webinstruct_test" in cfg.eval_benchmarks:
            print("  [data] WARNING: if_target_full_test=True AND webinstruct_test "
                  "in eval_benchmarks → eval LEAKS into the IF target. Use an "
                  "external eval dataset instead.")
        if_raw = raw
    else:
        if_raw, _ = _webinstruct_test_partition(cfg)
    print(f"  [data] IF target: {len(if_raw)} prompts "
          f"(domains={_webinstruct_test_domains(cfg)}, full_test={cfg.if_target_full_test})")
    return if_raw.map(
        _format_webinstruct_row,
        with_indices=True,
        remove_columns=if_raw.column_names,
    )


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
    """In-distribution eval — the eval half of the test partition (disjoint from
    the IF target). Respects `webinstruct_test_domains` (e.g. CS-only)."""
    _, eval_raw = _webinstruct_test_partition(cfg)
    if limit and len(eval_raw) > limit:
        eval_raw = eval_raw.select(range(limit))
    return [
        _eval_row(ex["question"], ex.get("answer", ""), source="webinstruct_test",
                  answer_type=ex.get("answer_type", ""), category=ex.get("category", ""))
        for ex in eval_raw
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


# ── MMLU-CS (external, clean, labeled CS) — used for BOTH the IF target and eval ─
# (disjoint splits). cais/mmlu has clean per-subject configs.
MMLU_CS_CONFIGS = (
    "college_computer_science",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
)


def _mmlu_cs_rows(cfg: ExperimentConfig) -> list[dict]:
    """Combined MMLU-CS `test` questions as WebInstruct-style rows.

    Each multiple-choice item becomes a prompt with lettered options + an
    instruction to answer with the letter; the gold is that letter.
    """
    rows: list[dict] = []
    for sub in MMLU_CS_CONFIGS:
        ds = load_dataset("cais/mmlu", sub, split="test")
        for ex in ds:
            choices = list(ex["choices"])
            idx = int(ex["answer"])
            letters = "ABCDEFGH"[: len(choices)]
            opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
            q = (f"{ex['question']}\n\nOptions:\n{opts}\n\n"
                 "Answer with the letter of the single correct option.")
            rows.append({
                "question": q,
                "answer": letters[idx],
                "answer_type": "Multiple Choice",
                "category": "Computer Science",
                "subject": sub,
            })
    return rows


def _mmlu_cs_partition(cfg: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    """Deterministic disjoint (IF target, eval) split of MMLU-CS test."""
    rows = _mmlu_cs_rows(cfg)
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    n = len(rows)
    n_if = cfg.n_if_target if cfg.n_if_target and cfg.n_if_target > 0 else n
    if n_if >= n:
        n_if = max(1, n // 2)
        print(f"  [data] only {n} MMLU-CS rows; splitting {n_if} IF / {n - n_if} eval.")
    return rows[:n_if], rows[n_if:]


def load_mmlu_cs(cfg: ExperimentConfig, limit: int) -> list[dict]:
    """CS eval = the eval half of the MMLU-CS partition (disjoint from IF target)."""
    _, eval_rows = _mmlu_cs_partition(cfg)
    if limit and len(eval_rows) > limit:
        eval_rows = eval_rows[:limit]
    return [
        _eval_row(r["question"], r["answer"], source="mmlu_cs",
                  answer_type=r["answer_type"], category=r["category"])
        for r in eval_rows
    ]


# Heavier benchmarks the proposal listed but not yet wired (need own harnesses).
_UNIMPLEMENTED = {
    "olympiadbench": "Hothan/OlympiadBench — multimodal subsets need filtering.",
    "aime25": "AIME 2025 — tiny (30 q); add a static loader.",
    "finqa": "ibm-research/finqa or dreamerdeo/finqa — needs table context in prompt.",
    "livecodebench": "livecodebench/* — needs the LCB execution harness, not verifier.",
    "swebench": "princeton-nlp/SWE-bench — agentic; out of scope for verifier scoring.",
    "theoremqa": "TIGER-Lab/TheoremQA HF release has no subject/field column → can't filter to CS.",
}

EVAL_LOADERS = {
    "webinstruct_test": load_webinstruct_test,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "mmlu_cs": load_mmlu_cs,
}


def load_eval_benchmark(name: str, cfg: ExperimentConfig, limit: int) -> list[dict]:
    if name not in EVAL_LOADERS:
        hint = _UNIMPLEMENTED.get(name, "no loader registered")
        raise NotImplementedError(f"Eval benchmark {name!r} not implemented yet ({hint}).")
    return EVAL_LOADERS[name](cfg, limit)
