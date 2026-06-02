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
    """Held-out target set the influence is measured against (disjoint from eval)."""
    if_raw, _ = _webinstruct_test_partition(cfg)
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
