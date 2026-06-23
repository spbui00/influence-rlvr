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
from collections import Counter, defaultdict
from typing import Iterable

from datasets import Dataset, concatenate_datasets, load_dataset

from .config import DOMAIN_TO_CATEGORIES, ExperimentConfig


# EXACT General-Reasoner recipe (their data_preprocess.py): GR-faithful ZERO-SHOT
# — the question + this one instruction line, single user message, joined by a
# space. No <think>, no system prompt, no exemplars. A base model bootstraps the
# format from the reward over many steps (GR did this on Qwen3-4B-Base — but at a
# MUCH larger batch than ours; that scale is what makes zero-shot bootstrap work).
# Used for BOTH train and eval (one source so they stay consistent).
GENERAL_REASONING_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_reasoning_prompt(question: str) -> list[dict]:
    return [{"role": "user", "content": f"{question} {GENERAL_REASONING_INSTRUCTION}"}]


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
        "difficulty": example.get("difficulty", ""),
        "train_index": idx,
    }


def _load_webinstruct_split(cfg: ExperimentConfig, split: str) -> Dataset:
    raw = load_dataset(cfg.train_dataset, split=split)
    keep = _categories_for_domains(cfg.domains)
    raw = raw.filter(lambda ex: ex.get("category") in keep)
    # Drop rows without a usable reference answer.
    raw = raw.filter(lambda ex: bool(str(ex.get("answer", "") or "").strip()))
    return raw


def _load_train_domain(cfg: ExperimentConfig, domain: str,
                       difficulties: tuple[str, ...] | None = None) -> Dataset:
    """Train split filtered to ONE domain's categories (+ nonempty answer). With
    `difficulties`, also restrict to those WebInstruct difficulty levels (e.g.
    ('University',) or ('Junior High School','Senior High School')) — the basis for
    difficulty-transfer (train easy → eval hard)."""
    cats = set(DOMAIN_TO_CATEGORIES[domain])
    raw = load_dataset(cfg.train_dataset, split="train")
    raw = raw.filter(lambda ex: ex.get("category") in cats)
    if difficulties:
        # Accept the underscore form ("Junior_High_School") from the shell-friendly CLI,
        # matched against the dataset's spaced labels ("Junior High School").
        diffs = {d.replace("_", " ") for d in difficulties}
        raw = raw.filter(lambda ex: ex.get("difficulty") in diffs)
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
        # Difficulty-transfer: restrict the TEST DOMAIN's pool rows to `pool_difficulty`
        # (e.g. high-school physics) so the pool never contains the target difficulty
        # (e.g. university physics). Distractor domains (math/finance) stay unrestricted.
        test_doms = set(_webinstruct_test_domains(cfg))
        parts = []
        for d in cfg.domains:
            diffs = cfg.pool_difficulty if (cfg.pool_difficulty and d in test_doms) else None
            raw_d = _load_train_domain(cfg, d, difficulties=diffs).shuffle(seed=cfg.seed)
            k = min(per, len(raw_d))
            if k < per:
                print(f"  [data] domain {d!r}: only {len(raw_d)} rows, using {k} (< {per})")
            parts.append(raw_d.select(range(k)))
        raw = concatenate_datasets(parts).shuffle(seed=cfg.seed + 5)
        print(f"  [data] balanced pool: {[(d, len(p)) for d, p in zip(cfg.domains, parts)]}"
              + (f" | test-domain pool difficulty={cfg.pool_difficulty}" if cfg.pool_difficulty else ""))
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


def _train_heldout_for_target(cfg: ExperimentConfig) -> tuple[Dataset, Dataset]:
    """`test_from_train`: carve a DISJOINT (if_target, eval) from the TRAIN split.

    The WebInstruct *test* split is tiny + math-skewed (351 Math / 62 Finance / 5 CS),
    so a single-domain target/eval can't live there. The *train* split is huge
    (78k / 13k / 1.2k). For each `webinstruct_test_domain` we hold out a slice from the
    SAME shuffle `load_train_pool` draws from (`_load_train_domain(d).shuffle(seed)`),
    but AFTER the pool's share `per`, so target+eval never overlap the pool. Layout per
    domain (one fixed shuffle):
        [0 : per]              -> consumed by load_train_pool (the training pool)
        [per : per+n_if]       -> IF target
        [per+n_if : +n_eval]   -> held-out eval
    `per` is the pool's per-domain share IFF d is a balanced pool domain, else 0.
    """
    n_if = cfg.n_if_target if cfg.n_if_target and cfg.n_if_target > 0 else 64
    n_eval = (cfg.test_from_train_eval if cfg.test_from_train_eval
              and cfg.test_from_train_eval > 0 else 1000)
    tgt_diffs = (cfg.target_difficulty,) if cfg.target_difficulty else None
    if_parts, eval_parts = [], []
    for d in _webinstruct_test_domains(cfg):
        raw = _load_train_domain(cfg, d, difficulties=tgt_diffs).shuffle(seed=cfg.seed)
        n = len(raw)
        # The pool consumes the first `per` of THIS identical shuffle iff d is a balanced
        # pool domain AT THE SAME difficulty. With difficulty-transfer (target_difficulty
        # set), the pool's same-domain rows are a DIFFERENT difficulty (e.g. pool=HS,
        # target=University) → not in this `raw` at all → per=0. The pool_q guard below
        # is the airtight backstop for either case.
        per = 0
        if (d in cfg.domains and cfg.balance_domains and cfg.n_train_pool > 0
                and not cfg.target_difficulty):
            per = min(cfg.n_train_pool // len(cfg.domains), n)
        if per + n_if >= n:
            raise ValueError(
                f"test_from_train: domain {d!r} has only {n} train rows; the pool "
                f"takes {per}, leaving {n - per} for IF target+eval (need > {n_if}). "
                f"Lower --n-train-pool or --n-if-target.")
        if_end = min(per + n_if, n)
        eval_end = min(if_end + n_eval, n)
        if_parts.append(raw.select(range(per, if_end)))
        eval_parts.append(raw.select(range(if_end, eval_end)))
        print(f"  [data] test_from_train {d!r}: pool[0:{per}] | "
              f"IF[{per}:{if_end}]={if_end - per} | eval[{if_end}:{eval_end}]="
              f"{eval_end - if_end}  (of {n} train rows)")
    if_raw = concatenate_datasets(if_parts) if len(if_parts) > 1 else if_parts[0]
    eval_raw = concatenate_datasets(eval_parts) if len(eval_parts) > 1 else eval_parts[0]
    # Airtight guard for ANY seed: the index offset separates UNIQUE questions from the
    # pool, but WebInstruct has ~2.5% duplicate questions and 134 questions tagged under
    # >1 category — either could drop a target/eval question into the pool's share at a
    # different seed. So additionally drop any target/eval row whose question text is in
    # the actual training pool, and any eval row whose question is in the IF target.
    # (For the default seed this removes 0 — the index partition is already clean.)
    pool_q = set(load_train_pool(cfg)["question"])
    if_raw = if_raw.filter(lambda e: e["question"] not in pool_q)
    tgt_q = set(if_raw["question"])
    eval_raw = eval_raw.filter(
        lambda e: e["question"] not in pool_q and e["question"] not in tgt_q)
    return if_raw, eval_raw


def load_if_target_set(cfg: ExperimentConfig) -> Dataset:
    """Held-out target set the influence is measured against.

    Default: the IF half of the disjoint WebInstruct test partition. With
    `if_target_source="mmlu_cs"` / `"mmlu_pro_cs"`: the target half of the disjoint
    MMLU(-Pro)-CS partition (same distribution as the matching eval — no proxy
    mismatch). With `if_target_full_test`: the ENTIRE filtered WebInstruct test slice.
    """
    if cfg.if_target_source in ("mmlu_cs", "mmlu_pro_cs"):
        partition = (_mmlu_pro_cs_partition if cfg.if_target_source == "mmlu_pro_cs"
                     else _mmlu_cs_partition)
        target_rows, _ = partition(cfg)
        raw = Dataset.from_list(target_rows)
        print(f"  [data] IF target: {len(raw)} {cfg.if_target_source} prompts "
              f"(disjoint from the matching eval half)")
        return raw.map(_format_webinstruct_row, with_indices=True,
                       remove_columns=raw.column_names)

    if cfg.test_from_train:
        if_raw, _ = _train_heldout_for_target(cfg)
    elif cfg.if_target_full_test:
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
    """In-distribution eval — the eval half of the partition (disjoint from the IF
    target). With `test_from_train` the partition is the train-held-out slice;
    otherwise the test slice. Respects `webinstruct_test_domains` (e.g. finance-only)."""
    if cfg.test_from_train:
        _, eval_raw = _train_heldout_for_target(cfg)
    else:
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


# ── External MMLU / MMLU-Pro CS (clean, labeled CS) — used for BOTH the IF target
# and the eval (disjoint, stratified splits). Same multiple-choice → lettered-prompt
# formatting for both; the gold is the correct option's letter. ────────────────────
MMLU_CS_CONFIGS = (
    "college_computer_science",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
)


def _mc_row(question: str, choices: list[str], gold_idx: int, *, subject: str) -> dict:
    """One multiple-choice item as a WebInstruct-style row: lettered options +
    an instruction to answer with the letter, gold = that letter."""
    letters = "ABCDEFGHIJ"[: len(choices)]
    opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    q = (f"{question}\n\nOptions:\n{opts}\n\n"
         "Answer with the letter of the single correct option.")
    return {
        "question": q,
        "answer": letters[gold_idx],
        "answer_type": "Multiple Choice",
        "category": "Computer Science",
        "subject": subject,
    }


def _mmlu_cs_rows(cfg: ExperimentConfig) -> list[dict]:
    """Combined MMLU-CS `test` questions (cais/mmlu, 4 CS configs, 4 options each)."""
    rows: list[dict] = []
    for sub in MMLU_CS_CONFIGS:
        for ex in load_dataset("cais/mmlu", sub, split="test"):
            rows.append(_mc_row(ex["question"], list(ex["choices"]), int(ex["answer"]),
                                subject=sub))
    return rows


def _mmlu_pro_cs_rows(cfg: ExperimentConfig) -> list[dict]:
    """MMLU-Pro CS `test` questions (TIGER-Lab/MMLU-Pro, "computer science" category,
    up to 10 options — much harder/more-reasoning than MMLU, so more eval headroom)."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ds = ds.filter(lambda ex: ex.get("category") == "computer science")
    rows: list[dict] = []
    for ex in ds:
        # MMLU-Pro gives the gold as both a letter ("answer") and index; trust index.
        rows.append(_mc_row(ex["question"], list(ex["options"]), int(ex["answer_index"]),
                            subject="mmlu_pro_cs"))
    return rows


def _stratified_mc_partition(
    rows: list[dict], cfg: ExperimentConfig, label: str,
) -> tuple[list[dict], list[dict]]:
    """Deterministic disjoint (IF target, eval) split, STRATIFIED by `subject`:
    the IF target takes ~n_if_target/#subjects from each subject so it spans them
    all; the eval is the remainder. Always leaves ≥1 per subject for eval, so the
    two sets never overlap. Both the target and eval loaders call this, so they
    always agree on the split."""
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sub[r["subject"]].append(r)
    rng = random.Random(cfg.seed)
    per = (cfg.n_if_target // max(1, len(by_sub))) if (cfg.n_if_target and cfg.n_if_target > 0) else 0
    target, eval_rows = [], []
    for sub in sorted(by_sub):
        srows = by_sub[sub]
        rng.shuffle(srows)
        k = min(per, max(0, len(srows) - 1)) if per > 0 else 0
        target.extend(srows[:k])
        eval_rows.extend(srows[k:])
    rng.shuffle(target)
    rng.shuffle(eval_rows)
    print(f"  [data] {label} IF target by subject: {dict(Counter(r['subject'] for r in target))}")
    return target, eval_rows


def _mmlu_cs_partition(cfg: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    return _stratified_mc_partition(_mmlu_cs_rows(cfg), cfg, "MMLU-CS")


def _mmlu_pro_cs_partition(cfg: ExperimentConfig) -> tuple[list[dict], list[dict]]:
    return _stratified_mc_partition(_mmlu_pro_cs_rows(cfg), cfg, "MMLU-Pro-CS")


def _load_mc_eval(eval_rows: list[dict], source: str, limit: int) -> list[dict]:
    """Eval half of an MC partition → eval rows (disjoint from the IF target)."""
    if limit and len(eval_rows) > limit:
        eval_rows = eval_rows[:limit]
    return [
        _eval_row(r["question"], r["answer"], source=source,
                  answer_type=r["answer_type"], category=r["category"])
        for r in eval_rows
    ]


def load_mmlu_cs(cfg: ExperimentConfig, limit: int) -> list[dict]:
    """CS eval = the eval half of the MMLU-CS partition (disjoint from IF target)."""
    return _load_mc_eval(_mmlu_cs_partition(cfg)[1], "mmlu_cs", limit)


def load_mmlu_pro_cs(cfg: ExperimentConfig, limit: int) -> list[dict]:
    """CS eval = the eval half of the MMLU-Pro-CS partition (disjoint from IF target)."""
    return _load_mc_eval(_mmlu_pro_cs_partition(cfg)[1], "mmlu_pro_cs", limit)


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
    "mmlu_pro_cs": load_mmlu_pro_cs,
}


def load_eval_benchmark(name: str, cfg: ExperimentConfig, limit: int) -> list[dict]:
    if name not in EVAL_LOADERS:
        hint = _UNIMPLEMENTED.get(name, "no loader registered")
        raise NotImplementedError(f"Eval benchmark {name!r} not implemented yet ({hint}).")
    return EVAL_LOADERS[name](cfg, limit)
