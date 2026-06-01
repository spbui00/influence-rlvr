"""Experiment configuration for the scaled GRPO + IF-pruning study.

One frozen-ish dataclass holds every knob. It is JSON-serializable so each run
writes a `config.json` next to its outputs, and Slurm jobs can override fields
from the command line via `ExperimentConfig.from_cli()`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


# WebInstruct-verified `category` strings we keep. The dataset uses these exact
# title-cased names (see the dataset card: 16 categories). We map the user-facing
# domain {math, cs, finance} to the dataset's category labels.
DOMAIN_TO_CATEGORIES: dict[str, tuple[str, ...]] = {
    "math": ("Mathematics",),
    "cs": ("Computer Science",),
    "finance": ("Finance",),
}


@dataclass
class ExperimentConfig:
    # ── Run identity / paths ────────────────────────────────────────────────
    run_name: str = "qwen3_4b_webinstruct"
    output_root: str = "./outputs"
    seed: int = 42

    # ── Regime ──────────────────────────────────────────────────────────────
    # "baseline"  — straight GRPO for `max_steps` on the full (sampled) pool.
    # "if_prune"  — train to `prune_step`, compute influence on the target set,
    #               keep the top `keep_fraction` of the pool, continue to
    #               `max_steps` on that pruned subset.
    regime: str = "baseline"
    prune_step: int = 50
    # Dynamic pruning: after `prune_step`, re-score the whole pool and re-select
    # the kept subset every `if_recompute_every` steps, until `max_steps` (the
    # toy's repeated-recompute regime — the subset adapts as the policy moves,
    # and a dropped example can re-enter a later window). Set <= 0 for a single
    # one-shot prune at `prune_step`. Boundaries must be multiples of save_steps
    # (resume across windows needs a full checkpoint there).
    if_recompute_every: int = 50
    # NOTE: in the dynamic ranked-pop regime the fraction of the pool actually
    # trained on is set by the window length (W × prompts_per_step prompts are
    # popped from the ranking per window), NOT by a hard cut. keep_fraction is
    # retained only for the legacy one-shot `select_keep_indices` helper.
    keep_fraction: float = 0.5
    # "if-guided" keeps highest-influence; "anti-if" keeps lowest (ablation);
    # "random" keeps a random subset of the same size (control).
    selection: str = "if-guided"

    # ── Model / LoRA ────────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen3-4B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    domains: tuple[str, ...] = ("math", "cs", "finance")
    train_dataset: str = "TIGER-Lab/WebInstruct-verified"
    # Cap the training pool (per the IF-pruning study we don't need all 229k).
    # <=0 means "use everything after filtering".
    n_train_pool: int = 6000
    # Held-out target set the influence is measured against (drawn from the
    # dataset `test` split, filtered to the same domains).
    n_if_target: int = 64
    max_prompt_length: int = 1024

    # ── GRPO ────────────────────────────────────────────────────────────────
    max_steps: int = 400
    save_steps: int = 25
    learning_rate: float = 1e-5
    # Effective batch = per_device_batch × grad_accum × num_processes = 64 here,
    # giving prompts_per_step = 64 / g_train = 8 distinct prompts per optimizer
    # step (16 rows resident per micro-step). On A100-40G, if 16 resident rows
    # OOM, use per_device_batch=8 + grad_accum=8 (same 8 prompts/step, half the
    # resident memory, ~2× the micro-steps).
    per_device_batch: int = 16
    grad_accum: int = 4
    g_train: int = 8                  # GRPO num_generations (group size)
    generation_batch_size: int | None = None
    grpo_beta: float = 0.0            # KL coeff; 0 disables the reference model
    grpo_epsilon: float = 0.2
    max_completion_length: int = 2048
    # Light format guardrail reward in addition to the verifier (encourages
    # <think>…</think> + \boxed{}). Set False for verifier-only reward.
    use_format_guardrail: bool = True

    # ── Generation backend ──────────────────────────────────────────────────
    use_vllm: bool = True
    vllm_mode: str = "colocate"       # "colocate" | "server"
    vllm_gpu_memory_utilization: float = 0.4
    vllm_max_model_len: int | None = 4096
    vllm_enable_sleep_mode: bool = True

    # ── Verifier (reward model) ─────────────────────────────────────────────
    verifier_model_id: str = "TIGER-Lab/general-verifier"
    verifier_max_new_tokens: int = 512
    verifier_batch_size: int = 16
    # Run the verifier on a separate device if available (e.g. "cuda:1"); else
    # it shares the policy GPU. None -> auto.
    verifier_device: str | None = None

    # ── Influence (IF) computation ──────────────────────────────────────────
    # "cg"           — true policy Fisher (analytic per-token softmax metric
    #                  M=diag(p)−ppᵀ), matrix-free FVP via double-backward, solved
    #                  with conjugate gradients. Full-rank, low-variance. Default.
    # "cg-empirical" — sampled policy Fisher from cached per-completion gradients
    #                  (rank ≤ N·G). Cheaper, coarser; good for ablation/compare.
    # "fisher"       — damped "outer-of-means" inverse, closed-form Woodbury
    #                  (rank ≤ N). Cheapest; for fast smoke runs.
    if_method: str = "cg"
    lambda_damp: float = 0.1
    cg_iters: int = 50
    cg_tol: float = 1e-6
    if_g_train: int = 8               # rollouts per example when scoring grads
    if_max_new_tokens: int = 1024
    # Fisher batch for CG: the Fisher is estimated over `cg_fisher_examples`
    # prompts × `cg_fisher_g` completions. For "cg" each completion is truncated
    # to `cg_fisher_max_tokens` positions (the Fisher needn't see full length).
    # For "cg-empirical" the cached stacks live on GPU: memory grows as
    # cg_fisher_examples · cg_fisher_g · D · 4 bytes. Lower these on A100-40G.
    cg_fisher_examples: int = 16
    cg_fisher_g: int = 4
    cg_fisher_max_tokens: int = 512

    # ── Eval ────────────────────────────────────────────────────────────────
    eval_benchmarks: tuple[str, ...] = (
        "webinstruct_test",   # in-distribution held-out, all 3 domains
        "gsm8k",
        "math500",
    )
    eval_max_examples: int = 200
    eval_max_new_tokens: int = 2048
    eval_temperature: float = 0.0     # greedy by default
    eval_top_p: float = 1.0

    def __post_init__(self) -> None:
        for d in self.domains:
            if d not in DOMAIN_TO_CATEGORIES:
                raise ValueError(
                    f"Unknown domain {d!r}. Known: {sorted(DOMAIN_TO_CATEGORIES)}"
                )
        if self.regime not in ("baseline", "if_prune"):
            raise ValueError(f"regime must be 'baseline' or 'if_prune', got {self.regime!r}")
        if self.selection not in ("if-guided", "anti-if", "random"):
            raise ValueError(f"selection must be if-guided|anti-if|random, got {self.selection!r}")
        if self.regime == "if_prune":
            # Window boundaries must land on full checkpoints so each window can
            # resume the optimizer/LR state from the previous one.
            if self.prune_step % self.save_steps != 0:
                raise ValueError(
                    f"prune_step ({self.prune_step}) must be a multiple of "
                    f"save_steps ({self.save_steps}) so the prune checkpoint exists."
                )
            if self.if_recompute_every > 0 and self.if_recompute_every % self.save_steps != 0:
                raise ValueError(
                    f"if_recompute_every ({self.if_recompute_every}) must be a "
                    f"multiple of save_steps ({self.save_steps})."
                )
        if self.if_method not in ("cg", "cg-empirical", "fisher"):
            raise ValueError(
                f"if_method must be cg|cg-empirical|fisher, got {self.if_method!r}"
            )

    # ── Derived paths ───────────────────────────────────────────────────────
    @property
    def run_dir(self) -> Path:
        return Path(self.output_root).expanduser().resolve() / self.run_name

    @property
    def grpo_output_dir(self) -> Path:
        return self.run_dir / "rlvr-output"

    @property
    def categories(self) -> list[str]:
        cats: list[str] = []
        for d in self.domains:
            cats.extend(DOMAIN_TO_CATEGORIES[d])
        return cats

    # ── (De)serialization ───────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Normalize tuples -> lists for clean JSON.
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path is not None else (self.run_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}
        for k, v in d.items():
            if k not in fields:
                continue
            # Restore tuple-typed fields.
            if k in {"lora_target_modules", "domains", "eval_benchmarks"} and isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ── CLI ─────────────────────────────────────────────────────────────────
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register one --flag per scalar field, with the dataclass default."""
        parser.add_argument("--config", type=str, default=None,
                            help="Path to a config.json to load as the base (CLI flags override).")
        for f in dataclasses.fields(cls):
            if f.name == "config":
                continue
            default = f.default
            name = "--" + f.name.replace("_", "-")
            if f.type == "bool" or isinstance(default, bool):
                # Support --flag / --no-flag.
                parser.add_argument(name, dest=f.name, action="store_true", default=None)
                parser.add_argument("--no-" + f.name.replace("_", "-"),
                                    dest=f.name, action="store_false")
            elif f.name in {"lora_target_modules", "domains", "eval_benchmarks"}:
                parser.add_argument(name, type=str, default=None,
                                    help="Comma-separated list.")
            elif isinstance(default, int):
                parser.add_argument(name, type=int, default=None)
            elif isinstance(default, float):
                parser.add_argument(name, type=float, default=None)
            else:
                parser.add_argument(name, type=str, default=None)

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "ExperimentConfig":
        parser = argparse.ArgumentParser(description="Scaled GRPO + IF-pruning experiment.")
        cls.add_arguments(parser)
        args = parser.parse_args(argv)

        base = cls.load(args.config) if args.config else cls()
        overrides: dict = {}
        for f in dataclasses.fields(cls):
            if f.name == "config":
                continue
            val = getattr(args, f.name, None)
            if val is None:
                continue
            if f.name in {"lora_target_modules", "domains", "eval_benchmarks"} and isinstance(val, str):
                val = tuple(p.strip() for p in val.split(",") if p.strip())
            overrides[f.name] = val
        merged = {**base.to_dict(), **overrides}
        return cls.from_dict(merged)


# Default singleton useful for `python -c` smoke checks.
DEFAULT_CONFIG = ExperimentConfig()
