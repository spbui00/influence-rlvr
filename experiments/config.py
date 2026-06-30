"""Experiment configuration for the scaled GRPO + IF-pruning study.

One frozen-ish dataclass holds every knob. It is JSON-serializable so each run
writes a `config.json` next to its outputs, and Slurm jobs can override fields
from the command line via `ExperimentConfig.from_cli()`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import types
import typing
from dataclasses import dataclass
from pathlib import Path


# WebInstruct-verified `category` strings we keep. The dataset uses these exact
# title-cased names (see the dataset card: 16 categories). We map the user-facing
# domain {math, cs, finance} to the dataset's category labels.
DOMAIN_TO_CATEGORIES: dict[str, tuple[str, ...]] = {
    "math": ("Mathematics",),       # 78k train — quant (Expression/Float)
    "cs": ("Computer Science",),    # 1.2k train — too small to target
    "finance": ("Finance",),        # 13k train — quant (Float/Percentage)
    "physics": ("Physics",),        # 55k train — quant (Float/Integer)
    "chemistry": ("Chemistry",),    # 24k train — quant
    "business": ("Business",),      # 23k train — mixed
    "economics": ("Economics",),    # 11k train — quant-ish
    "biology": ("Biology",),        # 5.7k train — verbal/knowledge (distractor)
    "history": ("History",),        # 5.9k train — verbal/knowledge (distractor)
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
    # Each re-rank keeps the top `keep_fraction` of the pool BY INFLUENCE. Breadth is
    # set here, decoupled from window length — smaller = stronger IF-vs-random contrast
    # but more epochs over fewer prompts.
    keep_fraction: float = 0.5
    # Order the kept set is trained in. True = shuffle (covers the whole kept set even
    # when a window is shorter than it; decorrelates batches). False = consume in
    # influence order, best->worst (a curriculum sweep — but if a window is shorter
    # than the kept set it only ever reaches the top, so size window ~= kept set).
    if_shuffle_kept: bool = True
    # "if-guided" keeps highest-influence; "anti-if" keeps lowest (ablation);
    # "random" keeps a random subset of the same size (control); "in-domain" keeps only
    # target-domain prompts (the "just train in-domain" heuristic baseline — needs no
    # influence, uses the SAME pool so the carve/eval matches the IF arm).
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
    n_train_pool: int = 6000 # <=0 means "use everything after filtering".
    # Draw an equal share from each domain (n_train_pool/#domains) instead of a
    # random (Math-heavy) sample. Keep on so a single-domain IF target (e.g. CS)
    # can pull in helpful examples from the other domains.
    balance_domains: bool = True
    # Held-out target set the influence is measured against (drawn from the
    # dataset `test` split). The test slice is partitioned into a disjoint IF
    # target (first n_if_target) and an in-distribution eval set (the rest), so
    # the eval never overlaps the influence target.
    n_if_target: int = 64
    # Restrict the WebInstruct *test* partition (both IF target AND the
    # in-distribution `webinstruct_test` eval) to these domains. Empty -> use
    # `domains`. e.g. ("cs",) to steer pruning toward, and test on, Computer
    # Science specifically while still training on all `domains`.
    webinstruct_test_domains: tuple[str, ...] = ()
    # Difficulty-transfer (train easy → eval hard), using WebInstruct's `difficulty` field
    # ('University','Senior High School','Junior High School','PhD'). target_difficulty
    # restricts the IF-target + held-out eval (the TEST domain) to that level; pool_difficulty
    # restricts the TEST domain's POOL rows to those levels (distractor domains unrestricted).
    # e.g. target_difficulty="University", pool_difficulty=("Junior High School","Senior High
    # School") → target/eval = uni physics, pool physics = HS only (disjoint by difficulty).
    target_difficulty: str = ""
    pool_difficulty: tuple[str, ...] = ()
    # Use the ENTIRE domain-filtered test slice as the IF target (no eval
    # carve-out). For tiny single-domain slices (e.g. CS has ~5 test rows) so all
    # of them guide the influence. REQUIRES eval on an external dataset — do NOT
    # keep "webinstruct_test" in eval_benchmarks with this on (it would leak).
    if_target_full_test: bool = False
    # Where the IF target comes from: "webinstruct" (held-out WebInstruct test
    # partition), "mmlu_cs" (held-out half of MMLU-CS), or "mmlu_pro_cs" (held-out
    # half of MMLU-Pro CS — harder, 10-option, more eval headroom). The MMLU(-Pro)
    # options are the SAME distribution as their matching eval, so the influence
    # steers toward what we measure. Training pool is always WebInstruct (the
    # cross-domain candidate set).
    if_target_source: str = "webinstruct"
    # Draw the IF target + held-out eval from the TRAIN split instead of the tiny,
    # math-skewed test slice (1000 rows total: 351 Math / 62 Finance / 5 CS). For a
    # single-domain target (e.g. finance) the test slice is far too small/noisy to
    # resolve arms; the train split is huge (78k / 13k / 1.2k). Carved from the SAME
    # per-domain shuffle the pool draws from but AFTER the pool's share, so the
    # target/eval stay DISJOINT from the training pool. Pick the target domain via
    # `webinstruct_test_domains` (e.g. ("finance",)). Pool stays all `domains`.
    test_from_train: bool = False
    test_from_train_eval: int = 1000   # held-out eval pool size per target domain
    max_prompt_length: int = 1024

    # ── GRPO ────────────────────────────────────────────────────────────────
    max_steps: int = 400
    save_steps: int = 25
    # Resume baseline training from the latest checkpoint in the run dir (set via
    # --resume). Survives time-limit kills, and lets you train N steps then bump
    # max_steps and continue. checkpoint-0 is the base anchor, not a resume point.
    resume: bool = False
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
    grpo_epsilon_high: float = 0.3    # asymmetric upper clip ("clip-higher", GR-4B/DAPO)
    grpo_temperature: float = 0.7     # rollout sampling temperature (GR-4B Table 9)
    # "group" divides advantage by group std (vanilla GRPO — over-weights easy/hard
    # questions, the Dr.GRPO critique); "none" removes it (Dr.GRPO). "batch" = batch std.
    grpo_scale_rewards: str = "group"
    # "grpo" = per-response length norm (length-bias); "dr_grpo" = constant norm (no
    # length bias); "bnpo" = token-level (DAPO-ish).
    grpo_loss_type: str = "grpo"
    max_completion_length: int = 2048
    # NCCL collective timeout (s), applied via GRPOConfig/TrainingArguments (NOT the
    # accelerate YAML — the Alliance build rejects a ddp_timeout key there). Default
    # 1800 (30 min) aborts a slow HF-gen round; 7200 covers any realistic generation.
    ddp_timeout: int = 7200
    # Reward shaping matches General-Reasoner (the WebInstruct-verified authors,
    # same Qwen3-4B-Base + GRPO): extraction fails (no \boxed{}/marker) -> reward
    # -extraction_penalty; verifier correct -> +1 - length_penalty; wrong -> 0.
    # The -0.5 keeps answers parseable AND injects within-group reward variance so
    # all-wrong GRPO groups still produce a gradient (the v2 null was a strict
    # <think>+\boxed guardrail that fired ~never -> dead term + group collapse).
    # length_penalty = length_penalty_coef * min(length_penalty_cap,
    #   |tok(answer) - tok(gold)|), correct answers only. Set extraction_penalty=0
    # AND length_penalty_coef=0 for pure verifier-only {0,1}.
    extraction_penalty: float = 0.5
    length_penalty_coef: float = 0.05
    length_penalty_cap: int = 10
    # Optional EXTRA \boxed{}/<think> shaping bonus on top of the GR reward. Off by
    # default (GR uses no separate format reward; its extraction penalty subsumes it).
    use_format_guardrail: bool = False

    # ── Generation backend ──────────────────────────────────────────────────
    use_vllm: bool = True
    # "colocate" — vLLM engine in the training process (1 GPU; can't span the
    #   if_prune windows → if_prune falls back to HF). "server" — vLLM runs as a
    #   separate `trl vllm-serve` process (own GPU); training connects over HTTP,
    #   so if_prune CAN use vLLM (no in-process engine). Use server with l40s:2.
    vllm_mode: str = "colocate"       # "colocate" | "server"
    vllm_gpu_memory_utilization: float = 0.4   # colocate only
    vllm_max_model_len: int | None = 4096
    vllm_enable_sleep_mode: bool = True        # colocate only
    vllm_server_host: str = "127.0.0.1"        # server mode
    vllm_server_port: int = 8000               # server mode (HTTP API)
    # TRL's weight-sync NCCL group port (server mode, TRL>=1.x). MUST differ from
    # vllm_server_port: TRL's own default for this is 51216 — which is exactly the
    # HTTP port our old slurm used, so the sync TCPStore hit the HTTP server and
    # failed ("Ping failed, invalid value"). Keep them disjoint. Ignored by TRL
    # 0.29 (filtered out in make_grpo_config), so the HF path is unaffected.
    vllm_group_port: int = 51217               # server mode (weight-sync group)
    vllm_server_timeout: float = 240.0         # server mode (init/connect timeout, s)

    # ── Verifier (reward model) ─────────────────────────────────────────────
    verifier_model_id: str = "TIGER-Lab/general-verifier"
    verifier_max_new_tokens: int = 512
    verifier_batch_size: int = 16
    # The general-verifier's hard context cap (Qwen2-1.5B, max_position_embeddings
    # =4096). The vLLM server is launched with this, and the client truncates each
    # grading prompt (input + verifier_max_new_tokens) below it so a long completion
    # can't 400 the request. Must match the server's --max-model-len.
    verifier_max_model_len: int = 4096
    # Run the verifier on a separate device if available (e.g. "cuda:1"); else
    # it shares the policy GPU. None -> auto.
    verifier_device: str | None = None
    # Verifier generation backend. "hf" = in-process HF .generate() (shares the
    # policy GPU; the whole batch stalls on the slowest "Final Decision:" — slow).
    # "vllm" = query a standalone OpenAI-compatible vLLM server (continuous
    # batching, ~10-20x faster). The verifier is a FIXED model, so the server needs
    # NO weight sync — just `vllm serve <verifier> on its own GPU`. With "vllm" you
    # can afford a high verifier_max_new_tokens (1024) so the verifier's reasoning
    # isn't truncated before its decision line (512 silently mis-grades the tail).
    verifier_backend: str = "hf"          # "hf" | "vllm"
    verifier_server_host: str = "127.0.0.1"
    verifier_server_port: int = 8100

    # ── Influence (IF) computation ──────────────────────────────────────────
    # "cg"           — true policy Fisher (analytic per-token softmax metric
    #                  M=diag(p)−ppᵀ), matrix-free FVP via double-backward, solved
    #                  with conjugate gradients. Full-rank, low-variance. Default.
    # "cg-empirical" — sampled policy Fisher from cached per-completion gradients
    #                  (rank ≤ N·G). Cheaper, coarser; good for ablation/compare.
    # "fisher"       — damped "outer-of-means" inverse, closed-form Woodbury
    #                  (rank ≤ N). Cheapest; for fast smoke runs.
    # "dot"          — first-order TracIn: IF = g_train·g_test (no Fisher/solve).
    # "tracin-adam"  — first-order TracIn preconditioned by Adam's diagonal,
    #                  IF = g_train·(P⊙g_test), P_d=1/(√v̂_d+ε) read from the
    #                  checkpoint's optimizer.pt. The faithful first-order effect
    #                  of one *AdamW* step (what training actually does), not SGD.
    #                  No λ/CG → nothing to converge; skips the Fisher entirely.
    if_method: str = "cg"
    # ── Influence gradient SOURCE (orthogonal to the if_method operator above) ──
    # What g_train/g_test ARE — the same operator (dot/tracin-adam/cg) applies to either:
    #   "rollout" (default) — on-policy GRPO gradient: sample if_g_train rollouts,
    #              advantage-weighted ∇logπ. Faithful, but needs generation (expensive).
    #   "gold"    — SFT gold-answer gradient ∇[−logπ(\boxed{y_gold}|x)]: ONE teacher-
    #              forced forward+backward, NO rollouts (~30 ms/ex). Cheap surrogate; e.g.
    #              if_method=tracin-adam + if_grad=gold = tracin-adam on the gold gradient.
    #              Reads `solution` (the verifier ground truth).
    if_grad: str = "rollout"
    # tracin-adam only: override Adam's optimization ε (≈1e-8) with a larger floor
    # on the preconditioner denominator 1/(√v̂+ε). 0 = use the optimizer's own ε
    # (faithful). Raise it (e.g. 1e-4) if dormant-coordinate P-spikes make the
    # influence ranking noisy — check the [tracin-adam] P-range log + seed-to-seed
    # ρ. Ignored by every other if_method.
    tracin_adam_eps: float = 0.0
    lambda_damp: float = 0.1
    cg_iters: int = 50
    cg_tol: float = 1e-6
    # Spectral-normalize the Fisher (rescale by its top eigenvalue → spectrum in
    # [0,1]) so lambda_damp is a scale-free knob in (0,1) and transfers across
    # checkpoints/pool sizes. Also conditions (F+λI), helping CG converge at small λ.
    cg_normalize_fisher: bool = True
    cg_power_iters: int = 15
    # rollouts/tokens per example when scoring grads. Each g_test/g_train backward
    # holds logits of shape (if_g_train × tokens × vocab≈152k) — keep modest on a
    # 48 GB L40S that also hosts the policy + verifier (8 OOMs there).
    if_g_train: int = 4 # how many rollouts per prompt for influence on the train pool
    if_max_new_tokens: int = 512
    # CG scoring minibatch: how many pool/target prompts to score per call. The
    # batched bundle generates all B×if_g_train rollouts in ONE forward (fills the
    # GPU) instead of one prompt at a time. Backward is still per-prompt inside, so
    # memory scales with generation (B × if_g_train × if_max_new_tokens), not B×D.
    # 1 == the old one-at-a-time loop. Raise until generation saturates the GPU.
    if_score_batch: int = 1
    # Offload the per-example ROLLOUT SAMPLING (not the gradient) in influence scoring
    # to vLLM — the slow half. Only active in vLLM *server* mode (no colocate engine to
    # clash with); the scoring engine coexists with the HF model on the train GPU, so
    # keep if_vllm_gpu_util small. The gradient stays exact (HF backward on the same
    # tokens), so this is a pure speedup, no quality change.
    if_vllm_gen: bool = False
    if_vllm_gpu_util: float = 0.3
    # Forward-mode (JVP) pool scoring for if_grad=gold ONLY. Replaces the per-example
    # backward+dot with ONE fp32 eager-attn forward carrying the fixed tangent
    # h_bar = H.mean(0) (= P⊙ḡ_target), so score(z)=⟨∇L(z),h_bar⟩ comes out directly —
    # no backward, no 34.8M-D gradient vector. Same ranking (validated Spearman≈1, incl.
    # the Adam-preconditioned operator). Loads a fresh fp32 model from checkpoint-{step}
    # (~2× base GPU mem per rank) since bf16 forward-mode AD is buggy through Qwen3 norms.
    # Errors unless if_grad=gold (rollout needs generation → no fixed differentiable loss).
    if_jvp: bool = False
    # Prompts per forward-mode pass in the JVP scorer. Bounds the [batch × seq × vocab≈152k]
    # logits tensor (fp32, ~doubled by the forward-mode tangent) — a DIFFERENT, much heavier
    # memory profile than if_score_batch (which sizes rollout GENERATION). Keep modest: 8 ≈
    # 10 GB logits at seq 1024; raise on an 80 GB H100, lower on a 48 GB L40S. Also caps the
    # first-call correctness gate's batch so it never OOMs past the production batch.
    if_jvp_batch: int = 8
    # if_cosine: rank pool examples by COSINE alignment to the target instead of the raw dot
    # (LESS-style). The raw dot scales with |g_train|, so large-gradient examples dominate
    # regardless of target-direction (a physics target selected mostly Economics). Cosine
    # strips magnitude → selects by direction. Reverse-mode only (JVP can't get |g_train|).
    if_cosine: bool = False
    # if_project_common_mode: before collapsing the target gradients to their mean tangent,
    # remove the top principal direction of H (the SHARED "common mode" across targets — for
    # a short-answer gold target this is the answer-format / answer-distribution direction
    # that makes the influence a format filter). Scores then rank by the CONTENT-specific
    # component (alignment with what makes THIS target right beyond generic format). Gram-trick
    # (eigh of H Hᵀ, [n_target,n_target]) so no [n_target,D] V is materialized. CLUSTER-VALIDATE.
    if_project_common_mode: bool = False
    # Logits microbatch for the per-token-logp forward during scoring. The lm_head
    # materializes (micro_batch × seq × vocab≈152k) logits — the dominant memory
    # spike in the scoring backward. 1 = one sequence's logits at a time (minimum
    # memory; needed on a 48 GB L40S at if_g_train≥4). Does NOT change the result,
    # only how many rollouts' logits are resident at once.
    if_logps_micro_batch: int = 1
    # Fisher batch for CG: the Fisher is estimated over `cg_fisher_examples`
    # prompts × `cg_fisher_g` completions. For "cg" each completion is truncated
    # to `cg_fisher_max_tokens` positions (the Fisher needn't see full length).
    # For "cg-empirical" the cached stacks live on GPU: memory grows as
    # cg_fisher_examples · cg_fisher_g · D · 4 bytes. Lower these on A100-40G.
    cg_fisher_examples: int = 16
    cg_fisher_g: int = 4
    # Caps BOTH the Fisher prompt and response length. The FVP's math-attention
    # double-backward is O(seq²) in memory, so keep this modest on a 48 GB L40S
    # (total seq ≈ 2× this). 512 OOMs on long prompts; 256 is safe.
    cg_fisher_max_tokens: int = 256

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

    # ── Live (in-training) held-out eval ────────────────────────────────────
    # Periodically score the disjoint held-out eval set (the eval half of the
    # WebInstruct test partition, CS-only if webinstruct_test_domains=("cs",))
    # with the verifier and log eval/accuracy to W&B + a CSV, so baseline and
    # if_prune yield comparable held-out-accuracy-vs-step curves. This is the
    # fair comparison (training reward is over different data per regime).
    live_eval: bool = True
    live_eval_every: int = 0          # 0 -> use save_steps
    live_eval_examples: int = 64      # held-out prompts scored each time (kept small)
    live_eval_max_new_tokens: int = 1024
    # Which benchmark the in-training curve uses (any key in data.EVAL_LOADERS).
    # Default webinstruct_test; set to an EXTERNAL set (e.g. theoremqa_cs) when
    # if_target_full_test consumes the whole webinstruct test slice for the IF target.
    live_eval_benchmark: str = "webinstruct_test"

    def __post_init__(self) -> None:
        for d in (*self.domains, *self.webinstruct_test_domains):
            if d not in DOMAIN_TO_CATEGORIES:
                raise ValueError(
                    f"Unknown domain {d!r}. Known: {sorted(DOMAIN_TO_CATEGORIES)}"
                )
        if self.regime not in ("baseline", "if_prune"):
            raise ValueError(f"regime must be 'baseline' or 'if_prune', got {self.regime!r}")
        if self.selection not in ("if-guided", "anti-if", "random", "in-domain"):
            raise ValueError(f"selection must be if-guided|anti-if|random|in-domain, got {self.selection!r}")
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
        if self.if_target_source not in ("webinstruct", "mmlu_cs", "mmlu_pro_cs"):
            raise ValueError(
                f"if_target_source must be webinstruct|mmlu_cs|mmlu_pro_cs, "
                f"got {self.if_target_source!r}"
            )
        if self.if_method not in ("cg", "cg-empirical", "fisher", "dot", "tracin-adam"):
            raise ValueError(
                f"if_method must be cg|cg-empirical|fisher|dot|tracin-adam, "
                f"got {self.if_method!r}"
            )
        if self.if_grad not in ("rollout", "gold"):
            raise ValueError(f"if_grad must be rollout|gold, got {self.if_grad!r}")
        if self.if_cosine and self.if_jvp:
            raise ValueError(
                "if_cosine + if_jvp are incompatible: cosine needs each pool example's "
                "gradient NORM |g_train|, which the forward-mode JVP path never materializes. "
                "Use reverse-mode (drop --if-jvp) with --if-cosine."
            )
        if self.if_jvp and self.if_grad != "gold":
            raise ValueError(
                "if_jvp requires if_grad=gold (forward-mode JVP scores the SFT gold "
                "gradient; rollout gradients need generation, so there is no fixed "
                "differentiable loss to take the directional derivative of)."
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
            if k in {"lora_target_modules", "domains", "eval_benchmarks", "webinstruct_test_domains", "pool_difficulty"} and isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ── CLI ─────────────────────────────────────────────────────────────────
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register one --flag per scalar field, with the dataclass default.

        Dispatch the argparse ``type=`` from the field's DECLARED type, not the
        default's runtime type. A field like ``generation_batch_size: int | None
        = None`` has default ``None``, so ``isinstance(default, int)`` is False —
        it would silently parse as a string ("56"), which then blows up
        downstream (TRL does ``"56" % n`` -> "not all arguments converted during
        string formatting"). Resolving the annotation handles ``int | None``.
        """
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {}

        def scalar_type(field_name: str):
            """Non-None scalar of a field's type, unwrapping Optional / ``X|None``."""
            tp = hints.get(field_name)
            origin = typing.get_origin(tp)
            if origin is typing.Union or origin is getattr(types, "UnionType", object()):
                non_none = [a for a in typing.get_args(tp) if a is not type(None)]
                if len(non_none) == 1:
                    return non_none[0]
            return tp

        parser.add_argument("--config", type=str, default=None,
                            help="Path to a config.json to load as the base (CLI flags override).")
        for f in dataclasses.fields(cls):
            if f.name == "config":
                continue
            default = f.default
            name = "--" + f.name.replace("_", "-")
            base = scalar_type(f.name)
            if base is bool or isinstance(default, bool):
                # Support --flag / --no-flag.
                parser.add_argument(name, dest=f.name, action="store_true", default=None)
                parser.add_argument("--no-" + f.name.replace("_", "-"),
                                    dest=f.name, action="store_false")
            elif f.name in {"lora_target_modules", "domains", "eval_benchmarks", "webinstruct_test_domains", "pool_difficulty"}:
                parser.add_argument(name, type=str, default=None,
                                    help="Comma-separated list.")
            elif base is int or (base is None and isinstance(default, int)):
                parser.add_argument(name, type=int, default=None)
            elif base is float or (base is None and isinstance(default, float)):
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
            if f.name in {"lora_target_modules", "domains", "eval_benchmarks", "webinstruct_test_domains", "pool_difficulty"} and isinstance(val, str):
                val = tuple(p.strip() for p in val.split(",") if p.strip())
            overrides[f.name] = val
        merged = {**base.to_dict(), **overrides}
        return cls.from_dict(merged)


# Default singleton useful for `python -c` smoke checks.
DEFAULT_CONFIG = ExperimentConfig()
