# GRPO + influence-based data pruning

Scales the toy GRPO data-pruning study (`scripts/train_with_if_pruning.py`) to a
real LLM, to test whether **pruning the training pool by influence** improves an
RLVR run.

| Piece     | Choice |
|-----------|--------|
| Policy    | `Qwen/Qwen3-4B` + LoRA (rank 16, all attn+MLP proj) |
| Algorithm | GRPO (TRL), vLLM-colocated generation |
| Reward    | `TIGER-Lab/general-verifier` — generative answer-equivalence verifier |
| Data      | `TIGER-Lab/WebInstruct-verified` filtered to **Math / CS / Finance** |
| Influence | **conjugate gradients on the true sampled policy Fisher** (cached-gradient FVP) |

## The comparison

Two matched runs, identical compute budget (`max_steps`):

- **`baseline`** — straight GRPO on the full (sampled) pool for `max_steps`.
- **`if_prune`** — **dynamic ranked-pop pruning** (the toy's `build_schedule`
  regime): warm up on the full pool to `prune_step` (50), then **every
  `if_recompute_every` steps** (triggers 50, 100, 150, …) re-score the *whole*
  pool's influence on a held-out target set, **rank all of it**, and train the
  next window by consuming prompts **in ranked order, most-influential first**
  (`shuffle_dataset=False`). A window of `W` steps pops the top `W × prompts_per_step`
  ranked prompts; the low-influence tail is simply never reached — that's the
  pruning, with no arbitrary cut. The ranking refreshes each window (a prompt
  dropped early can resurface later), and the optimizer/LR state carries across
  windows via checkpoint resume, so total compute matches baseline.
  `--if-recompute-every 0` = a single one-shot ranking. (Boundaries must be
  multiples of `save_steps`.)

`prompts_per_step = (per_device_batch × grad_accum × num_processes) / g_train`
= (16 × 4) / 8 = **8** with the defaults. The whole run consumes
`max_steps × prompts_per_step` prompt-instances total (400 × 8 = 3200), so with a
6000 pool most of it is still never trained on — the ranking decides which
prompts get the budget. A `W`-step window pops the top `W × 8` ranked prompts,
each **once**, in descending-influence order (no repetition unless `W × 8`
exceeds the pool size, which would tile). To cover a fraction `f` of the pool
per window, set `if_recompute_every ≈ f × pool / prompts_per_step` (full coverage
of 6000 at 8 prompts/step ≈ 750 steps/window). The job prints this at startup.

Controls/ablations via `--selection`: `if-guided` (most influential first),
`anti-if` (least first — should *hurt* if IF has signal), `random` (size-matched).

### Held-out target & eval (no leakage)

The WebInstruct `test` slice is split **once, deterministically, into two disjoint
halves**: the first `n_if_target` rows are the **IF target** (what pruning steers
toward), the rest are the in-distribution **`webinstruct_test` eval**. They never
overlap, so the eval isn't measuring the examples the data-selection was tuned on.
The training pool comes from the `train` split, disjoint from both. External
benchmarks (`gsm8k`, `math500`, …) are separate datasets — the cleanest signal.

**Targeting a single domain (e.g. Computer Science):** `--webinstruct-test-domains cs`
restricts *both* the IF target and the `webinstruct_test` eval to CS, while training
still spans all `--domains`. So you prune the (math+cs+finance) pool by influence on
held-out **CS** and test on held-out **CS**:

```bash
sbatch --export=ALL,REGIME=if_prune,RUN_NAME=qwen3_4b_prune_cs,\
EXTRA_ARGS="--webinstruct-test-domains cs --eval-benchmarks webinstruct_test" \
  experiments/cluster/train.slurm
```

(When `webinstruct_test` spans all three domains, `evaluate.py` already reports a
per-category accuracy breakdown, so you also get CS/Math/Finance numbers separately.)

### Held-out accuracy *during* training (the comparison curve)

TRL only logs the **training reward** (verifier pass-rate on the prompts trained
that step) — and since `baseline` and `if_prune` train on *different* prompts each
step, that curve isn't comparable across regimes. So a `LiveEvalCallback` also runs
every `live_eval_every` steps (default = `save_steps`): it generates on the disjoint
held-out eval set (CS-only if you set `--webinstruct-test-domains cs`), scores with
the verifier, and logs `eval/accuracy` (+ `eval/acc_<category>`) to **W&B** and
`outputs/<run>/live_eval.csv`. Plot `eval/accuracy` vs step for both runs — *that's*
the fair baseline-vs-if_prune comparison. Disable with `--no-live-eval`; tune
`--live-eval-examples` / `--live-eval-max-new-tokens` for cost.

> **Cost:** every recompute re-scores the entire pool (one rollout + gradient per
> pool example). With K triggers that's K × `n_train_pool` scoring rollouts — the
> dominant cost. Keep `n_train_pool` modest, or raise `if_recompute_every`, when
> running many windows.

## Files

```
experiments/
  config.py     ExperimentConfig — every knob; --flag per field; JSON save/load
  data.py       WebInstruct train pool + IF-target set; eval-benchmark registry
  verifier.py   GeneralVerifier wrapper + TRL reward function
  influence.py  per-train influence at a checkpoint (reuses collect_checkpoint_infos)
  train.py      driver: `baseline` and `if_prune` regimes
  evaluate.py   score a checkpoint on benchmark suites (verifier-judged)
  cluster/      setup.sh · prefetch.py · train.slurm · eval.slurm
```

## Local smoke (tiny, CPU/1-GPU)

```bash
uv run python -m experiments.train \
  --regime if_prune --run-name smoke \
  --n-train-pool 16 --n-if-target 4 \
  --max-steps 9 --prune-step 3 --if-recompute-every 3 --save-steps 3 \
  --g-train 2 --per-device-batch 2 --grad-accum 1 \
  --cg-fisher-examples 4 --cg-fisher-g 2 --cg-iters 5 \
  --max-completion-length 256 --no-use-vllm \
  --verifier-max-new-tokens 128 --live-eval-examples 2
```

(Generation/verifier are slow on CPU — this just exercises the full code path.)

## On the Alliance cluster (Killarney → Narval → Nibi)

Compute nodes have **no internet**, so steps 1–2 run on a **login node**.

**1. Get the code + environment (login node)**
```bash
cd ~/scratch                      # build in scratch (space + visible to compute)
git clone <your-fork-url> influence-rlvr && cd influence-rlvr
bash experiments/cluster/setup.sh                # builds ~/envs/influence-rlvr
```
`setup.sh` loads `StdEnv/2023 gcc python/3.11 cuda arrow`, makes a virtualenv,
installs this repo + TRL/PEFT/transformers/datasets/vLLM. If a module version
isn't found, run `module spider python cuda arrow` and edit the versions at the
top of `setup.sh`. (Killarney is new — versions may differ from Narval/Nibi.)

**2. Pre-download models + datasets (login node)**
```bash
source ~/envs/influence-rlvr/bin/activate
HF_HOME=$HOME/scratch/hf_cache python experiments/cluster/prefetch.py
```
Caches Qwen3-4B, general-verifier, WebInstruct-verified, GSM8K, MATH-500.

**3. Set your Slurm account.** Edit `#SBATCH --account=` in `train.slurm` /
`eval.slurm` to your group: `rrg-zhijing` (RRG 2026, rapi `hqw-052-ab`) or
`def-zhijing` for the default allocation. Confirm with `sshare -U $USER` or
`sacctmgr show user $USER`.

**4. Submit (from a login node)**
```bash
# baseline and if-prune, separate jobs (run several seeds for significance)
sbatch --export=ALL,REGIME=baseline,RUN_NAME=qwen3_4b_base,SEED=42  experiments/cluster/train.slurm
sbatch --export=ALL,REGIME=if_prune,RUN_NAME=qwen3_4b_prune,SEED=42 experiments/cluster/train.slurm

# extra flags pass through EXTRA_ARGS, e.g. an anti-IF ablation:
sbatch --export=ALL,REGIME=if_prune,RUN_NAME=qwen3_4b_antiif,EXTRA_ARGS="--selection anti-if" \
       experiments/cluster/train.slurm
```
The job requests **2 H100s** by default and puts the verifier on `cuda:1` so it
doesn't compete with the policy + vLLM for VRAM on `cuda:0`. For a single GPU
use `--gpus-per-node=h100:1` (verifier colocates) and lower
`--vllm-gpu-memory-utilization`. On **Narval** switch to `--gpus-per-node=a100:N`
and the Narval account.

**5. Evaluate**
```bash
sbatch --export=ALL,RUN_NAME=qwen3_4b_base,STEP=latest  experiments/cluster/eval.slurm
sbatch --export=ALL,RUN_NAME=qwen3_4b_prune,STEP=latest experiments/cluster/eval.slurm
```
Writes `outputs/<run>/eval/eval_step<N>.json` with per-benchmark + per-domain
accuracy. Compare baseline vs if_prune across seeds.

**Monitoring:** `squeue -u $USER`, `sacct -j <jobid>`, tail `slurm-*.out`.
**W&B** runs in `offline` mode on compute nodes — `wandb sync wandb/offline-run-*`
from a login node afterward.

## Influence (CG) — how it works

All CG variants solve `(F+λI)hⱼ = g_testⱼ` with conjugate gradients, then **stream**
the train pool — compute each `g_trainᵢ`, dot it against the stacked `hⱼ`,
accumulate, discard — so we never hold all pool gradients at once. The per-train
score is `mean_j g_trainᵢ·hⱼ`. They differ only in the Fisher operator `F`:

- **`--if-method cg` (default; `influence_rlvr/fisher_fvp.py`)** — the **true
  policy Fisher** with the analytic per-token metric `M=diag(p)−ppᵀ`:
  `F = (1/N)Σ_seq Σ_t Jₜᵀ Mₜ Jₜ`. The over-vocabulary expectation is taken in
  closed form (no completion sampling for the Fisher), so it's **full-rank and
  low-variance**. `F·v` is **matrix-free** via a double-backward (reverse-mode
  only — works through SDPA/eager attention, no forward-mode AD). One forward +
  a few backwards over the small Fisher batch per CG iteration. This is the
  LLM-scale recipe the toy's FVP docstring points to.

- **`--if-method cg-empirical`** — the toy's exact route scaled up: cache
  per-completion score gradients `∇logπ(yᵤ|z)` from `G` sampled completions and
  feed them to `policy_fisher_fvp_from_grad_cache`. An *unbiased but low-rank*
  (`≤ N·G`) sampled Fisher; cheap, good for ablation/comparison.

- **`--if-method fisher`** — damped "outer-of-means" inverse (rank ≤ N), solved
  in closed form (Woodbury). Cheapest; for fast smoke runs.

Why two CG variants: with a small Fisher batch the *sampled* (`cg-empirical`)
Fisher is low-rank, so `(F+λI)⁻¹` barely differs from a damped gradient dot
(TracIn-like). The analytic per-token (`cg`) Fisher is full-rank and preconditions
properly — at the cost of forward/backward passes per CG iteration.

Memory (policy GPU): for `cg`, the Fisher batch forward (`cg_fisher_examples ×
cg_fisher_g` completions truncated to `cg_fisher_max_tokens`) + `H` (`n_if_target·D`)
+ one live gradient. For `cg-empirical`, the gradient cache
(`cg_fisher_examples·cg_fisher_g·D·4 B`) dominates. On A100-40G lower
`--cg-fisher-examples`, `--cg-fisher-g`, `--cg-fisher-max-tokens`, `--n-if-target`.

## Notes / known follow-ups

- **Extra benchmarks** the proposal lists (TheoremQA, OlympiadBench, AIME25,
  FinQA, LiveCodeBench, SWE-bench) are registered as `NotImplementedError` stubs
  in `data.py` with a note each. The verifier-scorable ones (TheoremQA, AIME25,
  FinQA, OlympiadBench) are small loaders; LiveCodeBench/SWE-bench need execution
  harnesses, not the verifier.
- **In-process two-phase if_prune** creates two TRL trainers in one process. If
  vLLM's engine lifecycle complains on a given cluster, split the phases into
  separate Slurm steps (train→influence→train) reusing the prune-step checkpoint.
```
