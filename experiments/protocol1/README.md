# Protocol 1 [PREDICTION] — LDS at LLM scale

**Question.** Which training prompt would contribute most to target reward if we
*continue* training the current model on a distribution where it is upweighted?
Influence is computed **at a single checkpoint** (no trajectory sum — that is
Protocol 2). **Measurement = LDS**: retrain M continuation models from that
checkpoint on random α-subsets of the pool and Spearman-correlate the
influence-predicted target reward with the measured one.

**Method comparison (the headline).** Three estimators on the same rollout
GRPO gradients — the preconditioner ladder from the thesis: **dot** (identity),
**tracin-adam** (Adam diagonal from optimizer.pt), **ekfac** (Kronecker-factored
eigenbasis inverse of the policy Fisher, fitted at the checkpoint). Same g_test /
g_train everywhere; only the operator differs, so LDS gaps are attributable to
the curvature model. **Cosine is the default for all three** (the preconditioned
target tangent h and each g_train are unit-normalized — direction, not
magnitude), applied uniformly so the comparison stays fair; `--no-cosine` gives
the raw magnitude-carrying dot as an ablation (rollout magnitude ≈ learnability).

**Checkpoint sweep.** The whole procedure is repeated at each checkpoint of one
reference training run (default steps 50, 100, 150, 200): score IF at step c,
retrain the M subsets from step c, LDS(c). The same `subsets.npy` is reused at
every step, so subsets are comparable across c. (Scoring *exactly at step 0* is
possible for dot but degenerate: fresh zero-init LoRA has `dL/dA = 0`
everywhere, and there is no optimizer.pt so tracin-adam falls back to dot.)

**Horizon sweep (prediction drift).** Each subset run trains once to the max
continuation length (default 200 steps) and measures target reward *in-training*
at every `--eval-at` horizon (default 50, 100, 150, 200) — so one run yields
LDS(c, k) for every k: how far ahead the single-checkpoint IF prediction stays
good before it drifts. Evals are batched HF generation inside a trainer callback
(~5–8 min per horizon) and written incrementally, so a preempted run keeps its
completed horizons.

Spec (updates deck, p.18): 1.5B + LoRA, GRPO g=8, pool ~1–2k prompts,
M = 30–40 subsets at α = 0.5, continuation ≈ 200–300 steps, 20–50 target prompts.

## Time budget (GH200, ~0.5–1 min/train step at 32 prompts/step)

| stage | per unit | units | GPU-h | wall @ 12 GPUs | wall @ 8 GPUs |
|---|---|---|---|---|---|
| ref run (200 steps) | 2–3.5 h | 1 | 2–3.5 | 2–3.5 h (serial, first) | same |
| scoring rollout×{dot, tracin-adam, ekfac} | 1–2.5 h | 6 (2 ckpts × 3) | 7–15 | ~1–2 h (parallel) | ~1–3 h |
| subset retrains (200 steps + 5 horizon evals) | 2–4 h | 200 (2 ckpts × M=100) | 400–800 | 33–67 h | 50–100 h |
| **total** | | | **≈ 410–820** | **≈ 1.5–2.9 days** | **≈ 2.2–4.3 days** |

The plan is 2 ckpts {100, 200} × **M=100** (per-cell LDS noise ±0.10, matching
the diffusion-IF paper's subset counts) — it fits 2–3 days at **~12 concurrent
GPUs**; at 8 it can stretch past 3 days at the pessimistic per-step estimate.
Levers: `--array=0-63` first (±0.13) and extend with `--array=64-99` mid-flight
once the queue's pace is known — runs are idempotent and `lds.py` reports
whatever exists.

## Allocation & resumability (12 h max walltime)

Parallelism is **job-level**: every training job (ref + each subset task) is
single-GPU by design — the 128 array tasks are the multi-GPU usage. Scoring is
the exception: `score.slurm` torchrun-shards the pool across however many GPUs
the job gets (`--gres=gpu:4` on one node ≈ 4× faster scan; the EK-FAC factor
fit is per-rank redundant, so only the scan part scales).

Every job type fits inside 12 h (`--time` applies **per array task**, so the
array outlives it). If anything is preempted or you run out of allocation,
**resubmit the same command** — nothing is lost beyond the interrupted unit:

| job | fits 12 h? | on resubmit |
|---|---|---|
| ref run | 2–3.5 h ✓ | auto-resumes from the latest full checkpoint (adapter + optimizer + step restored); finished run = no-op |
| scoring | 1–2.5 h ✓ | re-runs that one (checkpoint, method) job from scratch |
| subset task | 2–4 h ✓ | fully-measured task exits instantly; partial task restarts from the ref checkpoint (≤4 h lost; set `SAVE_STEPS=50` to make tasks mid-run-resumable at ~0.5 GB/ckpt/run) |
| `lds.py` | seconds | reads whatever horizons/steps/variants exist |

## Setup

| knob | default | where |
|---|---|---|
| model | Qwen2.5-1.5B-Instruct + LoRA r=32 (all proj) | `train.py` |
| data | GSM8K; pool = 1000 train prompts (liveness-first from protocol2's pilot bands), targets = 32 held-out test prompts (in-band first) | `make_pool.py` |
| GRPO | g=8, temp 1.0, top_p 1.0, β=0.04, lr 1e-5 const, adam_β2 0.99, 32 prompts/step (16×16/8) | `train.py` |
| reference run | 200 steps on the full pool (≈6 epochs), checkpoint + optimizer.pt every 50 — the sweep grid | `ref.slurm` |
| methods | rollout grads × {dot, tracin-adam, ekfac}, cosine ON for all; ekfac fits on 64 prompts × 4 completions ≤256 tok, per-block rel damping 0.1 | `score_ref.py` |
| subsets | M=100 (α=0.5 → k=500, fixed size, so Σ-IF ≡ visit-weighted predictor), shared across checkpoints; the `--array` range picks M — noise ±0.10 at 100, ±0.13 at 64 | `make_pool.py` |
| continuation | 200 steps max from checkpoint-c's adapter, fresh optimizer (PBRF: a new proximal phase); ≈ 13 epochs over the subset at k=200 | `subsets.slurm` |
| horizons | evals at k = 1, 50, 100, 150, 200 (final always included) via `--eval-at`; k=1 is a null anchor — the step is warmup-damped (~0.1× LR), so expect LDS ≈ 0 there | `train.py` |
| measured y | per-target pass rate, 16 samples @ temp 1.0 (training temperature, same as the pilot band), fixed eval seed across runs and horizons | `eval_targets.py` |

## Pipeline

```bash
# 0 · local, no GPU — pool.jsonl + target.jsonl + subsets.npy (+ lds_meta.json)
python -m experiments.protocol1.make_pool

# 1 · reference run (GPU, ~2–3.5 h): full pool, 200 steps, ckpt+optimizer.pt every 50
REF_DIR=$SCRATCH/p1_runs/p1_ref sbatch ... experiments/protocol1/scripts/ref.slurm

# 2 · influence per (checkpoint, method) — 6 jobs, ~1–2.5 h each, all parallel;
#     each writes step-suffixed artifacts into its own subdir (cosine is default)
for C in 100 200; do
  for METH in dot tracin-adam ekfac; do
    REF_DIR=... REF_STEP=$C IF_GRAD=rollout IF_METHOD=$METH sbatch ... scripts/score.slurm
  done
done
# (optional near-free extras: IF_GRAD=gold IF_METHOD=dot — ~15 min/ckpt control;
#  COSINE=0 IF_METHOD=dot — raw magnitude-carrying dot ablation)

# 3 · the retraining sweep — THE dominant cost: 2 ckpts × M=100 runs, each 200
#     continuation steps (~1.7–3.3 h) + 5 horizon evals (~30 min) on a GH200
#     ≈ 400–800 GPU-h ≈ 1.5–2.9 days at 12 concurrent GPUs (2.2–4.3 at 8).
#     Unsure of the queue? Submit --array=0-63 now, --array=64-99 mid-flight.
for C in 100 200; do
  REF_DIR=... REF_STEP=$C RUNS_ROOT=$SCRATCH/p1_runs \
    sbatch --array=0-99%12 ... experiments/protocol1/scripts/subsets.slurm
done

# 4 · the report (local, numpy-only): one LDS row per (checkpoint, horizon,
#     variant), step-matched — IF@c is only correlated against retrains-from-c
python -m experiments.protocol1.lds --ref-dir <ref> --runs-root <runs>
```

## Output

`lds_report.json` + a table grouped by checkpoint: per (step, horizon k,
variant), the **pooled LDS** (Spearman over subsets of Σᵢ∈S score[i] vs mean
target reward) and the **per-target LDS** (TRAK-style, i.e. the LDS as defined
in the TRAK paper: one Spearman per target from the [T, N] matrix, mean ± std
over targets) — plus, per (step, k), a no-influence **difficulty baseline**
(subset mean base pass rate) that any variant must beat to be interesting.
Reading the table: fix k and scan steps → where in training IF is most
predictive; fix a step and scan k → how far ahead the prediction stays good
before drifting. On a synthetic drift fixture the per-target column tracks the
planted decay cleanly, while the pooled column is noticeably noisier at long
horizons with M≈32 — read the drift off the per-target column.

Context from the toy (memory): random-subset LDS in the delta-finetune regime has
a low ceiling (~0.03 there) with signal concentrated in the tails — so treat a
small-but-positive LDS above the baseline as signal, and follow up with
extreme-subset probes (top/bottom-k by score) if the bulk correlation is flat.

## Clariden (4× GH200/node)

No python on login nodes → build data locally, rsync it in; everything runs via
`--environment=pt` + the `$SCRATCH/envs/ifrlvr` venv (rebuild with
`experiments/cluster/clariden_rebuild.sh` if the scratch purge ate it). Use
`scripts/subsets_node.slurm` for the sweep — one array task = one node = 4
subset runs pinned to the 4 GPUs (`--array=0-24` covers M=100 per checkpoint);
give scoring jobs `--gres=gpu:4` (torchrun shards the pool 4-way). The full
command sequence is in the slurm headers; everything is idempotent, so recovery
from any preemption = resubmit the same commands.

## Files

- `make_pool.py` — stage 0 (draws on `../protocol2/dataset/data/*_scored.jsonl`)
- `train.py` — stages 1+3: reference run (`--save-steps 10`) and subset continuation
  (`--subset-id m --init-adapter <ref>/checkpoint-<c>`); ends with the target eval
- `eval_targets.py` — sampled pass-rate eval (in-process + standalone CLI)
- `score_ref.py` — stage 2: per-target influence matrix via the shared streaming
  scorer (`if_target_matrix_rows = T`)
- `lds.py` — stage 4: the report
- `scripts/{ref,score,subsets}.slurm` + `scripts/subsets_node.slurm` (Clariden
  node-packed: 4 subset runs per node)
