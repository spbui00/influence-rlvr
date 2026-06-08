# Scale-up plan: GR-faithful GRPO + IF-pruning on Nibi (H100)

## 1. The diagnosis — it was SCALE, not knobs

The 1–2 GPU signal check stayed flat because our **batch was ~128× smaller than
the recipe that works.** Table 9 (General-Reasoner-4B, our exact model+data):
`train_batch_size = 1024 prompts/step`. Ours: **8**. Everything we tuned
(grading, completion length, the empty-box hack, the prompt) was real but
secondary. The base model bootstraps coherence from the reward *given enough
batch*; at batch 8 there isn't enough gradient signal per step to escape the
incoherent-base-model regime. **The fix is batch.**

## 2. Target = General-Reasoner-4B (Table 9), and our gaps

| param | GR-4B | ours now | action |
|---|---|---|---|
| backbone | Qwen3-4B-Base | ✓ | keep |
| **train_batch_size** | **1024 prompts/step** | **8** | **scale via DP — the main fix** |
| rollout_n / g_train | 8 | 8 | keep |
| max_prompt_length | 1024 | 1024 | keep |
| **max_response_length** | **4096** | 2048 | raise to 4096 |
| learning_rate | 5e-7 (full-param) | 1e-5 (LoRA) | keep ~1e-5 (LoRA LR ≠ full-param; 5e-7 is too low for LoRA) |
| param mode | full-param | LoRA | **keep LoRA** (see §3) |
| clip | low 0.2 / high 0.3 | symmetric 0.2 | add `epsilon_high=0.3` (clip-higher) |
| KL | kl_loss_coef 1e-4, low_var_kl | none (beta=0) | optional: small KL 1e-4 + ref model |
| temperature | 0.7 | 1.0 | set 0.7 |
| n_gpu | 32 (4×8 H100) | 1–2 | 8–16 H100 (1–2 Nibi nodes) |
| framework | verl | TRL | keep TRL (our influence code lives here) |

## 3. The one deviation we KEEP: LoRA (non-negotiable for the influence study)

GR uses full-param. We **cannot**, because IF-pruning needs *per-example gradients*:
- LoRA → per-example grad is the ~132 MB adapter → tractable to store/score.
- full-param → per-example grad is **4B params (~30× bigger)** → influence scoring
  blows up in memory and compute.

So the influence-pruning experiment **requires LoRA**. We match GR on everything
else and close the batch gap; LoRA (with a higher rank, r=32–64, for capacity) is
the justified deviation, and it's consistent across all arms so the if-guided-vs-
anti comparison stays valid.

## 4. Nibi resources (see nibi-cluster-notes)

- Account **`def-rgrosse`**; full H100 nodes `g[1-29] = gpu:h100:8` (8× **H100 80GB**,
  112 CPU, 2 TB RAM). `--gres=gpu:h100:N`. Avoid MIG slices `g[30-37]`.
- `def-` = default/rapid-access → queue-bound, realistically **1–2 nodes at a time**.
  Reproducing GR's full 32-H100 / batch-1024 is unlikely; we target a feasible fraction.

## 5. Multi-GPU DP for batch — architecture + math

**Architecture (1 H100 node = 8 GPU):** vLLM **server** mode (shared engines), not
per-process colocate:
- ~6 GPU: data-parallel TRL training (accelerate + DeepSpeed ZeRO-2)
- 1 GPU: policy generation vLLM server (shared by the DP workers)
- 1 GPU: verifier vLLM server (already built)

**Batch math (TRL):** `prompts_per_step = num_processes × grad_accum × (per_device_batch / g_train)`.
With 6 DP workers, `per_device_batch=16`, `g_train=8` (→2 prompts/device/microstep):
- `grad_accum=11` → **132 prompts/step** (16× our 8; ~1/8 of GR)
- `grad_accum=22` → **264 prompts/step** (~1/4 of GR)

So **batch 128–256 prompts/step on one H100 node** — the regime where bootstrap should work.

## 6. HP alignment (config/GRPOConfig changes)

- `max_completion_length 4096`, `max_prompt_length 1024`
- `temperature 0.7`
- clip-higher: `epsilon=0.2`, `epsilon_high=0.3` (TRL supports it)
- LoRA `r=32–64` (more capacity toward full-param expressivity)
- LR ~1e-5 (LoRA-appropriate; do NOT drop to 5e-7)
- optional KL: `beta=1e-4` (enables ref model) — minor; can start at 0

## 7. Two tiers

**Tier 1 — prove the bootstrap (1 H100 node, baseline only).** batch 128–256, LoRA r32,
GR HPs, ~300 steps. **GO/NO-GO gate:** `box_rate` and `acc` climb (watch the live
`[reward]` breakdown), and eval moves **≥15 pts** off the base. This finally answers
"does GR-regime training make Qwen3-4B-Base learn?" at a feasible scale. If NO here,
the testbed is wrong (not the infra).

**Tier 2 — the IF-pruning grid (only if Tier 1 passes).** arms {baseline, ifg_cg,
ifg_dot, ifg_tadam, anti} × 2–3 seeds, batch 128–256, pool 1000–2000 with distractor
data, static (one-shot) prune to start. Needs the influence sharding + server infra.

## 8. Compute & time (estimates, need cluster validation)

Per-step at batch 256 on 1 H100 node (LoRA, resp 4096, vLLM gen + verifier):
~90–150 s/step → **~300 steps ≈ 8–13 h/run** (may exceed a 12 h job → checkpoint+resume
or 2 nodes). Tier-2 grid (9–15 runs) is **queue-bound on a def- account → days–a week**.
H100 is ~2–3× an L40S, so each run is meaningfully faster than the Killarney estimates.

## 9. Infra task list (the real work) + sequencing

1. **Nibi cluster scripts** — `env.sh` (def-rgrosse, module spider), slurm with
   `--gres=gpu:h100:N`, accelerate/DeepSpeed launch. *(small)*
2. **Multi-GPU DP training** — accelerate + ZeRO-2 + vLLM **server** generation;
   verify batch scales and LoRA weights sync to the gen server. *(medium — the crux)*
3. **HP alignment** — add `epsilon_high`, `temperature`, resp 4096, LoRA rank to
   config/GRPOConfig. *(small)*
4. **Tier-1 run + gate.** *(1 node, ~1 day)*
5. **Sharded influence** — wire `dist_utils` into in-loop `compute_pool_influence`
   (still not integrated); + distractor pool. *(medium)*
6. **Tier-2 grid.** *(queue-bound)*

**Recommendation:** do 1→2→3→4 first (prove the bootstrap at batch 128–256 on one
H100 node). Everything else (sharded influence, the grid) is wasted effort until
Tier-1 confirms GR-regime training actually moves Qwen3-4B-Base. The batch is the
hypothesis; test it cheaply before building the grid.
