# influence-rlvr

Training data attribution for RLVR (GRPO). Uses influence functions adapted from SFT to RL; see `docs/derivation.pdf`.

## Setup

```bash
uv sync          # Python 3.10+, CUDA GPU recommended
# Optional vLLM backend (Linux/CUDA only)
uv sync --extra vllm
```

## Run

```bash
# Edit RUN_NAME and config at the top of main_pipeline.py first
PYTHONUNBUFFERED=1 uv run python -u main_pipeline.py 2>&1 | tee outputs/run1/run.log
```

### Standalone GRPO + eval statistics (`training_script.py`)

From the repo root (with `uv run` or your env that has TRL/vLLM as needed):

- Writes `run_config.json`, `eval_baseline.json`, and `eval_after_train.json` under `--output-dir` (eval JSON includes `metadata`: seed, model, LoRA).
- Use `--seed`, `--lora-r`, `--lora-alpha`, and comma-separated `--lora-target-modules` for reproducibility and capacity.
- See `python training_script.py --help` epilog for a multi-seed shell loop and suggested hyperparameters.

After one or more runs, compare accuracy and paired significance (Wilson CIs, bootstrap on mean paired delta, McNemar):

```bash
uv run python scripts/compare_gsm8k_eval.py --run-dir outputs/your_run/rlvr-output
uv run python scripts/compare_gsm8k_eval.py --multi-run 'outputs/nemotron_math_s*/rlvr-output'
```

All artifacts land in a single folder: `outputs/<RUN_NAME>/`.

- `rlvr-output/` holds the training checkpoints and batch history.
- Analysis outputs are saved as `results1/`, `results2/`, `results3/`, ...
- If you rerun analysis with the same saved config, the pipeline warns before reusing that results folder. Press `n` to allocate a fresh numbered results directory instead.

Key settings in `main_pipeline.py`:

- `RUN_NAME` — folder name under `outputs/`
- `MODEL_ID` — default base model, now `Qwen/Qwen2.5-Math-1.5B`
- `MAX_STEPS` — GRPO training steps
- `N_MATH` / `N_CODE` — train/test dataset sizes
- `EXPERIMENT_MODE` — `"math_grpo"`, `"code_grpo"`, or `"base_eval"`
- `INFLUENCE_MODE` — `"historical"` (default) or `"dense"`
- `SKIP_TRAINING` — reuse existing checkpoints
- `RESULTS_REUSE_POLICY` — `"ask"` (default), `"reuse"`, or `"new"`
- `GENERATION_BACKEND` — `GenerationBackend.HF` (default) or `GenerationBackend.VLLM`
- `VLLM_CONFIG` — runtime settings for replay/eval vLLM plus optional `training_use_vllm`
- `CODE_EVAL_NUM_SAMPLES` / `CODE_EVAL_TEMPERATURE` / `CODE_EVAL_TOP_P` — sampled code-eval settings for pass@k-style benchmarking
- `REPLAY_GRADIENT_CONFIG.max_new_tokens` / `.temperature` / `.top_p` — replay rollout sampling settings used for `g_train` / `g_test`

Math training now follows a DeepSeek-R1-style prompt/reward setup:

- no math system prompt
- reasoning is requested inside `<think></think>`
- the final answer is requested inside `\boxed{}`
- held-out math verification uses the same boxed-answer parser as training

## Generalization Study

The intended comparison is:

1. `EXPERIMENT_MODE="base_eval"` to measure checkpoint-0 math/code behavior.
2. `EXPERIMENT_MODE="math_grpo"` to train only on GSM8K math prompts and evaluate held-out code.
3. `EXPERIMENT_MODE="code_grpo"` as a matched direct-code baseline under the same GRPO budget.

Evidence of positive math-to-code transfer is a held-out code gain in the `math_grpo` run relative to checkpoint 0, ideally with math also improving at the same time. The `code_grpo` run is the direct-code upper bound for that budget and helps separate cross-domain transfer from the effect of RL on the target domain itself.

## vLLM Backend

The repo now supports a phased vLLM path for replay gradients, held-out eval, smoke runs, and the probe scripts.

- HF scoring remains the source of truth for log-probs, KL, and gradients.
- vLLM is used only for sampling/generation when `GENERATION_BACKEND=GenerationBackend.VLLM`.
- Phase 1 training stays on HF/TRL unless you also set `VLLM_CONFIG.training_use_vllm=True` and run on a machine that satisfies TRL/vLLM training requirements.

Smoke test:

```bash
uv run python main_pipeline_smoke.py
```

Math probe with HF:

```bash
uv run python scripts/probe_max_new_tokens.py \
  --run-dir outputs/run6 --checkpoint-step 200 \
  --backend hf
```

Math probe with vLLM:

```bash
uv run python scripts/probe_max_new_tokens.py \
  --run-dir outputs/run6 --checkpoint-step 200 \
  --backend vllm
```

Code probe with vLLM:

```bash
uv run python scripts/probe_code_max_new_tokens.py \
  --run-dir outputs/run6 --checkpoint-step 200 \
  --backend vllm
```

## Analyze

```bash
uv run python -m analysis outputs/run1/results1/
```

## Structure

- `main_pipeline.py` — end-to-end script (train, replay, influence, save)
- `influence_rlvr/` — gradients, trajectory replay, attribution methods
- `analysis/` — schema, loader, analyzer, plots, CLI

