# influence-rlvr

Training data attribution for RLVR (GRPO). Uses influence functions adapted from SFT to RL; see `docs/derivation.pdf`.

## Setup

```bash
uv sync          # Python 3.10+, CUDA GPU recommended
```

## Run

```bash
# Edit RUN_NAME and config at the top of main_pipeline.py first
PYTHONUNBUFFERED=1 uv run python -u main_pipeline.py 2>&1 | tee outputs/run1/run.log
```

All artifacts land in a single folder: `outputs/<RUN_NAME>/`.

- `rlvr-output/` holds the training checkpoints and batch history.
- Analysis outputs are saved as `results1/`, `results2/`, `results3/`, ...
- If you rerun analysis with the same saved config, the pipeline warns before reusing that results folder. Press `n` to allocate a fresh numbered results directory instead.

Key settings in `main_pipeline.py`:

- `RUN_NAME` — folder name under `outputs/`
- `MAX_STEPS` — GRPO training steps
- `N_MATH` / `N_CODE` — train/test dataset sizes
- `INFLUENCE_MODE` — `"historical"` (default) or `"dense"`
- `SKIP_TRAINING` — reuse existing checkpoints
- `RESULTS_REUSE_POLICY` — `"ask"` (default), `"reuse"`, or `"new"`

## Analyze

```bash
uv run python -m analysis outputs/run1/results1/
```

## Structure

- `main_pipeline.py` — end-to-end script (train, replay, influence, save)
- `influence_rlvr/` — gradients, trajectory replay, attribution methods
- `analysis/` — schema, loader, analyzer, plots, CLI

