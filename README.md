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

All artifacts land in a single folder: `outputs/<RUN_NAME>/` containing `rlvr-output/` (checkpoints) and `results/` (influence matrices, figures).

Key settings in `main_pipeline.py`:

- `RUN_NAME` — folder name under `outputs/`
- `MAX_STEPS` — GRPO training steps
- `N_MATH` / `N_CODE` — train/test dataset sizes
- `INFLUENCE_MODE` — `"historical"` (default) or `"dense"`
- `SKIP_TRAINING` — reuse existing checkpoints

## Analyze

```bash
uv run python -m analysis outputs/run1/results/
```

## Structure

- `main_pipeline.py` — end-to-end script (train, replay, influence, save)
- `influence_rlvr/` — gradients, trajectory replay, attribution methods
- `analysis/` — schema, loader, analyzer, plots, CLI

