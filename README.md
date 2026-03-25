# influence-rlvr

A training data attribution study for RLVR trained using [GRPO](https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300). We do TDA using the [influence function](https://arxiv.org/abs/2308.03296). Since IFs have been mostly studied on SFT, we needed to derive a formula for RLVR, read in `docs/derivation.pdf`.

## Setup

Requires Python 3.10+ and a CUDA GPU.

```bash
uv sync
```

## Run

```bash
PYTHONUNBUFFERED=1 uv run python -u main_pipeline.py 2>&1 | tee run.log
```

Edit the configuration block at the top of `main_pipeline.py` before launching. Key settings:

- `MAX_STEPS` — GRPO training steps
- `N_MATH` / `N_CODE` — train/test dataset sizes
- `N_TRAIN_REPLAY` — how many train samples to replay per checkpoint (set to `N_MATH` for full coverage)
- `INFLUENCE_MODE` — `"historical"` (batch-aware, default) or `"dense"` (counterfactual)
- `SKIP_TRAINING` — set `True` to reuse existing checkpoints

## Analyze results

```bash
uv run python -m analysis results/
```

Or use `analysis.InfluenceAnalyzer` directly:

```python
from analysis import InfluenceAnalyzer
analyzer = InfluenceAnalyzer.from_directory("results/")
analyzer.write_default_artifacts()
```

## Project structure

- `main_pipeline.py` — end-to-end script (train, replay, influence, save)
- `influence_rlvr/` — gradient computation, trajectory replay, attribution methods
- `analysis/` — schema, loader, analyzer, plots, CLI

