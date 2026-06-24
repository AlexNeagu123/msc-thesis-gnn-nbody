[![CI](https://github.com/AlexNeagu123/msc-thesis-gnn-nbody/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/AlexNeagu123/msc-thesis-gnn-nbody/actions/workflows/ci.yaml?query=branch%3Amain)

# msc-thesis-gnn-nbody

Implementation repository for the thesis **Learning Three-Body Dynamics with Graph Neural Networks**.

The code compares two graph neural-network models on a shared gravitational three-body benchmark:

- **EGNN**, an E(n)-Equivariant Graph Neural Network adapted as an autoregressive next-state predictor.
- **HGNN**, a Hamiltonian Graph Neural Network that learns a decomposed Hamiltonian and advances the state with a velocity-Verlet update.

The repository contains the full workflow used in the thesis: dataset generation, model implementation, training, evaluation, report generation, trajectory animations, and symbolic-regression analysis.

## Repository Map

| Thesis component | Location | Purpose |
|---|---|---|
| Dataset generation | `configs/data.yaml`, `data/` | Generate the 2D three-body datasets with REBOUND/IAS15 and assign trajectory classes by minimum pairwise distance. |
| Deterministic baselines | `models/baselines.py` | Persistence, constant-velocity, mean-velocity, and mean-position reference predictors. |
| EGNN model | `models/egnn.py` | E(n)-equivariant message-passing model adapted to one-step state prediction. |
| HGNN model | `models/hgnn.py` | Hamiltonian graph model with kinetic and potential branches. |
| Training framework | `configs/egnn.yaml`, `configs/hgnn.yaml`, `training/` | Curriculum rollout training, validation scoring, checkpointing, and training diagnostics. |
| Evaluation protocol | `evaluation/evaluate.py`, `evaluation/evaluate_baseline.py`, `evaluation/metrics.py` | Autoregressive rollout evaluation, position MSE, relative energy drift, and finite-trajectory diagnostics. |
| Result reports | `evaluation/report.py`, `evaluation/report_figures.py`, `evaluation/report_tables.py` | Figures and tables used for the three-body comparison. |
| Zero-shot generalization | `data/generate_eval_set.py`, `evaluation/report_generalization.py` | Additional `N=2`, `N=4`, and `N=5` test datasets and summary reports. |
| Trajectory animations | `evaluation/animate_best.py`, `evaluation/visual_diagnostics.ipynb` | Representative predicted trajectories compared with ground truth. |
| Symbolic regression | `interpretability/` | HGNN kinetic and potential mappings sampled and analyzed with PySR. |
| Colab execution | `colab/` | Notebooks used for remote training and artifact generation. |
| Project notes | `docs/` | Detailed project flow and repository architecture notes. |

Generated datasets and experiment outputs are intentionally ignored by git. The main generated locations are:

```text
data/output/
runs/
```

## Dataset

The main benchmark consists of 2D three-body trajectories generated through mutual gravitational interaction. The canonical generation command is:

```bash
uv run python -m data.generate --config configs/data.yaml
```

It writes:

```text
data/output/train.h5
data/output/val.h5
data/output/test.h5
```

The split sizes are:

| Split | Trajectories |
|---|---:|
| Train | 1000 |
| Validation | 600 |
| Test | 600 |

Each trajectory contains 200 sampled states. Each body state is stored as:

```text
[x, y, vx, vy, m]
```

Validation and test trajectories are grouped into five trajectory classes using the minimum pairwise Euclidean distance reached during the trajectory:

| Trajectory class | Criterion |
|---|---:|
| `close` | `0.00 <= d_min < 0.02` |
| `near` | `0.02 <= d_min < 0.05` |
| `mid` | `0.05 <= d_min < 0.15` |
| `wide` | `0.15 <= d_min < 0.50` |
| `far` | `0.50 <= d_min` |

The three-body datasets used in the thesis are publicly available on Hugging Face:

```text
https://huggingface.co/datasets/AlexNeagu123/three-body-trajectory-benchmark
```

## Training

The two main training configurations are:

```text
configs/egnn.yaml
configs/hgnn.yaml
```

Run training with:

```bash
uv run python -m training.train --config configs/egnn.yaml
uv run python -m training.train --config configs/hgnn.yaml
```

Training uses curriculum rollout horizons, a multi-step prediction loss, and validation-score checkpointing. The selected checkpoint is the one with the best rollout-based validation score across trajectory classes.

Each training run writes artifacts under:

```text
runs/egnn/<run_id>/
runs/hgnn/<run_id>/
```

Typical run contents:

```text
best.pt
latest.pt
metrics.csv
diagnostics.log
```

## Evaluation

Evaluate a trained checkpoint with:

```bash
uv run python -m evaluation.evaluate \
  --config configs/egnn.yaml \
  --checkpoint runs/egnn/<run_id>/best.pt \
  --test-path data/output/test.h5
```

Evaluate the constant-velocity baseline with:

```bash
uv run python -m evaluation.evaluate_baseline \
  --baseline constant_velocity \
  --test-path data/output/test.h5
```

Generate the main comparison report with:

```bash
uv run python -m evaluation.report \
  --egnn runs/egnn/<egnn_run_id>/evaluation/metrics.json \
  --hgnn runs/hgnn/<hgnn_run_id>/evaluation/metrics.json \
  --baseline runs/baselines/constant_velocity/evaluation/metrics.json \
  --output runs/reports/official_1k
```

The report writes:

```text
runs/reports/official_1k/report.md
runs/reports/official_1k/figures/
runs/reports/official_1k/tables/
```

## Symbolic Regression

The interpretability analysis focuses on HGNN. The code samples the learned kinetic and potential components and applies symbolic regression to the resulting mappings.

Relevant files:

```text
interpretability/analysis.py
interpretability/probes.py
interpretability/symbolic.py
interpretability/interpretability.ipynb
```

## Development Checks

The GitHub workflow runs the same checks locally available here:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

Notebooks are kept as execution artifacts and are excluded from Ruff linting and formatting. Tested source code lives in the Python modules.

## Notes

- `runs/` and `data/output/` are ignored because they contain generated artifacts.
- `evaluation/evaluate_chunked.py` is supplementary tooling for short-horizon corrected forecasts. It is not central to the thesis narrative, but it is tested and integrated into the report generator when chunked artifacts are present.
- The top-level workflow and module responsibilities are described in more detail in `docs/product-specification.md` and `docs/repository-architecture.md`.
