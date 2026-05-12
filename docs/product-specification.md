# Project Flow

Author: Alexandru Neagu

This repository supports a master's thesis comparing two ways of learning chaotic 3-body gravitational motion:

- **EGNN:** E(n)-equivariant graph message passing with direct next-state prediction.
- **HGNN:** graph-based Hamiltonian dynamics with a learned decomposed Hamiltonian.

The code follows one main path: generate the dataset, train both models under the same conditions, evaluate them on the same test set, and produce the figures, tables, and animations used in the presentation and thesis.

## Data

The dataset used for the final experiments is grouped by how close the three bodies get during each trajectory. This lets us compare easy, medium, and hard motion regimes instead of reporting one average over everything. The config is:

```bash
uv run python -m data.generate --config configs/data.yaml
```

It writes:

```text
data/output/train.h5
data/output/val.h5
data/output/test.h5
```

The split is `1000 / 600 / 600`, with balanced groups:

| Bin | Distance interval |
| --- | --- |
| `close` | `[0.00, 0.02)` |
| `near` | `[0.02, 0.05)` |
| `mid` | `[0.05, 0.15)` |
| `wide` | `[0.15, 0.50)` |
| `far` | `[0.50, +inf)` |

Every HDF5 file includes trajectories, energies, metadata, and the group label for each trajectory. Training uses `train.h5` and `val.h5`; final comparisons use `test.h5`.

## Training

The training configs are:

```text
configs/egnn.yaml
configs/hgnn.yaml
```

Both use:

- `data/output/train.h5`
- `data/output/val.h5`
- curriculum horizons `[1, 5, 10, 20, 50, 100, 150, 199]`
- validation checked separately for every trajectory group

During training, validation is checked group by group. The code scores `close`, `near`, `mid`, `wide`, and `far` trajectories separately, then combines those scores when deciding which checkpoint is best. This keeps the selected model from looking good only because it performs well on the easier cases.

Run locally:

```bash
uv run python -m training.train --config configs/egnn.yaml
uv run python -m training.train --config configs/hgnn.yaml
```

The configs write directly to:

```text
runs/egnn/<run_id>/
runs/hgnn/<run_id>/
```

Each run contains:

```text
best.pt
latest.pt
metrics.csv
diagnostics.log
```

For Colab training, use:

```text
colab/train_colab.ipynb
```

The notebook expects the dataset under:

```text
MyDrive/masters-thesis/data/output/train.h5
MyDrive/masters-thesis/data/output/val.h5
MyDrive/masters-thesis/data/output/test.h5
```

and writes model runs to:

```text
MyDrive/masters-thesis/runs/<model>/<run_id>/
```

## Evaluation

Evaluate both trained checkpoints on the same grouped test set:

```bash
uv run python -m evaluation.evaluate \
  --config configs/egnn.yaml \
  --checkpoint runs/egnn/<run_id>/best.pt \
  --test-path data/output/test.h5 \
  --device auto

uv run python -m evaluation.evaluate \
  --config configs/hgnn.yaml \
  --checkpoint runs/hgnn/<run_id>/best.pt \
  --test-path data/output/test.h5 \
  --device auto
```

For checkpoints under `runs/`, evaluation writes next to the checkpoint:

```text
runs/<model>/<run_id>/evaluation/metrics.json
runs/<model>/<run_id>/evaluation/summary.csv
```

Evaluate the constant-velocity baseline:

```bash
uv run python -m evaluation.evaluate_baseline \
  --baseline constant_velocity \
  --train-path data/output/train.h5 \
  --test-path data/output/test.h5 \
  --output-dir runs/baselines/constant_velocity/evaluation \
  --device auto
```

`metrics.json` is the detailed evaluation output. The report generator reads it to build the final tables and figures.

## Reports and Figures

Generate the comparison report after the EGNN, HGNN, and constant-velocity evaluations exist:

```bash
uv run python -m evaluation.report \
  --egnn runs/egnn/<egnn_run_id>/evaluation/metrics.json \
  --hgnn runs/hgnn/<hgnn_run_id>/evaluation/metrics.json \
  --baseline runs/baselines/constant_velocity/evaluation/metrics.json \
  --output runs/reports/official_1k
```

The report directory contains:

```text
runs/reports/official_1k/
  report.md
  tables/
    per_bin_summary.csv
    key_timestep_summary.csv
  figures/
    01_rollout_position_mse_by_bin.{png,pdf}
    02_energy_drift_by_bin.{png,pdf}
    03_h1_by_bin.{png,pdf}
    ...
```

The main number shown in plots is **position MSE**, because it maps directly to visible trajectory error. State and velocity MSE are still saved in `metrics.json` and the CSV files for deeper analysis.

## Chunked Forecasting

Chunked forecasting asks a practical question: if the model is periodically corrected with true observations, how large can the prediction window be?

```bash
uv run python -m evaluation.evaluate_chunked \
  --egnn-checkpoint runs/egnn/<egnn_run_id>/best.pt \
  --hgnn-checkpoint runs/hgnn/<hgnn_run_id>/best.pt \
  --egnn-config configs/egnn.yaml \
  --hgnn-config configs/hgnn.yaml \
  --train-path data/output/train.h5 \
  --test-path data/output/test.h5 \
  --output-dir runs/reports/official_1k/chunked \
  --chunks 1 3 5 10 25 \
  --device auto
```

It writes:

```text
chunked_summary.csv
chunked_endpoints.csv
chunked_report.md
chunked_endpoint_position_mse_by_bin.{png,pdf}
```

The output includes a usable-K table. A chunk size is marked as usable when median endpoint position RMSE is at most `0.5` coordinate units.

## Animations

Animations are generated from checkpoints and the test set. By default, the selector picks one representative trajectory per bin. For presentation use, pass a manual YAML selection:

```yaml
close: 291
near: 389
mid: 247
wide: 425
far: 218
```

Run:

```bash
uv run python -m evaluation.animate_best \
  --egnn-checkpoint runs/egnn/<egnn_run_id>/best.pt \
  --hgnn-checkpoint runs/hgnn/<hgnn_run_id>/best.pt \
  --egnn-config configs/egnn.yaml \
  --hgnn-config configs/hgnn.yaml \
  --test-path data/output/test.h5 \
  --output-dir runs/reports/official_1k/animations \
  --selection-file runs/reports/official_1k/selections/manual.yaml \
  --device auto \
  --fps 20
```

Use `evaluation/visual_diagnostics.ipynb` to inspect trajectories and maintain the manual selection file.

## Colab Evaluation Notebook

For expensive evaluation/report/animation work, use:

```text
colab/evaluate_report_artifacts.ipynb
```

It can:

1. verify/copy the dataset from Drive,
2. evaluate EGNN and HGNN,
3. evaluate the constant-velocity baseline,
4. generate the report,
5. optionally run chunked forecasting,
6. optionally render animations.

## Validation

Run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Passing tests means the main code paths still behave as expected. Final results should still reference the saved run folders, `metrics.json`, generated reports, and exact configs used.
