# Evaluation

This folder contains evaluation code and interactive visual diagnostics.

- `evaluate.py`: official numeric report runner. It writes `metrics.json` and `summary.csv` under `results/evaluation/<model>/<run_id>/`.
- `metrics.py`: shared rollout, energy, and plotting helpers used by notebooks and the report runner.
- `egnn_visual_diagnostics.ipynb`: interactive EGNN rollout and plotting notebook.
- `hgnn_visual_diagnostics.ipynb`: interactive HGNN rollout and plotting notebook.
- `test_evaluate.py`, `test_metrics.py`: regression tests for evaluation behavior.
