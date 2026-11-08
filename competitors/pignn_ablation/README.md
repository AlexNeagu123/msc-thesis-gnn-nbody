# PI-GNN: force formula and position update

I used this package to retrain four versions of PI-GNN on the shared dataset. Each version changes the force target, the position update during training, or both.

The model uses **TensorFlow/Keras** and has **34,690 parameters**. The four versions share the same network architecture.

Results are in [Section C of the handover](../../HANDOVER.md#c-pi-gnn-physics-informed-direct-acceleration-network).

## The four versions

[`recipes.py`](recipes.py) defines the variants as `CELLS`:

| Name | Additive softening `s` | Position update during training | Sheet row |
|---|---|---|---:|
| A00 | `0.05` | Explicit | 25 |
| A01 | `0.05` | Semi-implicit | 26 |
| A10 | `0.05**2` | Explicit | 27 |
| A11 | `0.05**2` | Semi-implicit | 28 |

**A00** follows the original training recipe. **A01** changes the position update. **A10** changes the gravity target to match the dataset generator. **A11** combines those changes.

Here, `s` appears in the gravity denominator as `(distance_squared + s)**1.5`.

## What the network learns

PI-GNN predicts acceleration. During training, the code converts that acceleration into a predicted next position and velocity:

```text
v_next = v + dt * predicted_acceleration

Explicit:       q_next = q + dt * v
Semi-implicit:  q_next = q + dt * v_next
```

All four versions use the semi-implicit update for evaluation rollouts. A01 and A11 also use it during training.

The training loss is:

```text
data loss    = MSE(predicted next [q, v], recorded next [q, v])
physics loss = MSE(predicted acceleration, analytical gravity)
total loss   = data loss + 0.1 * physics loss
```

The implementation is in [`StepEngine.loss_parts()`](../gnode_pignn_training/engine.py). The force target and position-update functions are in the same file.

## Where to look

- [`recipes.py`](recipes.py): the four settings and their differences.
- [`../gnode_pignn_tf/models.py`](../gnode_pignn_tf/models.py): the network layers.
- [`../gnode_pignn_training/engine.py`](../gnode_pignn_training/engine.py): loss, gradients and Adam updates.
- [`production.py`](production.py): the guarded training entry point for A10 and A01. It checks the recorded source files and runtime before starting or resuming.
- [`preflight.py`](preflight.py): a short TRAIN-only check, with 21 disposable updates per variant. It writes measurements rather than reusable weights.
- [`test_preparation.py`](test_preparation.py) and [`test_production_adapter.py`](test_production_adapter.py): recipe and runner checks.

The shared recipe uses seed 42, Adam at `5e-4`, batch size 512 and ordered one-step training pairs. A full 100-epoch run has 125,000 updates. Each model is initialized from scratch; the original published weights were used separately for reproduction checks.

## How this relates to the results

A00 and A01 completed their training runs. A10 and A11 stopped at epoch 48 after nonfinite validation results. Their handover scores refer to the best eligible checkpoints saved earlier.

The clipped A10 experiment in sheet row 29 uses the later [`ablation_controls`](../ablation_controls/README.md) code. Its setting is `pi_ordered_clip1`.

The run records, readiness files and checkpoints live under the local, Git-ignored `runs/` directory. The original readiness files refer to the code before the handover renaming; a new run needs fresh checks against the current files.

## Tests

From the repository root, in an environment with TensorFlow installed:

```bash
python -m pytest competitors/pignn_ablation -q
```

`test_preparation.py` skips when TensorFlow is unavailable. The runner-gate tests in `test_production_adapter.py` can run without it.
