# Model and training controls

I added this package to try specific changes to HGNN, GNODE and PI-GNN while keeping the surrounding training setup fixed where possible.

The folder contains model, loss, sampling and checkpoint components. The experiment results and their settings are listed in the [handover](../../HANDOVER.md).

## What's here

| File | Purpose |
|---|---|
| [`recipes.py`](recipes.py) | Names and settings for 13 combinations across five groups |
| [`hgnn.py`](hgnn.py) | PyTorch HGNN with learned or exact kinetic energy, plus its trajectory and gravity losses |
| [`tf_adapters.py`](tf_adapters.py) | TensorFlow GNODE updates, PI-GNN ordering and clipping, and a central-force network |
| [`schedule.py`](schedule.py) | Batch ordering and a saved cursor for resuming |
| [`torch_checkpoint.py`](torch_checkpoint.py) | Saving and restoring model, optimizer, progress and random-number state |
| [`prepare.py`](prepare.py) | Writes the proposed experiment settings and matched-tail batch order |

## HGNN: kinetic energy and gravity loss

I compared a learned kinetic-energy function with the exact expression:

```text
T = sum(p² / (2m))
p = m * v
```

`KineticAblationHGNN(kinetic="learned")` keeps the learned branch. `kinetic="exact"` replaces it with the expression above. Both retain the learned potential-energy network.

The `objective()` function computes the multi-step trajectory loss. Its `gravity` argument selects:

- `"off"`: trajectory loss alone.
- `"scaled"`: trajectory loss plus `0.1 × acceleration MSE / 882.8257392887597`.

The acceleration comparison is made at the starting state of each training example. The denominator is the TRAIN acceleration mean square; the gravity target uses additive softening `0.05**2`.

The exact-kinetic experiments appear in **sheet rows 14–17**, with seeds 42 and 43. Replacing the learned kinetic branch also reduces the parameter count, from 191,040 to 182,400.

## GNODE: position update

`GNODEEngine` provides two settings:

- `gnode_native`: the original GNODE update, with the raw multi-step trajectory loss.
- `gnode_semiimplicit`: updates velocity first, then uses the new velocity to update position during both training and inference.

The semi-implicit formula is:

```text
v_next = v + dt * predicted_acceleration
q_next = q + dt * v_next
```

The training code backpropagates through the full prediction sequence and accumulates microbatch gradients before each Adam update. The semi-implicit result is in **sheet row 23**.

The handover also lists one-step and scaled-loss GNODE results. Those experiments used separate historical runners; this module provides the native raw-loss and semi-implicit engines.

## PI-GNN: pair order and clipping

`PairEngine` starts from the **A10** recipe described in [`pignn_ablation`](../pignn_ablation/README.md): explicit training position update and generator-matched gravity target.

The settings combine two choices:

| Setting | Training-pair order | Gradient clipping |
|---|---|---|
| `pi_ordered_unclipped` | Original order | None |
| `pi_ordered_clip1` | Original order | Global norm 1.0 |
| `pi_shuffled_unclipped` | Shuffled each epoch | None |
| `pi_shuffled_clip1` | Shuffled each epoch | Global norm 1.0 |

Clipping is applied to the complete gradient before Adam. The sampler saves its order, cursor and random-number state for resuming.

The ordered, clipped experiment is **sheet row 29**. It stopped at epoch 16 after nonfinite validation results. The shuffled variants remain follow-up implementations.

## Two further controls

**Matched-tail curriculum:** `matched_tail_order()` shuffles the first 2,000 assigned batches and preserves the final 500 batches at horizon 100. This keeps the end of training identical while changing the earlier order.

**Central-force network:** `CentralPair` predicts equal-and-opposite pair forces along the line between bodies, using distance and masses. It has 17,153 parameters. Its `central_pair_gravity_off` and `central_pair_gravity_scaled` settings compare trajectory-only training with added scaled gravity supervision.

These two controls have synthetic tests. Their production experiments remain pending.

## Using the code

For TensorFlow components, configure the CPU runtime before importing the adapters:

```python
from competitors.gnode_pignn_training.production import runtime

runtime()

from competitors.ablation_controls.recipes import CELLS
from competitors.ablation_controls.tf_adapters import PairEngine

engine = PairEngine(CELLS["pi_ordered_clip1"])
```

This constructs a model and optimizer. A training runner supplies batches, validation, stopping limits and saved checkpoints.

`prepare.py` writes planning files under `runs/daniela-ablation-next-development-v1/`. It reads the shared batch bank from `runs/daniela-hgnn-order-reconstruction-audit-v1/`. A copy of that bank is included in [`archive/hgnn_gravity/`](../../archive/hgnn_gravity/); the preparation script currently uses the original local path.

The recipe status fields describe the initial development plan. Use the handover for the completed results. The current modules contain path changes and code extraction made during cleanup; original run checksums refer to the earlier files.

## Tests

Run from the repository root, using the appropriate framework environment:

```bash
python -m pytest competitors/ablation_controls/test_torch.py -q
python -m pytest competitors/ablation_controls/test_tf.py -q
```

The tests use synthetic states to check equations, gradients, ordering and save/restore behaviour. The TensorFlow module skips when TensorFlow is unavailable.
