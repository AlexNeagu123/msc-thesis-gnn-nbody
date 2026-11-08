# Handover Guide: HGNN, GNODE & PI-GNN Comparison

This guide explains where to find the results, how the shared dataset is set up, and which files in the repository correspond to each experiment in the comparison table.

---

## 1. Results & Documentation Links

- **Google Sheet (`NBody_Comparison_Matrix_Classic`):**
  - [Main Comparison Table](https://docs.google.com/spreadsheets/d/1wzKkCZGLHFLwPOaz-4Qm8OhZ8MhU6mVcx98ToaC4NQI/edit#gid=1292983321)
  - [HGNN Validation Curves](https://docs.google.com/spreadsheets/d/1wzKkCZGLHFLwPOaz-4Qm8OhZ8MhU6mVcx98ToaC4NQI/edit#gid=431043)
  - [Follow-up Validation Curves](https://docs.google.com/spreadsheets/d/1wzKkCZGLHFLwPOaz-4Qm8OhZ8MhU6mVcx98ToaC4NQI/edit#gid=431044)
- **Google Doc (Model Definitions, Diagrams & Experiment Notes):**
  - [Companion Guide](https://docs.google.com/document/d/1EdmC_roVJEul6SMDSv_RPP359c05Ku0cVoAgDU4mArs/edit)

---

## 2. Shared Dataset Setup

All models are trained and compared on the same 500 three-body RK4 trajectories from Daniela's dataset (`2,000 frames × 3 bodies × 5 features`, $\Delta t = 0.001$, $G = 1$, softening $\epsilon = 0.05$).

To convert the original CSV into the `.h5` splits used by this repository:

```bash
python -m data.import_daniela
```

This creates three whole-trajectory splits under `data/output/daniela-rk4/`:
- `train.h5` — **320 trajectories** (used for training)
- `val.h5` — **80 trajectories** (used to pick the best checkpoint based on 1,999-step rollout position MAE)
- `test.h5` — **100 trajectories** (extra comparison after checkpoint selection; I had already inspected this set during development)

In the tables below, *Pending* means I have yet to evaluate that checkpoint on the 100-system set. For stopped runs, the VAL80 score is the best eligible checkpoint saved before the run stopped.

### Frameworks and model code

**GNODE and PI-GNN use TensorFlow/Keras.** I turned the model and training code from the original `nbody-net` notebooks into Python modules under [`competitors/`](competitors/). My aim was to retrain the same architectures on the shared split, keeping their parameter counts: **83,714 for GNODE** and **34,690 for PI-GNN**. Keeping Keras also let me load the original checkpoints and check the predictions.

**HGNN uses PyTorch.** I built [`models/hgnn_canonical.py`](models/hgnn_canonical.py) for the variable-mass dataset, using momentum `p = m * v`. I also implemented gravity-loss variants and an exact-kinetic-energy version.

I used `train.h5` for training and `val.h5` for checkpoint selection across all three model families. I scored their predicted trajectories with the same NumPy/float64 metrics in [`evaluation/rk4_metrics.py`](evaluation/rk4_metrics.py).

---

## 3. Code Map for the Experiments

### A. HGNN (Hamiltonian Graph Neural Network)
- **Model code:** [`models/hgnn_canonical.py`](models/hgnn_canonical.py)
- **Training & validation scripts:** [`training/run_cpu_rk4.py`](training/run_cpu_rk4.py), [`training/cpu_microbatch.py`](training/cpu_microbatch.py), [`training/fixed_validation.py`](training/fixed_validation.py)
- **Exact kinetic energy ($T = \sum p_i^2 / 2m_i$) & gravity loss controls:** [`competitors/ablation_controls/hgnn.py`](competitors/ablation_controls/hgnn.py)

#### Where the loss is defined

I split the gravity experiments between YAML settings and Python training code.

**The YAML files hold the usual training settings:** model size, data paths, learning rate, batch size and curriculum. These two fields select the trajectory loss and its scaling:

```yaml
loss: mse
dynamic_loss_scaling: elapsed_time_v1
```

**I added the gravity term in Python**, in each experiment's `adapter.py`. It compares the acceleration from the learned potential, `-(dV/dq)/m`, with analytical gravity at the starting state of each training example.

The three variants are:

- **Row 11:** trajectory loss + `0.1 × acceleration MSE`, with additive softening `s = 0.05`.
- **Row 12:** the same loss, with `s = 0.05**2` to match the generator's force formula.
- **Row 13:** the row 12 gravity term divided by `882.8257392887597`, the TRAIN acceleration mean square.

Here, `s` appears in the gravity denominator as `(distance_squared + s)**1.5`.

The original adapters are stored locally at:

```text
runs/daniela-hgnn-gravity-soft005-v1/adapter.py
runs/daniela-hgnn-gravity-genforce-v1/adapter.py
runs/daniela-hgnn-gravity-norm-v1/adapter.py
```

`runs/` is excluded from Git. These files still use the original module and config names from the experiments.

For the later controls, I put the model and loss functions in [`competitors/ablation_controls/hgnn.py`](competitors/ablation_controls/hgnn.py):

- `objective(..., gravity="off")` uses the trajectory loss.
- `objective(..., gravity="scaled")` adds the scaled gravity term above.
- `KineticAblationHGNN(kinetic="exact")` uses `T = sum(p² / 2m)`.

The training loop selects and calls these functions. The links below point to the YAML settings or model/loss module for each result.

| Sheet Row | Experiment | Seed | VAL80 MAE | 100-System MAE | Settings / Module |
|---:|---|---:|---:|---:|---|
| 6 | **HGNN (Staged)** | 42 | `0.0563` | `0.0578` | [`configs/hgnn-cpu-long-horizons.yaml`](configs/hgnn-cpu-long-horizons.yaml) |
| 7 | `HGNN: 1-Step Only` | 42 | `0.1976` | `0.1986` | [`configs/hgnn-one-step-control-v1.yaml`](configs/hgnn-one-step-control-v1.yaml) |
| 8 | `HGNN: Shuffled Horizons` | 42 | `0.0821` | `0.0900` | [`configs/hgnn-shuffled-horizons-v1.yaml`](configs/hgnn-shuffled-horizons-v1.yaml) |
| 9 | `HGNN: Staged` | 43 | `0.0494` | `0.0469` | [`configs/hgnn-staged-seed43-v1.yaml`](configs/hgnn-staged-seed43-v1.yaml) |
| 10 | `HGNN: Shuffled Horizons` | 43 | `0.1087` | `0.1066` | [`configs/hgnn-shuffled-seed43-v1.yaml`](configs/hgnn-shuffled-seed43-v1.yaml) |
| 11 | `HGNN: Gravity Loss (s=0.05)` | 42 | `0.1712` | `0.2016` | [`configs/hgnn-gravity-soft005-v1.yaml`](configs/hgnn-gravity-soft005-v1.yaml) |
| 12 | `HGNN: Gravity Loss (Gen Force)` | 42 | `0.1266` | `0.1325` | [`configs/hgnn-gravity-genforce-v1.yaml`](configs/hgnn-gravity-genforce-v1.yaml) |
| 13 | `HGNN: Scaled Gravity Loss` | 42 | `0.0817` | `0.0843` | [`configs/hgnn-gravity-norm-v1.yaml`](configs/hgnn-gravity-norm-v1.yaml) |
| 14 | **`HGNN: Exact Kinetic (T)`** | 42 | **`0.0431`** | *Pending* | [`competitors/ablation_controls/hgnn.py`](competitors/ablation_controls/hgnn.py) |
| 15 | `HGNN: Exact T + Scaled Grav` | 42 | `0.0717` | *Pending* | [`competitors/ablation_controls/hgnn.py`](competitors/ablation_controls/hgnn.py) |
| 16 | `HGNN: Exact Kinetic (T)` | 43 | `0.0473` | *Pending* | [`competitors/ablation_controls/hgnn.py`](competitors/ablation_controls/hgnn.py) |
| 17 | `HGNN: Exact T + Scaled Grav` | 43 | `0.0866` | *Pending* | [`competitors/ablation_controls/hgnn.py`](competitors/ablation_controls/hgnn.py) |

---

### B. GNODE (Graph Neural ODE)
- **Model & rollout code:** [`competitors/gnode_pignn_tf/models.py`](competitors/gnode_pignn_tf/models.py)
- **Training engine:** [`competitors/gnode_pignn_training/engine.py`](competitors/gnode_pignn_training/engine.py)
- **Multi-step & semi-implicit step controls:** [`competitors/ablation_controls/tf_adapters.py`](competitors/ablation_controls/tf_adapters.py)

| Sheet Row | Experiment | Seed | VAL80 MAE | 100-System MAE | Module |
|---:|---|---:|---:|---:|---|
| 19 | **GNODE (125k 1-Step)** | 42 | `0.1485` | `0.1466` | [`competitors/gnode_pignn_training/`](competitors/gnode_pignn_training/) |
| 20 | `GNODE: Staged Multi-Step` | 42 | `0.4086` | `0.4106` | [`competitors/ablation_controls/tf_adapters.py`](competitors/ablation_controls/tf_adapters.py) |
| 21 | `GNODE: 1-Step Control (2.5k)` | 42 | `0.4202` | *Pending* | [`competitors/ablation_controls/tf_adapters.py`](competitors/ablation_controls/tf_adapters.py) |
| 22 | `GNODE: Scaled Multi-Step` | 42 | `0.4111` | *Pending* | [`competitors/ablation_controls/tf_adapters.py`](competitors/ablation_controls/tf_adapters.py) |
| 23 | `GNODE: Semi-Implicit Step` | 42 | `0.4202` | *Pending* | [`competitors/ablation_controls/tf_adapters.py`](competitors/ablation_controls/tf_adapters.py) |

---

### C. PI-GNN (Physics-Informed Direct Acceleration Network)
- **Model & training recipes:** [`competitors/pignn_ablation/recipes.py`](competitors/pignn_ablation/recipes.py)
- **Clipping, pair-shuffling & central-force controls:** [`competitors/ablation_controls/tf_adapters.py`](competitors/ablation_controls/tf_adapters.py)

| Sheet Row | Experiment | Seed | VAL80 MAE | 100-System MAE | Module |
|---:|---|---:|---:|---:|---|
| 25 | **PI-GNN (Original `A00`)** | 42 | `0.2267` | `0.2492` | [`competitors/pignn_ablation/`](competitors/pignn_ablation/) |
| 26 | `PI-GNN: Consistent Step (A01)` | 42 | `0.2069` | `0.2342` | [`competitors/pignn_ablation/`](competitors/pignn_ablation/) |
| 27 | `PI-GNN: Generator Force (A10)` | 42 | `39.6269` *(Stopped Ep 48)* | *Stopped* | [`competitors/pignn_ablation/`](competitors/pignn_ablation/) |
| 28 | `PI-GNN: Both Changes (A11)` | 42 | `39.6269` *(Stopped Ep 48)* | *Stopped* | [`competitors/pignn_ablation/`](competitors/pignn_ablation/) |
| 29 | `PI-GNN: Gen Force + Clip 1.0` | 42 | `5043.0916` *(Stopped Ep 16)* | *Stopped* | [`competitors/ablation_controls/tf_adapters.py`](competitors/ablation_controls/tf_adapters.py) |

---

## 4. Follow-Up Experiments Already Implemented in Code

The [`competitors/ablation_controls/`](competitors/ablation_controls/) package also contains tested implementations for three follow-up ideas if anyone wants to run them next:

1. **Matched-tail curriculum (`hgnn_matched_tail` in `schedule.py`):** Shuffles the first 2,000 training batches (`h=1, 5, 20, 50`) while keeping the last 500 `h=100` batches in the exact same order.
2. **Shuffled training pairs for PI-GNN (`pi_shuffled_unclipped` in `tf_adapters.py`):** Shuffles the 639,680 training pairs each epoch instead of training on trajectories in sequential order.
3. **Symmetric central-force pair network (`central_pair_gravity_off` / `scaled` in `tf_adapters.py`):** A smaller network that predicts pairwise equal-and-opposite central forces along the line connecting each pair of bodies.
