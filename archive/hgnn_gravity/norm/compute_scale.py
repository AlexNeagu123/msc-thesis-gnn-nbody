"""TRAIN-only float64 acceleration RMS; no validation or diagnostic access."""
import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from training.fixed_validation import file_sha256
ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('raw_adapter', ROOT / 'runs/daniela-hgnn-gravity-genforce-v1/adapter.py')
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    source = ROOT / 'data/output/daniela-rk4/train.h5'
    sums, means, count = [], [], 0
    with h5py.File(source) as f:
        assert f['trajectories'].shape == (320, 2000, 3, 5)
        for i in range(320):
            states = torch.from_numpy(f['trajectories'][i].astype(np.float64))
            acc = a.analytical_gravity_acceleration(states, additive=0.05**2)
            # Independent NumPy direct pair loop verifies target and coordinate ordering.
            x = states.numpy(); independent = np.zeros((2000,3,2), dtype=np.float64)
            for j in range(3):
                for k in range(3):
                    if j != k:
                        delta = x[:,k,:2] - x[:,j,:2]
                        independent[:,j] += x[:,k,4,None]*delta/(np.sum(delta**2,axis=-1,keepdims=True)+0.05**2)**1.5
            np.testing.assert_allclose(acc.numpy(), independent, rtol=1e-13, atol=1e-11)
            sums.append(float(acc.square().sum())); means.append(float(acc.sum())); count += acc.numel()
    import math
    value = math.fsum(sums)/count
    report = dict(mean_square=value, rms=math.sqrt(value), component_mean=math.fsum(means)/count,
                  definition='uncentred mean square over TRAIN320 all2000 frames, 3 bodies, 2 components',
                  additive_softening=0.05**2, count=count, train_sha256=file_sha256(source),
                  independent_numpy_target_verified=True, script_sha256=file_sha256(Path(__file__)))
    with (WORK/'acceleration_scale.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__ == '__main__': main()
