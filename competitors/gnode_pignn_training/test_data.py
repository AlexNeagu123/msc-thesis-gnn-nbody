"""Framework-free ordering and sampler continuation checks."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from competitors.gnode_pignn_training.data import PairSampler, load_train_pairs


@pytest.mark.parametrize("shuffle", [False, True])
def test_sampler_epoch_remainder_and_rng_resume(shuffle: bool) -> None:
    """Every pair appears once; replay includes a remainder and new permutation."""
    sampler = PairSampler(13, 5, shuffle)
    first = sampler.next_indices()
    second = sampler.next_indices()
    state = sampler.state()
    last = sampler.next_indices()
    assert len(last) == 3
    np.testing.assert_array_equal(np.sort(np.concatenate([first, second, last])), np.arange(13))
    next_epoch = sampler.next_indices()
    restored = PairSampler(13, 5, shuffle, seed=912)
    restored.restore(state)
    np.testing.assert_array_equal(restored.next_indices(), last)
    np.testing.assert_array_equal(restored.next_indices(), next_epoch)
    assert restored.epoch == 1
    with pytest.raises(ValueError, match="identity"):
        PairSampler(14, 5, shuffle).restore(state)


def test_train_pair_layout_and_no_system_boundary_crossing() -> None:
    """Independent direct HDF5 samples verify sorted-ID flattening and features."""
    path = Path(__file__).resolve().parents[2] / "data/output/daniela-rk4/train.h5"
    if not path.exists():
        pytest.skip("local imported data required")
    x, y, ids = load_train_pairs(path)
    assert x.shape == (639680, 3, 5)
    assert y.shape == (639680, 3, 4)
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    with h5py.File(path) as handle:
        original_ids = handle["simulation_ids"][:]
        for system, step in [(0, 0), (0, 1998), (1, 0), (160, 1000), (319, 1998)]:
            index = int(np.flatnonzero(original_ids == ids[system])[0])
            a = handle["trajectories"][index, step]
            b = handle["trajectories"][index, step + 1]
            flat = system * 1999 + step
            np.testing.assert_array_equal(x[flat], a[:, [4, 0, 1, 2, 3]].astype(np.float32))
            np.testing.assert_array_equal(y[flat], b[:, :4].astype(np.float32))
    with pytest.raises(ValueError, match="TRAIN"):
        load_train_pairs(path.with_name("val.h5"))
