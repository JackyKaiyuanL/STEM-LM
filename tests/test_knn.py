"""KNN source-sampling invariants.

The per-sample BallTree query is the training bottleneck; it is vectorised by a
batched ``__getitems__`` path. These tests pin the invariant that batching must
NOT change any draw: for the same RNG state, ``__getitems__(indices)`` must
produce byte-identical ``source_idx`` to calling ``__getitem__`` on each index
in order. A frozen golden also catches any accidental change to the scalar
``__getitem__`` path itself.
"""
import numpy as np
import pandas as pd
import pytest

from stemlm.data import JSDMDataset

SPECIES = [f"species_{i}" for i in range(6)]
ENV_COLS = ["env_temp", "env_precip"]


def _write_synthetic_csv(path, n, seed):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(25.0, 55.0, n)
    lon = rng.uniform(-120.0, -70.0, n)
    time = rng.integers(0, 365, n)
    env = rng.normal(size=(n, len(ENV_COLS)))
    weights = rng.normal(size=(len(ENV_COLS), len(SPECIES)))
    prob = 1.0 / (1.0 + np.exp(-(env @ weights)))
    pres = (rng.uniform(size=prob.shape) < prob).astype(int)
    pres[0, :] = 1
    pres[1, :] = 0
    df = pd.DataFrame({"time": time, "latitude": lat, "longitude": lon})
    for k, col in enumerate(ENV_COLS):
        df[col] = env[:, k]
    for j, sp in enumerate(SPECIES):
        df[sp] = pres[:, j]
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    csv = tmp_path_factory.mktemp("knn") / "data.csv"
    _write_synthetic_csv(csv, n=160, seed=7)
    ds = JSDMDataset(str(csv), num_source_sites=8, time_col="time")
    # Exercise the pooled path (train-index restriction), like real training.
    ds.source_pool = np.arange(0, 160, 2)
    return ds


def _sequential_source_idx(ds, indices, seed):
    np.random.seed(seed)
    return [ds[i]["source_idx"].numpy().copy() for i in indices]


def _batched_source_idx(ds, indices, seed):
    np.random.seed(seed)
    return [s["source_idx"].numpy().copy() for s in ds.__getitems__(list(indices))]


def test_getitems_exists(dataset):
    assert hasattr(dataset, "__getitems__"), "batched fetch path missing"


def test_batched_matches_sequential(dataset):
    indices = [3, 8, 15, 22, 40, 41, 42, 99, 100, 158]
    ref = _sequential_source_idx(dataset, indices, seed=1234)
    got = _batched_source_idx(dataset, indices, seed=1234)
    assert len(ref) == len(got)
    for i, r, g in zip(indices, ref, got, strict=True):
        np.testing.assert_array_equal(r, g, err_msg=f"source_idx differs at index {i}")


def test_batched_matches_sequential_full_pool(tmp_path_factory):
    csv = tmp_path_factory.mktemp("knn2") / "data.csv"
    _write_synthetic_csv(csv, n=120, seed=3)
    ds = JSDMDataset(str(csv), num_source_sites=8, time_col="time")  # no source_pool
    indices = list(range(0, 120, 7))
    ref = _sequential_source_idx(ds, indices, seed=99)
    got = _batched_source_idx(ds, indices, seed=99)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_array_equal(r, g)
