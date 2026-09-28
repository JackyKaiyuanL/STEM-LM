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

from stemlm.data import (
    _K_MAX,
    _K_QUERY_OVERFETCH,
    JSDMDataset,
    haversine_pairs_np,
)

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


def _bruteforce_candidates(ds, idx):
    """Exact nearest-in-pool candidate set the FAISS path should reproduce:
    the in-pool points among the k_query true-nearest by haversine."""
    n = len(ds.lats)
    k_query = min(_K_MAX + 1, n)
    if ds._source_pool_mask is not None:
        k_query = min(k_query * _K_QUERY_OVERFETCH, n)
    d = haversine_pairs_np(ds.lats[idx], ds.lons[idx], ds.lats, ds.lons)
    order = np.argsort(d, kind="stable")[:k_query]
    keep = order != idx
    if ds._source_pool_mask is not None:
        keep &= ds._source_pool_mask[order]
    return order[keep]


def test_faiss_candidates_match_bruteforce(dataset):
    """FAISS (exact Flat at this N) must return exactly the brute-force
    haversine nearest-in-pool — validates the xyz L2 == great-circle ordering."""
    for idx in [3, 8, 40, 99, 158]:
        cand, sp = dataset._knn_candidates(idx)
        expected = _bruteforce_candidates(dataset, idx)
        assert set(cand.tolist()) == set(expected.tolist()), f"candidate set differs at {idx}"
        assert np.all(np.diff(sp) >= -1e-3), "candidates not sorted by distance"


def test_random_exclusion_keeps_batched_equal_to_sequential(dataset):
    dataset.random_exclusion_rows = np.arange(0, 160, 3)
    try:
        indices = [3, 8, 15, 22, 40, 41, 42, 99, 100, 158]
        ref = _sequential_source_idx(dataset, indices, seed=5)
        got = _batched_source_idx(dataset, indices, seed=5)
        for r, g in zip(ref, got, strict=True):
            np.testing.assert_array_equal(r, g)
    finally:
        dataset._random_exclusion_mask = None


def test_random_window_draws_positive_radii(dataset):
    np.random.seed(11)
    radii = [dataset._random_window(dataset._knn_candidates(i)[1]) for i in range(160)]
    assert 0 < np.mean(np.array(radii) > 0) < 1
    for i in range(0, 160, 5):
        sp = dataset._knn_candidates(i)[1]
        assert dataset._random_window(sp) <= sp.max()


def test_sources_are_the_nearest_in_pool(dataset):
    for i in [3, 40, 99, 158]:
        src = dataset[i]["source_idx"].numpy()
        expected = _bruteforce_candidates(dataset, i)[:dataset.num_source_sites]
        np.testing.assert_array_equal(src, expected)


def test_eval_exclusion_days_removes_near_in_time(dataset):
    dataset.eval_exclusion_days = 100.0
    try:
        for i in [3, 40, 99]:
            src = dataset[i]["source_idx"].numpy()
            assert (np.abs(dataset.times[src] - dataset.times[i]) >= 100.0).all()
    finally:
        dataset.eval_exclusion_days = 0.0


def test_causal_context_keeps_only_earlier_sources(dataset):
    dataset.causal_context = True
    try:
        for i in [3, 40, 99]:
            src = dataset[i]["source_idx"].numpy()
            assert (dataset.times[src] <= dataset.times[i]).all()
    finally:
        dataset.causal_context = False


def test_random_windows_never_empty_the_sources(dataset):
    dataset.random_exclusion_rows = np.arange(160)
    try:
        for seed in range(3):
            np.random.seed(seed)
            for i in range(160):
                assert dataset[i]["source_idx"].shape == (dataset.num_source_sites,)
    finally:
        dataset._random_exclusion_mask = None


def test_random_window_is_zero_without_positive_gaps(dataset):
    np.random.seed(0)
    assert all(dataset._random_window(np.zeros(10)) == 0.0 for _ in range(20))


def test_eval_exclusion_removes_near_sources(dataset):
    dataset.eval_exclusion_km = 300.0
    try:
        np.random.seed(2)
        for i in [3, 40, 99]:
            src = dataset[i]["source_idx"].numpy()
            d = haversine_pairs_np(dataset.lats[i], dataset.lons[i],
                                   dataset.lats[src], dataset.lons[src])
            assert (d >= 300.0).all()
    finally:
        dataset.eval_exclusion_km = 0.0


def test_batched_matches_sequential_full_pool(tmp_path_factory):
    csv = tmp_path_factory.mktemp("knn2") / "data.csv"
    _write_synthetic_csv(csv, n=120, seed=3)
    ds = JSDMDataset(str(csv), num_source_sites=8, time_col="time")  # no source_pool
    indices = list(range(0, 120, 7))
    ref = _sequential_source_idx(ds, indices, seed=99)
    got = _batched_source_idx(ds, indices, seed=99)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_array_equal(r, g)


def test_heldout_rows_are_sources_only_within_their_split_and_outside_their_cell(dataset):
    from stemlm.data import heldout_split_ids

    pool, val, test = set(range(0, 160, 2)), set(range(1, 160, 4)), set(range(3, 160, 4))
    dataset.heldout_split = heldout_split_ids(160, sorted(val), sorted(test))
    dataset.cell = np.arange(160) // 2
    try:
        for i in [1, 5, 41]:
            src = set(dataset[i]["source_idx"].numpy())
            assert src <= (pool | val) and not (src & test) and (i - 1) not in src
        for i in [3, 7, 43]:
            src = set(dataset[i]["source_idx"].numpy())
            assert src <= (pool | test) and not (src & val) and (i - 1) not in src
        for i in [0, 4, 40]:
            assert set(dataset[i]["source_idx"].numpy()) <= pool
    finally:
        dataset.heldout_split = None
        dataset.cell = None
