"""The fast paths in dataset loading must reproduce the slow ones exactly.

Both changes replace per-row Python loops over the full dataset:
  * the species CSR now comes from the Arrow list column's offsets/values;
  * the H3 block split now tests membership in integer cell codes, not strings.
Each is pinned here against a direct implementation of what it replaced.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from stemlm.data import _list_column_to_csr_arrays, h3_block_split


def _csr_arrays_rowwise(lists):
    """The previous implementation: one Python step per row."""
    counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
    indptr = np.empty(len(lists) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = (np.concatenate([np.asarray(x, dtype=np.int32) for x in lists])
               if len(lists) else np.array([], dtype=np.int32))
    return indptr, indices


@pytest.mark.parametrize("lists", [
    [[0, 3, 7], [], [199], [1, 2], [5]],           # mixed, including an empty row
    [[]],                                           # single empty row
    [[i % 200 for i in range(k)] for k in range(9)],  # ragged
])
def test_csr_from_arrow_matches_rowwise(lists):
    col = pa.chunked_array([pa.array(lists, type=pa.list_(pa.int32()))])
    indptr, indices = _list_column_to_csr_arrays(col)
    ref_indptr, ref_indices = _csr_arrays_rowwise(lists)
    np.testing.assert_array_equal(indptr, ref_indptr)
    np.testing.assert_array_equal(indices, ref_indices)


def test_csr_from_arrow_handles_multiple_chunks():
    """Shards arrive as separate chunks; offsets must be rebased across them."""
    a = pa.array([[1, 2], [3]], type=pa.list_(pa.int32()))
    b = pa.array([[], [4, 5, 6]], type=pa.list_(pa.int32()))
    indptr, indices = _list_column_to_csr_arrays(pa.chunked_array([a, b]))
    ref_indptr, ref_indices = _csr_arrays_rowwise([[1, 2], [3], [], [4, 5, 6]])
    np.testing.assert_array_equal(indptr, ref_indptr)
    np.testing.assert_array_equal(indices, ref_indices)


def test_csr_from_arrow_rejects_nulls():
    col = pa.chunked_array([pa.array([[1], None], type=pa.list_(pa.int32()))])
    with pytest.raises(ValueError, match="nulls"):
        _list_column_to_csr_arrays(col)


def _h3_split_via_strings(lats, lons, resolution, train_frac, test_frac, seed):
    """The previous implementation: np.isin over arrays of cell strings."""
    import h3 as h3lib
    cells = np.array([h3lib.latlng_to_cell(float(a), float(b), resolution)
                      for a, b in zip(lats, lons, strict=False)])
    unique_cells = np.unique(cells)
    rng = np.random.RandomState(seed)
    unique_cells = unique_cells[rng.permutation(len(unique_cells))]
    n = len(unique_cells)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * (1 - train_frac - test_frac)))
    test_cells = set(unique_cells[:n_test])
    val_cells = set(unique_cells[n_test:n_test + n_val])
    return (np.where(~np.isin(cells, list(test_cells | val_cells)))[0],
            np.where(np.isin(cells, list(val_cells)))[0],
            np.where(np.isin(cells, list(test_cells)))[0])


@pytest.mark.parametrize("seed", [0, 42])
def test_h3_split_codes_match_string_version(seed):
    rng = np.random.default_rng(1)
    lats = rng.uniform(25.0, 55.0, 4000)
    lons = rng.uniform(-120.0, -70.0, 4000)
    got = h3_block_split(lats, lons, resolution=3, seed=seed)
    ref = _h3_split_via_strings(lats, lons, 3, 0.8, 0.1, seed)
    for g, r in zip(got, ref, strict=True):
        np.testing.assert_array_equal(g, r)


def test_h3_split_partitions_every_row():
    rng = np.random.default_rng(2)
    lats = rng.uniform(25.0, 55.0, 1500)
    lons = rng.uniform(-120.0, -70.0, 1500)
    tr, va, te = h3_block_split(lats, lons, resolution=2, seed=7)
    allocated = np.concatenate([tr, va, te])
    assert len(allocated) == 1500
    np.testing.assert_array_equal(np.sort(allocated), np.arange(1500))


def test_sparse_dataset_loads_from_directory(tmp_path):
    """End-to-end: the Arrow path builds a working dataset from sharded parquet."""
    import json

    from stemlm.data import JSDMSparseDataset

    vocab = [f"sp_{i}" for i in range(6)]
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rng = np.random.default_rng(3)
    for s in range(2):
        n = 60
        pd.DataFrame({
            "time": rng.integers(0, 365, n),
            "latitude": rng.uniform(25, 55, n),
            "longitude": rng.uniform(-120, -70, n),
            "env_a": rng.normal(size=n),
            "species_idx": [sorted(rng.choice(6, rng.integers(1, 4), replace=False).tolist())
                            for _ in range(n)],
        }).to_parquet(shard_dir / f"part_{s}.parquet", index=False)

    ds = JSDMSparseDataset(str(shard_dir), str(tmp_path / "vocab.json"),
                           num_source_sites=4)
    assert len(ds) == 120
    assert ds.num_species == 6
    item = ds[0]
    assert item["source_species"].shape == (6, 4)
    assert item["target_species"].shape == (6,)
