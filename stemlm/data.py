import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

EARTH_RADIUS_KM = 6371.0


def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)
    # Each worker runs its own FAISS queries; keep them single-threaded so N
    # workers don't oversubscribe the box (N x OpenMP pools).
    import faiss
    faiss.omp_set_num_threads(1)


_K_MAX = 1024
_K_QUERY_OVERFETCH = 4

_FAISS_IVF_MIN = 200_000   # below this, exact flat index (IVF training not worth it)
_FAISS_NLIST_MAX = 8192
_FAISS_NPROBE = 8          # 3D coords => recall is already 1.0 here (measured at
                           # 71M rows, k=4100: nprobe 8 and 32 both retrieve the
                           # exact neighbour set, but 8 is 1.8x faster; 4 drops to
                           # recall 0.9976, so 8 is the floor that stays exact).


def _lonlat_to_xyz(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Map lat/lon (degrees) to unit-sphere xyz. L2 order on xyz is monotone in
    great-circle distance, so nearest-by-L2 == nearest-by-haversine (exactly)."""
    la = np.radians(np.asarray(lats, dtype=np.float64))
    lo = np.radians(np.asarray(lons, dtype=np.float64))
    xyz = np.column_stack([np.cos(la) * np.cos(lo),
                           np.cos(la) * np.sin(lo),
                           np.sin(la)])
    return np.ascontiguousarray(xyz, dtype=np.float32)


class _HaversineKNNIndex:
    """FAISS nearest-neighbour index over unit-sphere xyz — drop-in for the
    sklearn BallTree. ~20x faster queries and ~90x faster build at 71M scale,
    with exact ordering (L2 on the sphere is monotone in great-circle distance).
    IVFFlat above _FAISS_IVF_MIN points; exact IndexFlatL2 below."""

    def __init__(self, lats: np.ndarray, lons: np.ndarray):
        import faiss
        xyz = _lonlat_to_xyz(lats, lons)
        n = len(lats)
        if n < _FAISS_IVF_MIN:
            index = faiss.IndexFlatL2(3)
        else:
            nlist = int(min(_FAISS_NLIST_MAX, max(64, round(math.sqrt(n)))))
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, nlist)
            n_train = min(n, max(nlist * 40, 100_000))
            sample = (xyz if n_train == n
                      else xyz[np.random.default_rng(0).choice(n, n_train, replace=False)])
            index.train(sample)
            index.nprobe = _FAISS_NPROBE
        index.add(xyz)
        self._index = index

    def query(self, coords_deg: np.ndarray, k: int) -> np.ndarray:
        """(B, k) global neighbour indices, nearest-first. Slots past the
        available count are -1 (FAISS padding)."""
        xyz = _lonlat_to_xyz(coords_deg[:, 0], coords_deg[:, 1])
        _, idx = self._index.search(xyz, k)
        return idx


def haversine_pairs_np(lat_a: np.ndarray, lon_a: np.ndarray,
                       lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
                       
    lat_a = np.radians(np.asarray(lat_a, dtype=np.float64))
    lon_a = np.radians(np.asarray(lon_a, dtype=np.float64))
    lat_b = np.radians(np.asarray(lat_b, dtype=np.float64))
    lon_b = np.radians(np.asarray(lon_b, dtype=np.float64))
    dlat = lat_a - lat_b
    dlon = lon_a - lon_b
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2.0) ** 2
    return (EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))).astype(np.float32)


def haversine_pairs_torch(lat_a: torch.Tensor, lon_a: torch.Tensor,
                          lat_b: torch.Tensor, lon_b: torch.Tensor) -> torch.Tensor:
                          
    lat_a = torch.deg2rad(lat_a)
    lon_a = torch.deg2rad(lon_a)
    lat_b = torch.deg2rad(lat_b)
    lon_b = torch.deg2rad(lon_b)
    dlat = lat_a - lat_b
    dlon = lon_a - lon_b
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat_a) * torch.cos(lat_b) * torch.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * torch.asin(torch.sqrt(a.clamp(min=0.0)))


def _bbox_max_distance(lats: np.ndarray, lons: np.ndarray, times: np.ndarray,
                       euclidean: bool):
    lat_min = float(lats.min())
    lat_max = float(lats.max())
    lon_min = float(lons.min())
    lon_max = float(lons.max())
    if euclidean:
        max_sp = float(np.hypot(lat_max - lat_min, lon_max - lon_min))
    else:
        c_lat = np.array([lat_min, lat_min, lat_max, lat_max], dtype=np.float64)
        c_lon = np.array([lon_min, lon_max, lon_min, lon_max], dtype=np.float64)
        i, j = np.triu_indices(4, k=1)
        max_sp = float(haversine_pairs_np(c_lat[i], c_lon[i], c_lat[j], c_lon[j]).max())
    max_tp = float(times.max() - times.min()) if len(times) else 0.0
    return max_sp, max_tp


def _positive_floor(d: np.ndarray, scale: float) -> float:
    pos = d[d > 0]
    return float(pos.min()) if pos.size else scale * 1e-6


def _normalize_time_col(df: pd.DataFrame, time_col: str, no_time: bool) -> bool:
    has_time = (not no_time) and (time_col in df.columns)
    if not has_time:
        reason = "--no_time" if no_time else f"column '{time_col}' not found"
        print(f"  Time ignored ({reason}) — purely spatial model")
        return False
    col = df[time_col]
    is_stringlike = (col.dtype == "object"
                     or pd.api.types.is_datetime64_any_dtype(col)
                     or pd.api.types.is_string_dtype(col))
    if is_stringlike:
        dt = pd.to_datetime(col)
        df[time_col] = (dt - dt.min()).dt.days.astype(float)
    else:
        df[time_col] = df[time_col].astype(float)
    return True


def _list_column_to_csr_arrays(column) -> tuple[np.ndarray, np.ndarray]:
    """CSR ``(indptr, indices)`` from an Arrow list column, without copying rows.

    A list array stores one offsets buffer plus one flat values buffer — the same
    layout CSR wants — so this is buffer arithmetic rather than a per-row loop.
    """
    import pyarrow as pa

    arr = column.combine_chunks()
    if isinstance(arr, pa.ChunkedArray):
        arr = (arr.chunk(0) if arr.num_chunks == 1
               else pa.concat_arrays([c for c in arr.iterchunks()]))
    if arr.null_count:
        raise ValueError("species_idx contains nulls; expected a list per row")

    offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    values = arr.values.to_numpy(zero_copy_only=False)
    # A sliced array's offsets do not start at 0; rebase so they index `values`.
    start = int(offsets[0])
    if start:
        offsets = offsets - start
    values = values[start:start + int(offsets[-1])]
    return offsets, values.astype(np.int32, copy=False)


class _SparseSpeciesData:
    def __init__(self, csr):
        self._csr = csr
        self.shape = csr.shape

    def __len__(self):
        return self._csr.shape[0]

    def __getitem__(self, key):
        sub = self._csr[key]
        arr = np.asarray(sub.todense(), dtype=np.float32)
        if isinstance(key, (int, np.integer)):
            return arr.ravel()
        return arr


class JSDMDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        num_source_sites: int = 64,
        num_scale_sites: int | None = None,
        time_col: str = "time",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        env_cols: list[str] | None = None,
        euclidean_coords: bool = False,
        no_time: bool = False,
    ):
        super().__init__()
        self.num_source_sites = num_source_sites
        self.num_scale_sites = num_scale_sites if num_scale_sites is not None else num_source_sites
        df = pd.read_csv(csv_path)

        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(
                "NaNs found in columns: " + ", ".join(nan_cols)
                + ". Please impute or drop missing values before training."
            )

        has_time = _normalize_time_col(df, time_col, no_time)
        coord_cols = ([time_col] if has_time else []) + [lat_col, lon_col]
        if env_cols is None:
            env_cols     = [c for c in df.columns if c not in coord_cols and c.startswith("env_")]
            species_cols = [c for c in df.columns if c not in coord_cols and not c.startswith("env_")]
        else:
            species_cols = [c for c in df.columns if c not in coord_cols and c not in env_cols]

        species_data = df[species_cols].values.astype(np.float32)
        self._setup_post_load(df, species_data, species_cols, env_cols,
                              time_col, lat_col, lon_col, has_time,
                              euclidean_coords)

    def _setup_post_load(self, df, species_data, species_cols, env_cols,
                         time_col, lat_col, lon_col, has_time,
                         euclidean_coords):
        self.species_cols = species_cols
        self.env_cols = env_cols
        self.num_species = len(species_cols)
        if not env_cols:
            print(
                "WARNING: no environmental columns detected (none with 'env_' prefix "
                "and no --env_cols given). Falling back to a single zero-valued env "
                "column; the model will train without environmental signal."
            )
        self.num_env_vars = len(env_cols) if env_cols else 1

        self.coords = df[[lat_col, lon_col]].values.astype(np.float32)
        self.lats = self.coords[:, 0]
        self.lons = self.coords[:, 1]
        N = len(df)
        self.times = df[time_col].values.astype(np.float32) if has_time else np.zeros(N, dtype=np.float32)
        self.species_data = species_data
        self.env_data = (
            df[env_cols].values.astype(np.float32) if env_cols
            else np.zeros((N, 1), dtype=np.float32)
        )
        self.euclidean_coords = bool(euclidean_coords)
        self.has_time = bool(has_time)

        print(f"Dataset: {N} observations, {self.num_species} species, {self.num_env_vars} env vars")
        max_sp, max_tp = _bbox_max_distance(self.lats, self.lons, self.times,
                                            euclidean=self.euclidean_coords)
        self._max_spatial = max_sp
        self._max_temporal = max_tp if has_time else 0.0

        if self.euclidean_coords:
            self._knn_index = None
        else:
            self._knn_index = _HaversineKNNIndex(self.lats, self.lons)
        self._source_pool = None
        self._source_pool_mask = None

    @property
    def source_pool(self):
        return self._source_pool

    @source_pool.setter
    def source_pool(self, value):
        self._source_pool = value
        if value is None:
            self._source_pool_mask = None
        else:
            mask = np.zeros(len(self.lats), dtype=bool)
            mask[np.asarray(value)] = True
            self._source_pool_mask = mask

    def __len__(self):
        return len(self.species_data)

    def _knn_candidates(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        N_total = len(self.lats)
        pool_mask = self._source_pool_mask
        k_query = min(_K_MAX + 1, N_total)
        if pool_mask is not None:
            k_query = min(k_query * _K_QUERY_OVERFETCH, N_total)

        while True:
            neigh = self._knn_index.query(self.coords[idx:idx + 1], k_query)[0]
            neigh = neigh[neigh >= 0]
            keep = neigh != idx
            if pool_mask is not None:
                keep &= pool_mask[neigh]
            neigh = neigh[keep]
            if len(neigh) >= max(self.num_source_sites, self.num_scale_sites) or k_query >= N_total:
                break
            k_query = min(k_query * 2, N_total)
        sp = haversine_pairs_np(self.lats[idx], self.lons[idx],
                                self.lats[neigh], self.lons[neigh])
        return neigh, sp

    def _candidates_scalar(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Neighbour indices + spatial distances for one target (no batching)."""
        if self._knn_index is None:
            cand_idx = (self._source_pool if self._source_pool is not None
                        else np.arange(len(self.lats)))
            cand_idx = np.asarray(cand_idx)
            cand_idx = cand_idx[cand_idx != idx]
            sp = haversine_pairs_np(self.lats[idx], self.lons[idx],
                                    self.lats[cand_idx], self.lons[cand_idx]) \
                if not self.euclidean_coords else \
                np.sqrt((self.lats[cand_idx] - self.lats[idx]) ** 2
                        + (self.lons[cand_idx] - self.lons[idx]) ** 2).astype(np.float32)
            return cand_idx, sp
        return self._knn_candidates(idx)

    def _candidates_batch(self, indices: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Neighbour indices + distances for a batch via ONE FAISS query.

        Distribution-identical to calling ``_candidates_scalar`` per index: FAISS
        ranks (recall ~1.0 in 3D) match the exact nearest, so kept neighbours and
        their order do not depend on batching, and any item too short after the
        shared query falls back to the scalar doubling path.
        """
        if self._knn_index is None:
            return [self._candidates_scalar(i) for i in indices]

        N_total = len(self.lats)
        pool_mask = self._source_pool_mask
        k_query = min(_K_MAX + 1, N_total)
        if pool_mask is not None:
            k_query = min(k_query * _K_QUERY_OVERFETCH, N_total)

        idx_arr = np.asarray(indices)
        neigh = self._knn_index.query(self.coords[idx_arr], k_query)

        need = max(self.num_source_sites, self.num_scale_sites)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for b, idx in enumerate(indices):
            nb = neigh[b]
            nb = nb[nb >= 0]
            keep = nb != idx
            if pool_mask is not None:
                keep &= pool_mask[nb]
            nb = nb[keep]
            if len(nb) >= need or k_query >= N_total:
                sp = haversine_pairs_np(self.lats[idx], self.lons[idx],
                                        self.lats[nb], self.lons[nb])
                out.append((nb, sp))
            else:
                # Rare short-pool tail: redo this one exactly as the scalar path.
                out.append(self._knn_candidates(idx))
        return out

    def _sample_from_candidates(self, idx: int, cand_idx: np.ndarray,
                                sp: np.ndarray) -> dict[str, Any]:
        """Weight candidates by distance and draw source sites. Consumes one
        ``np.random.choice`` — callers must invoke in a fixed index order to keep
        draws reproducible."""
        N = self.num_source_sites
        if self.has_time:
            tp = np.abs(self.times[cand_idx] - self.times[idx]).astype(np.float32)
        else:
            tp = np.zeros_like(sp)

        n_cand = len(cand_idx)
        replace = n_cand - 1 < N

        k1 = min(self.num_scale_sites, n_cand)
        nearest = np.argpartition(sp, k1 - 1)[:k1] if k1 < n_cand else np.arange(n_cand)
        s_sp = max(float(np.median(sp[nearest])), _positive_floor(sp, self._max_spatial))
        if self.has_time:
            s_tp = max(float(np.median(tp[nearest])), _positive_floor(tp, self._max_temporal))
        else:
            s_tp = 1.0

        combined = np.sqrt((sp / s_sp) ** 2 + (tp / s_tp) ** 2) if self.has_time else (sp / s_sp)
        inv_d = 1.0 / (combined + 1e-6)
        w2 = inv_d / inv_d.sum()
        source_idx = cand_idx[np.random.choice(n_cand, size=N, replace=replace, p=w2)]

        source_species = np.ascontiguousarray(self.species_data[source_idx].T)
        source_env = self.env_data[source_idx]
        target_env = self.env_data[idx]

        return {
            "target_species": torch.from_numpy(self.species_data[idx]),
            "source_species": torch.from_numpy(source_species),
            "source_env":     torch.from_numpy(source_env),
            "target_env":     torch.from_numpy(target_env),
            "target_idx":     torch.tensor(idx, dtype=torch.long),
            "source_idx":     torch.from_numpy(source_idx.astype(np.int64)),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cand_idx, sp = self._candidates_scalar(idx)
        return self._sample_from_candidates(idx, cand_idx, sp)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        """Batched fetch used by DataLoader: one tree query for the whole batch,
        then per-item sampling in list order — byte-identical to sequential
        ``__getitem__`` for the same RNG state."""
        cands = self._candidates_batch(indices)
        return [self._sample_from_candidates(i, c, s)
                for i, (c, s) in zip(indices, cands, strict=True)]


class JSDMSparseDataset(JSDMDataset):
    def __init__(
        self,
        parquet_path: str,
        vocab_path: str,
        num_source_sites: int = 64,
        num_scale_sites: int | None = None,
        time_col: str = "time",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        env_cols: list[str] | None = None,
        euclidean_coords: bool = False,
        no_time: bool = False,
    ):
        Dataset.__init__(self)
        self.num_source_sites = num_source_sites
        self.num_scale_sites = num_scale_sites if num_scale_sites is not None else num_source_sites

        import pyarrow.dataset as pa_ds

        if os.path.isdir(parquet_path):
            parts = sorted(os.path.join(parquet_path, f)
                           for f in os.listdir(parquet_path) if f.endswith(".parquet"))
            if not parts:
                raise FileNotFoundError(f"No .parquet shards in {parquet_path}")
        else:
            parts = [parquet_path]
        table = pa_ds.dataset(parts, format="parquet").to_table()
        with open(vocab_path) as f:
            species_cols = json.load(f)
        if "species_idx" not in table.column_names:
            raise ValueError(
                "Sparse parquet must have a 'species_idx' column "
                "(variable-length list<int32> per row). Got: "
                + ", ".join(table.column_names)
            )
        # Arrow already stores a list column as offsets + flat values, which is
        # exactly CSR's indptr + indices — take them directly rather than looping
        # over every row in Python.
        indptr, indices = _list_column_to_csr_arrays(table["species_idx"])
        df = table.drop(["species_idx"]).to_pandas()
        del table

        non_species = [c for c in df.columns if c != "species_idx"]
        if df[non_species].isna().any().any():
            nan_cols = [c for c in non_species if df[c].isna().any()]
            raise ValueError("NaNs found in columns: " + ", ".join(nan_cols))

        has_time = _normalize_time_col(df, time_col, no_time)
        coord_cols = ([time_col] if has_time else []) + [lat_col, lon_col]
        if env_cols is None:
            env_cols = [c for c in df.columns
                        if c not in coord_cols and c != "species_idx" and c.startswith("env_")]

        from scipy.sparse import csr_matrix
        N_rows = len(df)
        data = np.ones(indices.size, dtype=np.float32)
        csr = csr_matrix((data, indices, indptr), shape=(N_rows, len(species_cols)))
        species_data = _SparseSpeciesData(csr)

        self._setup_post_load(df, species_data, species_cols, env_cols,
                              time_col, lat_col, lon_col, has_time,
                              euclidean_coords)


def csv_to_sparse_parquet(
    csv_path: str,
    parquet_out: str,
    vocab_out: str,
    time_col: str = "time",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> None:
    df = pd.read_csv(csv_path)
    coord_cols = [c for c in (time_col, lat_col, lon_col) if c in df.columns]
    env_cols = [c for c in df.columns if c not in coord_cols and c.startswith("env_")]
    species_cols = [c for c in df.columns if c not in coord_cols and c not in env_cols]

    species_arr = df[species_cols].values.astype(bool)
    species_idx_per_row = [np.where(row)[0].astype(np.int32) for row in species_arr]

    out_df = df[coord_cols + env_cols].copy()
    out_df["species_idx"] = species_idx_per_row
    out_df.to_parquet(parquet_out, index=False)

    os.makedirs(os.path.dirname(os.path.abspath(vocab_out)) or ".", exist_ok=True)
    with open(vocab_out, "w") as f:
        json.dump(species_cols, f)


class JSDMDataCollator:
    def __init__(self, p=0.15, seed=None):
        self.p = self._canonicalize(p)
        self.generator = (torch.Generator().manual_seed(int(seed))
                          if seed is not None else None)

    @staticmethod
    def _canonicalize(r):
        if isinstance(r, str):
            if r == "unif":
                return "unif:0.0,1.0"
            if r.startswith("unif:"):
                try:
                    lo, hi = [float(x) for x in r[len("unif:"):].split(",")]
                except ValueError:
                    raise ValueError(f"unif range must be 'unif:lo,hi'; got {r!r}") from None
                if not (0.0 <= lo <= hi <= 1.0):
                    raise ValueError(
                        f"unif range must satisfy 0 <= lo <= hi <= 1; got [{lo}, {hi}]"
                    )
                return f"unif:{lo},{hi}"
            if r.startswith("beta:"):
                try:
                    a, b = [float(x) for x in r[len("beta:"):].split(",")]
                except ValueError:
                    raise ValueError(f"beta params must be 'beta:alpha,beta'; got {r!r}") from None
                if not (a > 0.0 and b > 0.0):
                    raise ValueError(f"beta alpha and beta must be > 0; got ({a}, {b})")
                return f"beta:{a},{b}"
            raise ValueError(
                f"mask rate string must be 'unif', 'unif:lo,hi', or 'beta:alpha,beta'; got {r!r}"
            )
        r = float(r)
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"mask rate must be in [0, 1] or 'unif[:lo,hi]' or 'beta:alpha,beta'; got {r}")
        return r

    def _sample_row_rates(self, B, r):
        if isinstance(r, str):
            if r.startswith("unif:"):
                lo, hi = [float(x) for x in r[len("unif:"):].split(",")]
                return torch.rand(B, generator=self.generator) * (hi - lo) + lo
            if r.startswith("beta:"):
                a, b = [float(x) for x in r[len("beta:"):].split(",")]
                if self.generator is not None:
                    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item())
                    rng = np.random.default_rng(seed)
                    return torch.from_numpy(rng.beta(a, b, size=B).astype(np.float32))
                return torch.from_numpy(np.random.beta(a, b, size=B).astype(np.float32))
        return torch.full((B,), float(r))

    @staticmethod
    def _stack(examples):
        return {k: torch.stack([ex[k] for ex in examples]) for k in examples[0]}

    @staticmethod
    def _force_one_masked(masked, generator):
        B, S = masked.shape
        for b in range(B):
            if not masked[b].any():
                masked[b, torch.randint(S, (1,), generator=generator)] = True

    @staticmethod
    def _finalize(batch, masked):
        target_species = batch.pop("target_species")
        target_ids = target_species.long()
        target_ids[masked] = 2
        labels = target_species.clone()
        labels[~masked] = -100

        batch["input_ids"] = target_ids.unsqueeze(-1)
        batch["source_ids"] = batch.pop("source_species").to(torch.uint8)
        batch["labels"] = labels.unsqueeze(-1)
        batch["env_data"] = batch.pop("source_env")
        batch["target_site_idx"] = batch.pop("target_idx").unsqueeze(-1)
        return batch

    def __call__(self, examples):
        batch = self._stack(examples)
        B, S = batch["target_species"].shape

        p_row = self._sample_row_rates(B, self.p)
        probability_matrix = p_row[:, None].expand(B, S)
        masked = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        self._force_one_masked(masked, self.generator)
        return self._finalize(batch, masked)


class _PerBatchSeededCollator(JSDMDataCollator):
    def __init__(self, p, base_seed=0):
        super().__init__(p=p)
        self.base_seed = int(base_seed)

    def _generator(self, batch):
        return torch.Generator().manual_seed(
            self.base_seed + int(batch["target_idx"][0].item()))


class AbsenceMaskCollator(_PerBatchSeededCollator):
    """Mask all absences + p fraction of presences."""

    def __call__(self, examples):
        batch = self._stack(examples)
        target_species = batch["target_species"]
        B, S = target_species.shape
        g = self._generator(batch)

        presence_mask = (target_species == 1) & torch.bernoulli(
            torch.full((B, S), self.p), generator=g
        ).bool()
        masked = (target_species == 0) | presence_mask
        self._force_one_masked(masked, g)
        return self._finalize(batch, masked)


class FixedPValCollator(_PerBatchSeededCollator):
    def __call__(self, examples):
        batch = self._stack(examples)
        B, S = batch["target_species"].shape
        g = self._generator(batch)

        masked = torch.bernoulli(torch.full((B, S), self.p), generator=g).bool()
        self._force_one_masked(masked, g)
        return self._finalize(batch, masked)


def build_val_loaders_fixed_p(dataset, val_indices, dist_info, p_values,
                               batch_size, num_workers=0, base_seed=0):
    from torch.utils.data import DataLoader, Subset
    subset = Subset(dataset, val_indices)
    loaders = []
    for i, p in enumerate(p_values):
        col = FixedPValCollator(p=p, base_seed=base_seed + 1000 * i)
        loaders.append((float(p), DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            collate_fn=col, num_workers=num_workers, pin_memory=True,
            worker_init_fn=seed_worker,
        )))
    return loaders


def compute_dist_info(dataset: "JSDMDataset") -> dict:
    return {
        "site_lats":  torch.as_tensor(dataset.lats,  dtype=torch.float32),
        "site_lons":  torch.as_tensor(dataset.lons,  dtype=torch.float32),
        "site_times": torch.as_tensor(dataset.times, dtype=torch.float32),
        "euclidean":  dataset.euclidean_coords,
        "max_spatial_dist": float(dataset._max_spatial),
        "max_temporal_dist": float(dataset._max_temporal),
    }


def grid_block_split(x, y, n_cells=20, train_frac=0.8, test_frac=0.1, seed=42):

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xi = np.floor((x - x.min()) / (np.ptp(x) + 1e-9) * n_cells).clip(0, n_cells - 1).astype(int)
    yi = np.floor((y - y.min()) / (np.ptp(y) + 1e-9) * n_cells).clip(0, n_cells - 1).astype(int)
    cell_ids = xi * n_cells + yi
    unique_cells = np.unique(cell_ids)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_cells))
    unique_cells = unique_cells[perm]

    n = len(unique_cells)
    n_test = max(1, round(n * test_frac))
    n_val  = max(1, round(n * (1 - train_frac - test_frac)))
    test_cells = set(unique_cells[:n_test])
    val_cells  = set(unique_cells[n_test : n_test + n_val])

    train_idx = np.where(~np.isin(cell_ids, list(test_cells | val_cells)))[0]
    val_idx   = np.where( np.isin(cell_ids, list(val_cells)))[0]
    test_idx  = np.where( np.isin(cell_ids, list(test_cells)))[0]

    n_cells_train = n - n_test - n_val
    print(f"  Grid {n_cells}×{n_cells} | {n} cells → {n_cells_train} train / {n_val} val / {n_test} test cells")
    return train_idx, val_idx, test_idx


def save_splits(path: str, train_idx, val_idx, test_idx, num_rows: int | None = None,
                meta: dict | None = None) -> None:

    payload = {
        "num_rows": int(num_rows) if num_rows is not None else None,
        "meta":     meta or {},
        "train":    [int(x) for x in np.asarray(train_idx).ravel()],
        "val":      [int(x) for x in np.asarray(val_idx).ravel()],
        "test":     [int(x) for x in np.asarray(test_idx).ravel()],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def load_splits(path: str, expected_num_rows: int | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    with open(path) as f:
        payload = json.load(f)
    saved_n = payload.get("num_rows")
    if expected_num_rows is not None and saved_n is not None and saved_n != expected_num_rows:
        raise ValueError(
            f"Split file at {path} was built for {saved_n} rows but the current "
            f"dataset has {expected_num_rows}. Splits are row-index based; the CSV "
            f"must not have been reordered or resized since the split was saved."
        )
    return (
        np.array(payload["train"], dtype=np.int64),
        np.array(payload["val"],   dtype=np.int64),
        np.array(payload["test"],  dtype=np.int64),
    )


def h3_block_split(lats, lons, resolution=2, train_frac=0.8, test_frac=0.1, seed=42):
    
    try:
        import h3 as h3lib
    except ImportError:
        raise ImportError(
            "h3 package required for --fold h3. Install with: uv add h3"
        ) from None
    cells = np.array([h3lib.latlng_to_cell(float(lat), float(lon), resolution)
                      for lat, lon in zip(lats, lons, strict=False)])
    # Label each row by its cell's rank among the sorted unique cells, then work
    # in those integer codes: membership tests over 71M cell *strings* cost
    # minutes, over int codes they are milliseconds. codes[i] == j exactly when
    # cells[i] == unique_cells[j], so the split is unchanged.
    unique_cells, codes = np.unique(cells, return_inverse=True)
    codes = codes.astype(np.int32, copy=False).ravel()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_cells))

    n = len(unique_cells)
    n_test = max(1, round(n * test_frac))
    n_val  = max(1, round(n * (1 - train_frac - test_frac)))
    test_codes = perm[:n_test]
    val_codes  = perm[n_test : n_test + n_val]

    train_idx = np.where(~np.isin(codes, np.concatenate([test_codes, val_codes])))[0]
    val_idx   = np.where( np.isin(codes, val_codes))[0]
    test_idx  = np.where( np.isin(codes, test_codes))[0]

    n_cells_train = n - n_test - n_val
    print(f"  H3 res={resolution} | {n} cells → {n_cells_train} train / {n_val} val / {n_test} test cells")
    return train_idx, val_idx, test_idx


def create_dataloaders(
    csv_path, batch_size=32, num_source_sites=64, num_scale_sites=None,
    p=0.15,
    train_frac=0.8, test_frac=0.1, num_workers=0,
    seed=42, env_cols=None,
    euclidean_coords=False, no_time=False,
    fold_method="random", resolution: int | None = None,
    saved_splits: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    restrict_source_pool_with_saved_splits: bool = True,
    vocab_path: str | None = None,
):

    if vocab_path is not None:
        dataset = JSDMSparseDataset(
            parquet_path=csv_path,
            vocab_path=vocab_path,
            num_source_sites=num_source_sites,
            num_scale_sites=num_scale_sites,
            env_cols=env_cols,
            euclidean_coords=euclidean_coords,
            no_time=no_time,
        )
    else:
        dataset = JSDMDataset(
            csv_path=csv_path,
            num_source_sites=num_source_sites,
            num_scale_sites=num_scale_sites,
            env_cols=env_cols,
            euclidean_coords=euclidean_coords,
            no_time=no_time,
        )

    print("Computing distance info...")
    dist_info = compute_dist_info(dataset)

    
    if saved_splits is not None:
        train_indices, val_indices, test_indices = saved_splits
        train_indices = np.asarray(train_indices, dtype=np.int64)
        val_indices   = np.asarray(val_indices,   dtype=np.int64)
        test_indices  = np.asarray(test_indices,  dtype=np.int64)
        source_pool_restricted = restrict_source_pool_with_saved_splits
        split_origin = "saved"
    elif fold_method == "h3":
        if euclidean_coords:
            raise ValueError("--fold h3 requires real lat/lon coordinates. Use --fold grid for euclidean datasets.")
        if resolution is None:
            resolution = 2
        if not isinstance(resolution, int) or not (0 <= resolution <= 15):
            raise ValueError("--resolution for --fold h3 must be an integer in [0, 15].")
        train_indices, val_indices, test_indices = h3_block_split(
            dataset.lats, dataset.lons,
            resolution=resolution, train_frac=train_frac, test_frac=test_frac, seed=seed,
        )
        split_origin = "h3"
        source_pool_restricted = True
    elif fold_method == "grid":
        if not euclidean_coords:
            raise ValueError("--fold grid is for euclidean/simulated datasets. Use --fold h3 for real lat/lon.")
        if resolution is None:
            resolution = 20
        if not isinstance(resolution, int) or resolution < 1:
            raise ValueError("--resolution for --fold grid must be a positive integer.")
        train_indices, val_indices, test_indices = grid_block_split(
            dataset.lats, dataset.lons,
            n_cells=resolution, train_frac=train_frac, test_frac=test_frac, seed=seed,
        )
        source_pool_restricted = True
        split_origin = "grid"
    else:
        if resolution is not None:
            raise ValueError("--resolution is only valid with --fold {h3,grid}.")
        np.random.seed(seed)
        n = len(dataset)
        indices = np.random.permutation(n)
        n_train = int(n * train_frac)
        n_test  = int(n * test_frac)
        train_indices = indices[:n_train]
        val_indices   = indices[n_train : n - n_test if n_test > 0 else n]
        test_indices  = indices[n - n_test:] if n_test > 0 else np.array([], dtype=int)
        source_pool_restricted = False
        split_origin = "random"

    if source_pool_restricted:
        dataset.source_pool = train_indices

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(dataset, val_indices)
    test_dataset  = torch.utils.data.Subset(dataset, test_indices) if len(test_indices) > 0 else None

    collator = JSDMDataCollator(p=p, seed=seed)

    train_shuffle_gen = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True,
                               worker_init_fn=seed_worker, generator=train_shuffle_gen,
                               persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True,
                               worker_init_fn=seed_worker)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True,
                               worker_init_fn=seed_worker) if test_dataset else None

    print(f"Split ({split_origin}): "
          f"{len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test")

    splits = {"train": train_indices, "val": val_indices, "test": test_indices}
    return train_loader, val_loader, test_loader, dataset, dist_info, splits
