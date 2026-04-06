"""
Data loading and preprocessing for ST-JSDM (citizen-science variant).
"""

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from typing import Optional, List, Dict, Any
import networkx as nx


def haversine_pairwise(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return (6371.0 * 2.0 * np.arcsin(np.sqrt(a))).astype(np.float32)


def haversine_to_point(
    lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float
) -> np.ndarray:
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    lat0 = math.radians(float(lat0_deg))
    lon0 = math.radians(float(lon0_deg))
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin(dlon / 2.0) ** 2
    return (6371.0 * 2.0 * np.arcsin(np.sqrt(a))).astype(np.float32)


def _estimate_max_spatial_dist(lat_deg: np.ndarray, lon_deg: np.ndarray) -> float:
    lat_min, lat_max = float(lat_deg.min()), float(lat_deg.max())
    lon_min, lon_max = float(lon_deg.min()), float(lon_deg.max())
    corners = np.array(
        [
            [lat_min, lon_min],
            [lat_min, lon_max],
            [lat_max, lon_min],
            [lat_max, lon_max],
        ],
        dtype=np.float32,
    )
    corner_dists = haversine_pairwise(corners[:, 0], corners[:, 1])
    return float(corner_dists.max())


def _pairwise_quantile_spatial(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    quantile: float = 0.95,
    sample_size: int = 1024,
) -> float:
    n = len(lat_deg)
    if n < 2:
        return 1.0
    take = min(n, sample_size)
    idx = np.random.choice(n, size=take, replace=False)
    dists = haversine_pairwise(lat_deg[idx], lon_deg[idx]).reshape(-1)
    dists = dists[dists > 0]
    if dists.size == 0:
        return 1.0
    return float(np.quantile(dists, quantile))


def _pairwise_quantile_temporal(
    times: np.ndarray,
    quantile: float = 0.95,
    sample_size: int = 1024,
) -> float:
    n = len(times)
    if n < 2:
        return 1.0
    take = min(n, sample_size)
    idx = np.random.choice(n, size=take, replace=False)
    sample = times[idx]
    dists = np.abs(sample[:, None] - sample[None, :]).reshape(-1)
    dists = dists[dists > 0]
    if dists.size == 0:
        return 1.0
    return float(np.quantile(dists, quantile))


def circular_doy_pairwise(doy: np.ndarray) -> np.ndarray:
    """Pairwise circular day-of-year distances in [0, 182]."""
    diff = np.abs(doy[:, None] - doy[None, :])
    return np.minimum(diff, 365 - diff).astype(np.float32)


def _resolve_scale(value: Optional[float], fallback: float, name: str) -> float:
    if value is None:
        if fallback <= 0:
            raise ValueError(f"Cannot auto-scale {name}: max pairwise distance is {fallback}")
        return float(fallback)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return float(value)


class JSDMDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        num_source_sites: int = 64,
        time_col: str = "time",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        doy_col: Optional[str] = "doy",
        env_cols: Optional[List[str]] = None,
        spatial_scale_km: Optional[float] = None,
        temporal_scale_days: Optional[float] = None,
        time_window_days: Optional[float] = None,
        sampling_strategy: str = "nearest",
    ):
        super().__init__()
        self.num_source_sites = num_source_sites
        df = pd.read_csv(csv_path)

        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(
                "NaNs found in columns: " + ", ".join(nan_cols)
                + ". Please impute or drop missing values before training."
            )

        # Parse time and extract day-of-year.
        # For datetime strings → elapsed days from first observation + DOY from calendar.
        # For numeric time with a separate doy column → use that column.
        # Otherwise → no seasonal structure (doy stays zero).
        doy_coords = np.zeros(len(df), dtype=np.float32)
        if df[time_col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[time_col]):
            dt = pd.to_datetime(df[time_col])
            doy_coords = (dt.dt.dayofyear - 1).values.astype(np.float32)  # 0-indexed, 0–364
            df[time_col] = (dt - dt.min()).dt.days.astype(float)
        else:
            df[time_col] = df[time_col].astype(float)
            if doy_col is not None and doy_col in df.columns:
                doy_coords = df[doy_col].values.astype(np.float32)

        coord_cols = [time_col, lat_col, lon_col]
        if doy_col and doy_col in df.columns:
            coord_cols.append(doy_col)

        if env_cols is None:
            env_cols = []
            species_cols = []
            for c in df.columns:
                if c in coord_cols:
                    continue
                vals = df[c].dropna().unique()
                if set(vals).issubset({0, 1, 0.0, 1.0}):
                    species_cols.append(c)
                else:
                    env_cols.append(c)
        else:
            species_cols = [c for c in df.columns if c not in coord_cols and c not in env_cols]

        self.species_cols = species_cols
        self.env_cols = env_cols
        self.num_species = len(species_cols)
        self.num_env_vars = len(env_cols) if env_cols else 1

        self.coords = df[[lat_col, lon_col]].values.astype(np.float32)
        self.lats = self.coords[:, 0]
        self.lons = self.coords[:, 1]
        self.times = df[time_col].values.astype(np.float32)
        self.doys = doy_coords
        self.species_data = df[species_cols].values.astype(np.float32)
        self.env_data = (
            df[env_cols].values.astype(np.float32) if env_cols
            else np.zeros((len(df), 1), dtype=np.float32)
        )
        self.env_mean = self.env_data.mean(axis=0)
        self.env_std = self.env_data.std(axis=0)
        self.env_std = np.where(self.env_std > 0, self.env_std, 1.0)
        self.env_norm = (self.env_data - self.env_mean) / self.env_std

        N = len(df)
        print(f"Dataset: {N} observations, {self.num_species} species, {self.num_env_vars} env vars")
        has_doy = doy_coords.any()
        print(f"Seasonal encoding: {'yes (DOY extracted)' if has_doy else 'no (time column is not a date)'}")

        self.max_spatial_dist = _estimate_max_spatial_dist(self.lats, self.lons)
        self.max_temporal_dist = float(self.times.max() - self.times.min())
        self.max_doy_dist = 182.0
        spatial_scale_default = _pairwise_quantile_spatial(self.lats, self.lons)
        temporal_scale_default = _pairwise_quantile_temporal(self.times)
        self.spatial_scale_km = _resolve_scale(
            spatial_scale_km, spatial_scale_default, "spatial_scale_km"
        )
        self.temporal_scale_days = _resolve_scale(
            temporal_scale_days, temporal_scale_default, "temporal_scale_days"
        )

        self.time_window_days = time_window_days
        self.sampling_strategy = sampling_strategy
        self.time_sorted_idx = np.argsort(self.times)
        self.time_sorted = self.times[self.time_sorted_idx]

    def __len__(self):
        return len(self.species_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        N_total = len(self.species_data)
        N = self.num_source_sites

        target_species = self.species_data[idx]

        t0 = float(self.times[idx])
        if self.time_window_days is not None and self.time_window_days > 0:
            left = np.searchsorted(self.time_sorted, t0 - self.time_window_days, side="left")
        else:
            left = 0
        right = np.searchsorted(self.time_sorted, t0, side="left")
        candidates = self.time_sorted_idx[left:right]

        candidates = candidates[candidates != idx]
        if candidates.size == 0:
            candidates = np.array([idx], dtype=np.int64)

        cand_spatial = haversine_to_point(
            self.lats[candidates], self.lons[candidates], self.lats[idx], self.lons[idx]
        )
        cand_temporal = (t0 - self.times[candidates]).astype(np.float32)
        cand_doy = np.abs(self.doys[candidates] - self.doys[idx]).astype(np.float32)
        cand_doy = np.minimum(cand_doy, 365.0 - cand_doy)
        env_diff = self.env_norm[candidates] - self.env_norm[idx]
        env_dist = np.linalg.norm(env_diff, axis=1) / math.sqrt(self.num_env_vars)

        spatial_scaled = cand_spatial / self.spatial_scale_km
        temporal_scaled = cand_temporal / self.temporal_scale_days
        doy_scaled = cand_doy / 182.0
        combined_dist = np.sqrt(
            spatial_scaled ** 2 + temporal_scaled ** 2 + env_dist ** 2 + doy_scaled ** 2
        )

        if self.sampling_strategy == "nearest":
            if candidates.size >= N:
                topk = np.argpartition(combined_dist, N - 1)[:N]
                source_idx = candidates[topk]
            else:
                extra = np.random.choice(
                    candidates, size=N - candidates.size, replace=True
                )
                source_idx = np.concatenate([candidates, extra], axis=0)
        elif self.sampling_strategy == "weighted":
            inv_dist = 1.0 / (combined_dist + 1e-6)
            probs = inv_dist / inv_dist.sum()
            source_idx = np.random.choice(
                candidates, size=N, replace=(N > candidates.size), p=probs
            )
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")

        source_spatial = haversine_to_point(
            self.lats[source_idx], self.lons[source_idx], self.lats[idx], self.lons[idx]
        )
        source_temporal = (t0 - self.times[source_idx]).astype(np.float32)
        source_doy = np.abs(self.doys[source_idx] - self.doys[idx]).astype(np.float32)
        source_doy = np.minimum(source_doy, 365.0 - source_doy)

        return {
            "target_species": torch.tensor(target_species, dtype=torch.float32),
            "source_species": torch.tensor(self.species_data[source_idx].T, dtype=torch.float32),
            "source_env":     torch.tensor(self.env_data[source_idx], dtype=torch.float32),
            "target_env":     torch.tensor(self.env_data[idx], dtype=torch.float32),
            "target_idx":     torch.tensor(idx, dtype=torch.long),
            "source_idx":     torch.tensor(source_idx, dtype=torch.long),
            "source_spatial_dist":  torch.tensor(source_spatial, dtype=torch.float32),
            "source_temporal_dist": torch.tensor(source_temporal, dtype=torch.float32),
            "source_doy_dist":      torch.tensor(source_doy, dtype=torch.float32),
            "source_time":          torch.tensor(self.times[source_idx], dtype=torch.float32),
            "target_time":          torch.tensor(self.times[idx], dtype=torch.float32),
        }


class JSDMDataCollator:
    def __init__(self, mlm_probability=0.15, mask_value=-1.0):
        self.mlm_probability = mlm_probability
        self.mask_value = mask_value

    def __call__(self, examples):
        batch = {
            key: torch.stack([ex[key] for ex in examples])
            for key in examples[0].keys()
        }

        target_species = batch["target_species"]  # (B, S)
        source_species = batch["source_species"]  # (B, S, N)
        B, S = target_species.shape
        N = source_species.shape[-1]

        labels = target_species.clone()

        probability_matrix = torch.full((B, S), self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        for b in range(B):
            if not masked_indices[b].any():
                masked_indices[b, torch.randint(S, (1,))] = True

        # Token ids: 0=absent, 1=present, 2=mask
        target_ids = target_species.long()
        indices_replaced = torch.bernoulli(torch.full((B, S), 0.9)).bool() & masked_indices
        target_ids[indices_replaced] = 2  # 10% of masked positions keep original

        for b in range(B):
            source_species[b, masked_indices[b], :] = self.mask_value

        labels[~masked_indices] = -100

        batch["input_ids"]       = target_ids.unsqueeze(-1)   # (B, S, 1)
        batch["source_ids"]      = source_species
        batch["labels"]          = labels.unsqueeze(-1)
        batch["env_data"]        = batch.pop("source_env")
        batch["target_site_idx"] = batch.pop("target_idx").unsqueeze(-1)

        del batch["target_species"]
        del batch["source_species"]

        return batch


def build_st_clusters(
    spatial_coords,
    temporal_coords,
    doy_coords,
    threshold=5.0,
    spatial_scale_km: Optional[float] = None,
    temporal_scale_days: Optional[float] = None,
    spatial_dist: Optional[np.ndarray] = None,
    temporal_dist: Optional[np.ndarray] = None,
    doy_dist: Optional[np.ndarray] = None,
):
    N = len(spatial_coords)
    if spatial_dist is None:
        spatial_dist = haversine_pairwise(spatial_coords[:, 0], spatial_coords[:, 1])
    else:
        spatial_dist = spatial_dist.astype(np.float32)
    if temporal_dist is None:
        temporal_dist = cdist(
            temporal_coords.reshape(-1, 1), temporal_coords.reshape(-1, 1)
        ).astype(np.float32)
    else:
        temporal_dist = temporal_dist.astype(np.float32)
    if doy_dist is None:
        doy_dist = circular_doy_pairwise(doy_coords)
    else:
        doy_dist = doy_dist.astype(np.float32)

    spatial_scale_km = _resolve_scale(
        spatial_scale_km, float(spatial_dist.max()), "spatial_scale_km"
    )
    temporal_scale_days = _resolve_scale(
        temporal_scale_days, float(temporal_dist.max()), "temporal_scale_days"
    )
    combined = np.sqrt(
        (spatial_dist / spatial_scale_km) ** 2
        + (temporal_dist / temporal_scale_days) ** 2
    )

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if combined[i, j] <= threshold:
                G.add_edge(i, j)

    cluster_dict = {
        i: nodes for i, nodes in enumerate(list(nx.connected_components(G)))
    }
    cluster_labels = torch.zeros(N, dtype=torch.long)
    for cid, sites in cluster_dict.items():
        for s in sites:
            cluster_labels[s] = cid

    in_cluster_spatial  = torch.zeros(N)
    in_cluster_temporal = torch.zeros(N)
    in_cluster_doy      = torch.zeros(N)
    for cid, sites in cluster_dict.items():
        sites_list = list(sites)
        if len(sites_list) > 1:
            center_s = spatial_coords[sites_list].mean(axis=0)
            center_t = temporal_coords[sites_list].mean()
            # Circular mean of DOY: use mean angle on unit circle
            doy_rad   = doy_coords[sites_list] * (2 * np.pi / 365)
            center_doy_rad = np.arctan2(np.sin(doy_rad).mean(), np.cos(doy_rad).mean())
            center_doy = float(center_doy_rad * 365 / (2 * np.pi)) % 365
            for s in sites_list:
                in_cluster_spatial[s] = float(haversine_pairwise(
                    np.array([spatial_coords[s, 0], center_s[0]]),
                    np.array([spatial_coords[s, 1], center_s[1]])
                )[0, 1])
                in_cluster_temporal[s] = float(abs(temporal_coords[s] - center_t))
                diff = abs(float(doy_coords[s]) - center_doy)
                in_cluster_doy[s] = float(min(diff, 365 - diff))

    return {
        "cluster_dict":              cluster_dict,
        "cluster_labels":            cluster_labels,
        "in_cluster_spatial_dist":   in_cluster_spatial,
        "in_cluster_temporal_dist":  in_cluster_temporal,
        "in_cluster_doy_dist":       in_cluster_doy,
        "spatial_dist_pairwise":     torch.tensor(spatial_dist),
        "temporal_dist_pairwise":    torch.tensor(temporal_dist),
        "doy_dist_pairwise":         torch.tensor(doy_dist),
        "temporal_coords":           torch.tensor(temporal_coords, dtype=torch.float32),
        "doy_coords":                torch.tensor(doy_coords, dtype=torch.float32),
        "max_spatial_dist":          float(spatial_dist.max()),
        "max_temporal_dist":         float(temporal_dist.max()),
        "max_doy_dist":              float(doy_dist.max()),
        "num_clusters":              len(cluster_dict),
        "spatial_scale_km":          spatial_scale_km,
        "temporal_scale_days":       temporal_scale_days,
    }


def create_dataloaders(
    csv_path, batch_size=32, num_source_sites=64, mlm_probability=0.15,
    mask_value=-1.0, train_frac=0.8, num_workers=0,
    seed=42, env_cols=None, spatial_scale_km=None, temporal_scale_days=None,
    time_window_days=None, sampling_strategy="nearest",
):
    np.random.seed(seed)
    dataset = JSDMDataset(
        csv_path=csv_path,
        num_source_sites=num_source_sites,
        env_cols=env_cols,
        spatial_scale_km=spatial_scale_km,
        temporal_scale_days=temporal_scale_days,
        time_window_days=time_window_days,
        sampling_strategy=sampling_strategy,
    )

    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(n * train_frac)

    train_dataset = torch.utils.data.Subset(dataset, indices[:split])
    val_dataset   = torch.utils.data.Subset(dataset, indices[split:])

    collator = JSDMDataCollator(
        mlm_probability=mlm_probability,
        mask_value=mask_value,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collator, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              collate_fn=collator, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, dataset
