"""
Data loading and preprocessing for ST-JSDM.

Mirrors GPN-star's data pipeline with corrected axis mapping:

    GPN-star                              →  ST-JSDM
    ──────────────────────────────────────────────────────────
    GenomeMSA: (L positions, N species)    →  CSV: (N_total rows, S species)
    get_msa_batch → (B, L, N)             →  get_site_batch → (B, S, N)

    tokenize_function:                     →  tokenize_function:
      Sample 20 target species from clades    Target = 1 site (the row to predict)
      input_ids = msa subsampled (B,L,T)      input_ids = target row (B, S, T=1)
      source_ids = full msa (B,L,N)           source_ids = sampled sites (B, S, N)

    DataCollator:                          →  DataCollator:
      Mask 15% of L positions                 Mask 15% of S species
      80% → mask token, 10% random, 10% keep  90% → mask_value, 10% keep
      Mask source_ids in same clade           Mask source_ids in same ST cluster
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from typing import Optional, List, Dict, Any
import networkx as nx


def haversine_pairwise(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """
    Compute pairwise haversine distances in kilometers.
    """
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    lat1 = lat[:, None]
    lat2 = lat[None, :]
    dlat = lat1 - lat2
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return (6371.0 * c).astype(np.float32)


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
        env_cols: Optional[List[str]] = None,
        spatial_scale_km: Optional[float] = None,
        temporal_scale_days: Optional[float] = None,
    ):
        super().__init__()
        self.num_source_sites = num_source_sites
        df = pd.read_csv(csv_path)

        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(
                "NaNs found in columns: "
                + ", ".join(nan_cols)
                + ". Please impute or drop missing values before training."
            )

        if df[time_col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
            df[time_col] = (df[time_col] - df[time_col].min()).dt.days.astype(float)
        else:
            df[time_col] = df[time_col].astype(float)

        coord_cols = [time_col, lat_col, lon_col]
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
        self.species_data = df[species_cols].values.astype(np.float32)
        self.env_data = (
            df[env_cols].values.astype(np.float32) if env_cols
            else np.zeros((len(df), 1), dtype=np.float32)
        )

        N = len(df)
        print(f"Dataset: {N} observations, {self.num_species} species, {self.num_env_vars} env vars")
        print("Computing pairwise distances...")
        self.spatial_dists = haversine_pairwise(self.lats, self.lons)
        self.temporal_dists = cdist(
            self.times.reshape(-1, 1), self.times.reshape(-1, 1)
        ).astype(np.float32)
        self.spatial_scale_km = _resolve_scale(
            spatial_scale_km, float(self.spatial_dists.max()), "spatial_scale_km"
        )
        self.temporal_scale_days = _resolve_scale(
            temporal_scale_days, float(self.temporal_dists.max()), "temporal_scale_days"
        )
        print("Done.")

    def __len__(self):
        return len(self.species_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        N_total = len(self.species_data)
        N = self.num_source_sites

        target_species = self.species_data[idx]  # (S,)

        spatial = self.spatial_dists[idx] / self.spatial_scale_km
        temporal = self.temporal_dists[idx] / self.temporal_scale_days
        combined_dist = np.sqrt(spatial ** 2 + temporal ** 2)
        combined_dist[idx] = np.inf
        inv_dist = 1.0 / (combined_dist + 1e-6)
        probs = inv_dist / inv_dist.sum()
        source_idx = np.random.choice(N_total, size=N, replace=(N > N_total - 1), p=probs)

        source_species = self.species_data[source_idx].T  # (S, N)
        source_env = self.env_data[source_idx]             # (N, E)

        return {
            "target_species": torch.tensor(target_species, dtype=torch.float32),
            "source_species": torch.tensor(source_species, dtype=torch.float32),
            "source_env": torch.tensor(source_env, dtype=torch.float32),
            "target_env": torch.tensor(self.env_data[idx], dtype=torch.float32),  # (E,) — target site covariates/effort
            "target_idx": torch.tensor(idx, dtype=torch.long),
            "source_idx": torch.tensor(source_idx, dtype=torch.long),
        }


class JSDMDataCollator:
    def __init__(self, mlm_probability=0.15, mask_value=-1.0, cluster_labels=None):
        self.mlm_probability = mlm_probability
        self.mask_value = mask_value
        self.cluster_labels = cluster_labels

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

        # Convert to discrete token ids for target embedding: 0=absent, 1=present, 2=mask
        # Source ids remain float (masked with self.mask_value below)
        target_ids = target_species.long()  # (B, S)
        indices_replaced = torch.bernoulli(torch.full((B, S), 0.9)).bool() & masked_indices
        target_ids[indices_replaced] = 2  # mask token; 10% of masked positions keep original token

        if self.cluster_labels is not None:
            source_idx = batch["source_idx"]
            target_idx = batch["target_idx"]
            for b in range(B):
                target_cluster = self.cluster_labels[target_idx[b]]
                for n in range(N):
                    if self.cluster_labels[source_idx[b, n]] == target_cluster:
                        source_species[b, masked_indices[b], n] = self.mask_value
        else:
            for b in range(B):
                source_species[b, masked_indices[b], :] = self.mask_value

        labels[~masked_indices] = -100

        batch["input_ids"] = target_ids.unsqueeze(-1)  # (B, S, 1) long
        batch["source_ids"] = source_species
        batch["labels"] = labels.unsqueeze(-1)
        batch["env_data"] = batch.pop("source_env")
        batch["target_site_idx"] = batch.pop("target_idx").unsqueeze(-1)

        del batch["target_species"]
        del batch["source_species"]

        return batch


def build_st_clusters(
    spatial_coords,
    temporal_coords,
    threshold=5.0,
    spatial_scale_km: Optional[float] = None,
    temporal_scale_days: Optional[float] = None,
    spatial_dist: Optional[np.ndarray] = None,
    temporal_dist: Optional[np.ndarray] = None,
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

    spatial_scale_km = _resolve_scale(
        spatial_scale_km, float(spatial_dist.max()), "spatial_scale_km"
    )
    temporal_scale_days = _resolve_scale(
        temporal_scale_days, float(temporal_dist.max()), "temporal_scale_days"
    )
    combined = np.sqrt(
        (spatial_dist / spatial_scale_km) ** 2 + (temporal_dist / temporal_scale_days) ** 2
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

    in_cluster_spatial = torch.zeros(N)
    in_cluster_temporal = torch.zeros(N)
    for cid, sites in cluster_dict.items():
        sites_list = list(sites)
        if len(sites_list) > 1:
            center_s = spatial_coords[sites_list].mean(axis=0)
            center_t = temporal_coords[sites_list].mean()
            for s in sites_list:
                # haversine from site to cluster centroid (km), consistent with pairwise spatial_dist
                in_cluster_spatial[s] = float(haversine_pairwise(
                    np.array([spatial_coords[s, 0], center_s[0]]),
                    np.array([spatial_coords[s, 1], center_s[1]])
                )[0, 1])
                in_cluster_temporal[s] = float(abs(temporal_coords[s] - center_t))

    return {
        "cluster_dict": cluster_dict,
        "cluster_labels": cluster_labels,
        "in_cluster_spatial_dist": in_cluster_spatial,
        "in_cluster_temporal_dist": in_cluster_temporal,
        "spatial_dist_pairwise": torch.tensor(spatial_dist),
        "temporal_dist_pairwise": torch.tensor(temporal_dist),
        "max_spatial_dist": float(spatial_dist.max()),
        "max_temporal_dist": float(temporal_dist.max()),
        "num_clusters": len(cluster_dict),
        "spatial_scale_km": spatial_scale_km,
        "temporal_scale_days": temporal_scale_days,
    }


def create_dataloaders(
    csv_path, batch_size=32, num_source_sites=64, mlm_probability=0.15,
    cluster_threshold=5.0, mask_value=-1.0, train_frac=0.8, num_workers=0,
    seed=42, env_cols=None, spatial_scale_km=None, temporal_scale_days=None,
):
    np.random.seed(seed)
    dataset = JSDMDataset(
        csv_path=csv_path,
        num_source_sites=num_source_sites,
        env_cols=env_cols,
        spatial_scale_km=spatial_scale_km,
        temporal_scale_days=temporal_scale_days,
    )

    print("Building spatial-temporal clusters...")
    cluster_info = build_st_clusters(
        dataset.coords,
        dataset.times,
        threshold=cluster_threshold,
        spatial_scale_km=dataset.spatial_scale_km,
        temporal_scale_days=dataset.temporal_scale_days,
        spatial_dist=dataset.spatial_dists,
        temporal_dist=dataset.temporal_dists,
    )
    print(f"  {cluster_info['num_clusters']} clusters from {len(dataset)} sites")

    n = len(dataset)
    indices = np.random.permutation(n)
    split = int(n * train_frac)

    train_dataset = torch.utils.data.Subset(dataset, indices[:split])
    val_dataset = torch.utils.data.Subset(dataset, indices[split:])

    collator = JSDMDataCollator(mlm_probability=mlm_probability, mask_value=mask_value,
                                 cluster_labels=cluster_info["cluster_labels"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collator, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, dataset, cluster_info
