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


def haversine_pairwise(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Pairwise haversine distances in kilometers. Use for real geographic coordinates."""
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    lat1 = lat[:, None]; lat2 = lat[None, :]
    dlat = lat1 - lat2
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (6371.0 * 2.0 * np.arcsin(np.sqrt(a))).astype(np.float32)


def euclidean_pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances. Use for simulated or arbitrary 2D coordinates."""
    coords = np.stack([x, y], axis=1).astype(np.float64)
    return cdist(coords, coords).astype(np.float32)


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
        euclidean_coords: bool = False,
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
            env_cols     = [c for c in df.columns if c not in coord_cols and c.startswith("env_")]
            species_cols = [c for c in df.columns if c not in coord_cols and not c.startswith("env_")]
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
        if euclidean_coords:
            self.spatial_dists = euclidean_pairwise(self.lats, self.lons)
        else:
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
            "target_idx": torch.tensor(idx, dtype=torch.long),
            "source_idx": torch.tensor(source_idx, dtype=torch.long),
        }


class JSDMDataCollator:
    def __init__(self, mlm_probability=0.15, combined_dist=None, blind_threshold=None):
        self.mlm_probability = mlm_probability
        self.combined_dist = combined_dist    # (N_total, N_total) numpy array or None
        self.blind_threshold = blind_threshold  # scalar float or None

    def __call__(self, examples):
        batch = {
            key: torch.stack([ex[key] for ex in examples])
            for key in examples[0].keys()
        }

        target_species = batch["target_species"]  # (B, S)
        source_species = batch["source_species"]  # (B, S, N) float {0, 1}
        B, S = target_species.shape
        N = source_species.shape[-1]

        labels = target_species.clone()

        probability_matrix = torch.full((B, S), self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        for b in range(B):
            if not masked_indices[b].any():
                masked_indices[b, torch.randint(S, (1,))] = True

        # Target token ids: 0=absent, 1=present, 2=mask
        target_ids = target_species.long()  # (B, S)
        indices_replaced = torch.bernoulli(torch.full((B, S), 0.9)).bool() & masked_indices
        target_ids[indices_replaced] = 2  # 90% of masked → mask token; 10% keep original

        # Source ids: convert float {0, 1} to long {0, 1}, then blind nearby sites to 2
        source_ids = source_species.long()  # (B, S, N)
        if self.combined_dist is not None and self.blind_threshold is not None:
            source_idx = batch["source_idx"]   # (B, N)
            target_idx = batch["target_idx"]   # (B,)
            src_idx_np = source_idx.numpy()    # (B, N)
            tgt_idx_np = target_idx.numpy()    # (B,)
            blind_dists = torch.tensor(
                self.combined_dist[tgt_idx_np[:, None], src_idx_np]  # (B, N)
            )
            is_blind = blind_dists <= self.blind_threshold  # (B, N)
            # blind_mask[b, s, n] = True iff species s is masked AND source n is nearby
            blind_mask = masked_indices[:, :, None] & is_blind[:, None, :]  # (B, S, N)
            source_ids[blind_mask] = 2
        else:
            # No proximity info: blind all source entries for masked species
            source_ids[masked_indices[:, :, None].expand_as(source_ids)] = 2

        labels[~masked_indices] = -100

        batch["input_ids"] = target_ids.unsqueeze(-1)   # (B, S, 1) long
        batch["source_ids"] = source_ids                 # (B, S, N) long
        batch["labels"] = labels.unsqueeze(-1)
        batch["env_data"] = batch.pop("source_env")
        batch["target_site_idx"] = batch.pop("target_idx").unsqueeze(-1)

        del batch["target_species"]
        del batch["source_species"]

        return batch


def compute_dist_info(
    spatial_dist: np.ndarray,
    temporal_dist: np.ndarray,
    spatial_scale_km: float,
    temporal_scale_days: float,
    blind_percentile: float = 2.0,
) -> dict:
    """
    Compute normalized pairwise distances and a proximity blind threshold.

    The blind_threshold marks the closest blind_percentile% of site pairs as
    "nearby". During training the collator sets source entries for these nearby
    sites to the mask token so the model cannot trivially copy from them.

    Returns a dict with keys consumed by JSDMDataCollator and JSDMModel.forward.
    """
    combined = np.sqrt(
        (spatial_dist / spatial_scale_km) ** 2 + (temporal_dist / temporal_scale_days) ** 2
    ).astype(np.float32)

    N = len(spatial_dist)
    triu = combined[np.triu_indices(N, k=1)]
    nonzero = triu[triu > 0]
    if len(nonzero) > 0:
        blind_threshold = float(np.percentile(nonzero, blind_percentile))
    else:
        blind_threshold = 0.0
    print(f"  Blind threshold: {blind_threshold:.4f} ({blind_percentile}th percentile of pairwise distances)")

    return {
        "combined_dist": combined,
        "blind_threshold": blind_threshold,
        "spatial_dist_pairwise": torch.tensor(spatial_dist),
        "temporal_dist_pairwise": torch.tensor(temporal_dist),
        "max_spatial_dist": float(spatial_dist.max()),
        "max_temporal_dist": float(temporal_dist.max()),
        "spatial_scale_km": spatial_scale_km,
        "temporal_scale_days": temporal_scale_days,
    }


def create_dataloaders(
    csv_path, batch_size=32, num_source_sites=64, mlm_probability=0.15,
    blind_percentile=2.0,
    train_frac=0.8, test_frac=0.1, num_workers=0,
    seed=42, env_cols=None, spatial_scale_km=None, temporal_scale_days=None,
    euclidean_coords=False,
):
    dataset = JSDMDataset(
        csv_path=csv_path,
        num_source_sites=num_source_sites,
        env_cols=env_cols,
        spatial_scale_km=spatial_scale_km,
        temporal_scale_days=temporal_scale_days,
        euclidean_coords=euclidean_coords,
    )

    print("Computing distance info...")
    dist_info = compute_dist_info(
        spatial_dist=dataset.spatial_dists,
        temporal_dist=dataset.temporal_dists,
        spatial_scale_km=dataset.spatial_scale_km,
        temporal_scale_days=dataset.temporal_scale_days,
        blind_percentile=blind_percentile,
    )

    # seed after distance computation so train/val split matches other baselines
    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)
    n_train = int(n * train_frac)
    n_test  = int(n * test_frac)
    # val gets the remainder: indices[n_train : n - n_test]
    train_dataset = torch.utils.data.Subset(dataset, indices[:n_train])
    val_dataset   = torch.utils.data.Subset(dataset, indices[n_train:n - n_test if n_test > 0 else n])
    test_dataset  = torch.utils.data.Subset(dataset, indices[n - n_test:]) if n_test > 0 else None

    collator = JSDMDataCollator(
        mlm_probability=mlm_probability,
        combined_dist=dist_info["combined_dist"],
        blind_threshold=dist_info["blind_threshold"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True) if test_dataset else None

    n_val = len(val_dataset)
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    return train_loader, val_loader, test_loader, dataset, dist_info
