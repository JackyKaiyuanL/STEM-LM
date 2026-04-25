import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any, Tuple


EARTH_RADIUS_KM = 6371.0


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


def _site_distance_rows_np(lats: np.ndarray, lons: np.ndarray,
                           times: np.ndarray, idx,
                           euclidean: bool) -> Tuple[np.ndarray, np.ndarray]:
    if euclidean:
        dx = lats - lats[idx]; dy = lons - lons[idx]
        spatial = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    else:
        spatial = haversine_pairs_np(lats[idx], lons[idx], lats, lons)
    temporal = np.abs(times - times[idx]).astype(np.float32)
    return spatial, temporal


def _tiled_stats(lats: np.ndarray, lons: np.ndarray, times: np.ndarray,
                 euclidean: bool, device: str = "cpu", tile: int = 4096):
    N = len(lats)
    lats_t = torch.as_tensor(lats, dtype=torch.float64, device=device)
    lons_t = torch.as_tensor(lons, dtype=torch.float64, device=device)
    times_t = torch.as_tensor(times, dtype=torch.float64, device=device)
    max_sp = torch.tensor(0.0, dtype=torch.float64, device=device)
    max_tp = torch.tensor(0.0, dtype=torch.float64, device=device)
    for r0 in range(0, N, tile):
        r1 = min(r0 + tile, N)
        la = lats_t[r0:r1, None]; lb = lats_t[None, :]
        lo_a = lons_t[r0:r1, None]; lo_b = lons_t[None, :]
        if euclidean:
            sp = torch.sqrt((la - lb) ** 2 + (lo_a - lo_b) ** 2)
        else:
            sp = haversine_pairs_torch(la, lo_a, lb, lo_b)
        tp = (times_t[r0:r1, None] - times_t[None, :]).abs()
        max_sp = torch.maximum(max_sp, sp.max())
        max_tp = torch.maximum(max_tp, tp.max())
    return float(max_sp.item()), float(max_tp.item())


def _tiled_spatial_quantile(lats: np.ndarray, lons: np.ndarray,
                            spatial_scale: float, percentile: float,
                            euclidean: bool,
                            device: str = "cpu", tile: int = 4096,
                            n_bins: int = 1_000_000) -> float:
                            
    N = len(lats)
    lats_t = torch.as_tensor(lats, dtype=torch.float64, device=device)
    lons_t = torch.as_tensor(lons, dtype=torch.float64, device=device)
    sp_scale = float(spatial_scale)

    def compute_block(r0, r1):
        la = lats_t[r0:r1, None]; lb = lats_t[None, :]
        lo_a = lons_t[r0:r1, None]; lo_b = lons_t[None, :]
        if euclidean:
            sp = torch.sqrt((la - lb) ** 2 + (lo_a - lo_b) ** 2)
        else:
            sp = haversine_pairs_torch(la, lo_a, lb, lo_b)
        return sp / sp_scale

    cmin = torch.tensor(float("inf"), dtype=torch.float64, device=device)
    cmax = torch.tensor(0.0, dtype=torch.float64, device=device)
    for r0 in range(0, N, tile):
        c = compute_block(r0, min(r0 + tile, N))
        c_pos = c[c > 0]
        if c_pos.numel() > 0:
            cmin = torch.minimum(cmin, c_pos.min())
            cmax = torch.maximum(cmax, c_pos.max())
    cmin_v = float(cmin.item()); cmax_v = float(cmax.item())
    if not np.isfinite(cmin_v) or cmax_v <= 0:
        return 0.0
        

    hist = torch.zeros(n_bins, dtype=torch.int64, device=device)
    total = 0
    span = cmax_v - cmin_v
    for r0 in range(0, N, tile):
        c = compute_block(r0, min(r0 + tile, N))
        c_pos = c[c > 0]
        if c_pos.numel() == 0:
            continue
        bin_idx = ((c_pos - cmin_v) / span * (n_bins - 1)).round().clamp(0, n_bins - 1).long()
        hist += torch.bincount(bin_idx, minlength=n_bins)
        total += int(c_pos.numel())

    if total == 0:
        return 0.0
    target = percentile / 100.0 * total
    cdf = torch.cumsum(hist, dim=0)
    bin_cut = int(torch.searchsorted(cdf, torch.tensor(target, device=device)).item())
    bin_cut = min(bin_cut, n_bins - 1)
    return cmin_v + (bin_cut / (n_bins - 1)) * span


def _resolve_scale(value: Optional[float], fallback: float, name: str) -> float:
    if value is None:
        if fallback <= 0:
            print(f"  Note: {name} auto-set to 1.0 (all pairwise distances are 0 — static dataset)")
            return 1.0
        print(f"  Note: {name} auto-set to {fallback:.3f} (max pairwise distance)")
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
        no_time: bool = False,
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

        has_time = (not no_time) and (time_col in df.columns)
        if not has_time:
            reason = "--no_time" if no_time else f"column '{time_col}' not found"
            print(f"  Time ignored ({reason}) — purely spatial model")

        if has_time:
            col = df[time_col]
            is_stringlike = (col.dtype == "object"
                             or pd.api.types.is_datetime64_any_dtype(col)
                             or pd.api.types.is_string_dtype(col))
            if is_stringlike:
                dt = pd.to_datetime(col)
                df[time_col] = (dt - dt.min()).dt.days.astype(float)
            else:
                df[time_col] = df[time_col].astype(float)

        coord_cols = ([time_col] if has_time else []) + [lat_col, lon_col]
        if env_cols is None:
            env_cols     = [c for c in df.columns if c not in coord_cols and c.startswith("env_")]
            species_cols = [c for c in df.columns if c not in coord_cols and not c.startswith("env_")]
        else:
            species_cols = [c for c in df.columns if c not in coord_cols and c not in env_cols]

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
        self.species_data = df[species_cols].values.astype(np.float32)
        self.env_data = (
            df[env_cols].values.astype(np.float32) if env_cols
            else np.zeros((N, 1), dtype=np.float32)
        )

        self.euclidean_coords = bool(euclidean_coords)
        self.has_time = bool(has_time)

        print(f"Dataset: {N} observations, {self.num_species} species, {self.num_env_vars} env vars")
        need_max = (spatial_scale_km is None) or (has_time and temporal_scale_days is None)
        if need_max:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Computing max pairwise distance (tiled, device={device})...")
            max_sp, max_tp = _tiled_stats(self.lats, self.lons, self.times,
                                          euclidean=self.euclidean_coords, device=device)
        else:
            max_sp = max_tp = 0.0
        self.spatial_scale_km = _resolve_scale(spatial_scale_km, max_sp, "spatial_scale_km")
        self.temporal_scale_days = 1.0 if not has_time else _resolve_scale(
            temporal_scale_days, max_tp, "temporal_scale_days"
        )
        self._max_spatial = max_sp
        self._max_temporal = max_tp if has_time else 0.0
        
        self.source_pool = None
        print("Done.")

    def site_distance_rows(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return _site_distance_rows_np(self.lats, self.lons, self.times, idx,
                                      euclidean=self.euclidean_coords)

    def __len__(self):
        return len(self.species_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        N_total = len(self.species_data)
        N = self.num_source_sites

        target_species = self.species_data[idx]  # (S,)

        sp_row, tp_row = self.site_distance_rows(idx)
        spatial = sp_row / self.spatial_scale_km
        temporal = tp_row / self.temporal_scale_days
        combined_dist = np.sqrt(spatial ** 2 + temporal ** 2)
        combined_dist[idx] = np.inf
        inv_dist = 1.0 / (combined_dist + 1e-6)

        if self.source_pool is not None:
            pool = self.source_pool
            p = inv_dist[pool]
            p = p / p.sum()
            source_idx = pool[np.random.choice(len(pool), size=N, replace=(N > len(pool) - 1), p=p)]
        else:
            probs = inv_dist / inv_dist.sum()
            source_idx = np.random.choice(N_total, size=N, replace=(N > N_total - 1), p=probs)

        source_species = np.ascontiguousarray(self.species_data[source_idx].T)
        source_env = self.env_data[source_idx]
        target_env = self.env_data[idx]

        return {
            "target_species": torch.from_numpy(target_species),
            "source_species": torch.from_numpy(source_species),
            "source_env":     torch.from_numpy(source_env),
            "target_env":     torch.from_numpy(target_env),
            "target_idx":     torch.tensor(idx, dtype=torch.long),
            "source_idx":     torch.from_numpy(source_idx.astype(np.int64)),
        }


def _spatial_blind_dists(site_lats, site_lons,
                         tgt_idx_np, src_idx_np,
                         spatial_scale_km, euclidean=False):
    """(B, N) normalized spatial target→source distances for blinding."""
    la_t = site_lats[tgt_idx_np][:, None]
    lo_t = site_lons[tgt_idx_np][:, None]
    la_s = site_lats[src_idx_np]
    lo_s = site_lons[src_idx_np]
    if euclidean:
        sp = np.sqrt((la_t - la_s) ** 2 + (lo_t - lo_s) ** 2).astype(np.float32)
    else:
        sp = haversine_pairs_np(la_t, lo_t, la_s, lo_s)
    return torch.from_numpy((sp / spatial_scale_km).astype(np.float32))


class JSDMDataCollator:
    def __init__(self, p=0.15,
                 site_lats=None, site_lons=None, site_times=None,
                 spatial_scale_km=1.0, temporal_scale_days=1.0,
                 euclidean=False,
                 blind_threshold=None, mask_token_prob=1.0, seed=None):

        self.p = self._canonicalize(p)
        def _to_np(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        self.site_lats = _to_np(site_lats)
        self.site_lons = _to_np(site_lons)
        self.site_times = _to_np(site_times)
        self.spatial_scale_km = float(spatial_scale_km)
        self.temporal_scale_days = float(temporal_scale_days)
        self.euclidean = bool(euclidean)
        self.blind_threshold = blind_threshold
        self._has_coords = self.site_lats is not None and self.site_lons is not None and self.site_times is not None
        self.mask_token_prob = mask_token_prob
        self.generator = (torch.Generator().manual_seed(int(seed))
                          if seed is not None else None)

    @staticmethod
    def _canonicalize(r):
        if isinstance(r, str):
            if r == "rand":
                return "rand:0.0,1.0"
            if r.startswith("rand:"):
                try:
                    lo, hi = [float(x) for x in r[len("rand:"):].split(",")]
                except Exception:
                    raise ValueError(f"rand range must be 'rand:lo,hi'; got {r!r}")
                if not (0.0 <= lo <= hi <= 1.0):
                    raise ValueError(
                        f"rand range must satisfy 0 <= lo <= hi <= 1; got [{lo}, {hi}]"
                    )
                return f"rand:{lo},{hi}"
            raise ValueError(
                f"mask rate string must be 'rand' or 'rand:lo,hi'; got {r!r}"
            )
        r = float(r)
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"mask rate must be in [0, 1] or 'rand[:lo,hi]'; got {r}")
        return r

    def _sample_row_rates(self, B, r):
        if isinstance(r, str):
            lo, hi = [float(x) for x in r[len("rand:"):].split(",")]
            return torch.rand(B, generator=self.generator) * (hi - lo) + lo
        return torch.full((B,), float(r))

    def __call__(self, examples):
        batch = {
            key: torch.stack([ex[key] for ex in examples])
            for key in examples[0].keys()
        }

        target_species = batch["target_species"]
        source_species = batch["source_species"]
        B, S = target_species.shape
        N = source_species.shape[-1]

        labels = target_species.clone()

        p_row = self._sample_row_rates(B, self.p)
        probability_matrix = p_row[:, None].expand(B, S)
        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        for b in range(B):
            if not masked_indices[b].any():
                masked_indices[b, torch.randint(S, (1,), generator=self.generator)] = True
        
        target_ids = target_species.long()
        if self.mask_token_prob >= 1.0:
            indices_replaced = masked_indices
        else:
            indices_replaced = (
                torch.bernoulli(torch.full((B, S), self.mask_token_prob),
                                generator=self.generator).bool()
                & masked_indices
            )
        target_ids[indices_replaced] = 2

        source_ids = source_species.long()
        if self._has_coords and self.blind_threshold is not None:
            source_idx = batch["source_idx"]
            target_idx = batch["target_idx"]
            src_idx_np = source_idx.numpy()
            tgt_idx_np = target_idx.numpy()
            blind_dists = _spatial_blind_dists(
                self.site_lats, self.site_lons,
                tgt_idx_np, src_idx_np,
                self.spatial_scale_km, euclidean=self.euclidean,
            )
            is_blind = blind_dists <= self.blind_threshold
            blind_mask = masked_indices[:, :, None] & is_blind[:, None, :]
            source_ids[blind_mask] = 2
        else:
            source_ids[masked_indices[:, :, None].expand_as(source_ids)] = 2

        labels[~masked_indices] = -100

        batch["input_ids"] = target_ids.unsqueeze(-1)
        batch["source_ids"] = source_ids
        batch["labels"] = labels.unsqueeze(-1)
        batch["env_data"] = batch.pop("source_env")
        batch["target_site_idx"] = batch.pop("target_idx").unsqueeze(-1)

        del batch["target_species"]
        del batch["source_species"]

        return batch


class FixedPValCollator(JSDMDataCollator):
    def __init__(self, p,
                 site_lats=None, site_lons=None, site_times=None,
                 spatial_scale_km=1.0, temporal_scale_days=1.0, euclidean=False,
                 blind_threshold=None, base_seed=0):
        super().__init__(p=p,
                         site_lats=site_lats, site_lons=site_lons, site_times=site_times,
                         spatial_scale_km=spatial_scale_km,
                         temporal_scale_days=temporal_scale_days,
                         euclidean=euclidean,
                         blind_threshold=blind_threshold,
                         mask_token_prob=1.0)
        self.p = float(p)
        self.base_seed = int(base_seed)

    def __call__(self, examples):
        batch = {k: torch.stack([ex[k] for ex in examples]) for k in examples[0].keys()}
        target_species = batch["target_species"]
        source_species = batch["source_species"]
        B, S = target_species.shape

        seed = self.base_seed + int(batch["target_idx"][0].item())
        g = torch.Generator().manual_seed(seed)

        masked = torch.bernoulli(torch.full((B, S), self.p), generator=g).bool()
        for b in range(B):
            if not masked[b].any():
                masked[b, torch.randint(S, (1,), generator=g)] = True

        target_ids = target_species.long()
        target_ids[masked] = 2

        source_ids = source_species.long()
        if self._has_coords and self.blind_threshold is not None:
            src_idx = batch["source_idx"].numpy()
            tgt_idx = batch["target_idx"].numpy()
            blind_dists = _spatial_blind_dists(
                self.site_lats, self.site_lons,
                tgt_idx, src_idx,
                self.spatial_scale_km, euclidean=self.euclidean,
            )
            is_blind = blind_dists <= self.blind_threshold
            blind_mask = masked[:, :, None] & is_blind[:, None, :]
            source_ids[blind_mask] = 2
        else:
            source_ids[masked[:, :, None].expand_as(source_ids)] = 2

        labels = target_species.clone()
        labels[~masked] = -100

        batch["input_ids"] = target_ids.unsqueeze(-1)
        batch["source_ids"] = source_ids
        batch["labels"] = labels.unsqueeze(-1)
        batch["env_data"] = batch.pop("source_env")
        batch["target_site_idx"] = batch.pop("target_idx").unsqueeze(-1)
        del batch["target_species"]
        del batch["source_species"]
        return batch


def build_val_loaders_fixed_p(dataset, val_indices, dist_info, p_values,
                               batch_size, num_workers=0, base_seed=0):
    from torch.utils.data import DataLoader, Subset
    subset = Subset(dataset, val_indices)
    loaders = []
    for i, p in enumerate(p_values):
        col = FixedPValCollator(
            p=p,
            site_lats=dist_info["site_lats"],
            site_lons=dist_info["site_lons"],
            site_times=dist_info["site_times"],
            spatial_scale_km=dist_info["spatial_scale_km"],
            temporal_scale_days=dist_info["temporal_scale_days"],
            euclidean=dist_info.get("euclidean", False),
            blind_threshold=dist_info["blind_threshold"],
            base_seed=base_seed + 1000 * i,
        )
        loaders.append((float(p), DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            collate_fn=col, num_workers=num_workers, pin_memory=True,
        )))
    return loaders


def auto_blind_percentile(
    dataset: "JSDMDataset",
    n_pairs: int = 100_000,
    seed: int = 0,
    min_pairs_per_bin: int = 50,
) -> Tuple[float, dict]:
    
    rng = np.random.RandomState(seed)
    N = len(dataset)
    i = rng.randint(0, N, n_pairs)
    j = rng.randint(0, N, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    lats = dataset.lats.astype(np.float64)
    lons = dataset.lons.astype(np.float64)
    d_km = haversine_pairs_np(lats[i], lons[i], lats[j], lons[j])

    S = dataset.species_data.astype(bool)
    inter = (S[i] & S[j]).sum(axis=1)
    union = (S[i] | S[j]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        jacc = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    bg = float(jacc.mean())

    d_min, d_max = max(1e-3, float(d_km.min())), float(d_km.max())
    edges = np.logspace(np.log10(max(d_min, 0.1)), np.log10(d_max), 20)
    edges = np.concatenate([[0.0], edges])
    j_bins: List[float] = []
    n_bins: List[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_km >= lo) & (d_km < hi)
        n_bins.append(int(m.sum()))
        j_bins.append(float(jacc[m].mean()) if m.sum() >= min_pairs_per_bin else float("nan"))

    valid = [k for k, n in enumerate(n_bins) if n >= min_pairs_per_bin]
    if not valid:
        raise ValueError("auto blind_percentile: no distance bin has enough pairs.")
    j0 = j_bins[valid[0]]
    if not np.isfinite(j0):
        raise ValueError("auto blind_percentile: nearest-bin Jaccard is undefined.")
    if j0 <= bg:
        raise ValueError(
            f"auto blind_percentile: nearest-bin Jaccard ({j0:.4f}) ≤ background "
            f"({bg:.4f}). The dataset shows no near-range autocorrelation, so "
            f"there's nothing for blinding to block. Pass an explicit "
            f"--blind_percentile instead."
        )

    tau = 0.5 * (j0 + bg)
    d_star = None
    for k in valid:
        if j_bins[k] <= tau:
            d_star = edges[k]
            break
    if d_star is None:
        raise ValueError(
            f"auto blind_percentile: mean Jaccard never drops to τ={tau:.4f} "
            f"within the observed range. The autocorrelation extends past the "
            f"sampled extent — set --blind_percentile manually."
        )

    d_norm = d_km / dataset.spatial_scale_km
    pct = float((d_norm <= d_star / dataset.spatial_scale_km).mean() * 100.0)
    diag = {
        "background_jaccard": bg,
        "nearest_bin_jaccard": j0,
        "tau": tau,
        "d_star_km": float(d_star),
        "d_star_norm": float(d_star / dataset.spatial_scale_km),
        "spatial_scale_km": dataset.spatial_scale_km,
        "bin_edges_km": edges.tolist(),
        "mean_jaccard_by_bin": j_bins,
        "n_pairs_by_bin": n_bins,
    }
    return pct, diag


def compute_dist_info(
    dataset: "JSDMDataset",
    blind_percentile="auto",   # float | "auto"
    tile: int = 4096,
    hist_bins: int = 1_000_000,
) -> dict:
    
    if isinstance(blind_percentile, str):
        if blind_percentile.lower() != "auto":
            raise ValueError(f"blind_percentile must be a float or 'auto'; got {blind_percentile!r}")
        print("  blind_percentile='auto': measuring Jaccard(d) to pick half-decay radius...")
        pct, diag = auto_blind_percentile(dataset)
        print(f"    background Jaccard = {diag['background_jaccard']:.4f}")
        print(f"    nearest-bin Jaccard = {diag['nearest_bin_jaccard']:.4f}")
        print(f"    τ (half-decay)      = {diag['tau']:.4f}")
        print(f"    d* = {diag['d_star_km']:.1f} km  →  percentile = {pct:.3f}%")
        blind_percentile = pct

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Computing blind-threshold percentile (tiled histogram, device={device})...")
    blind_threshold = _tiled_spatial_quantile(
        dataset.lats, dataset.lons,
        spatial_scale=dataset.spatial_scale_km,
        percentile=float(blind_percentile),
        euclidean=dataset.euclidean_coords,
        device=device, tile=tile, n_bins=hist_bins,
    )
    print(f"  Blind threshold: {blind_threshold:.4f} ({blind_percentile:.3f}th percentile)")

    return {
        # Coord arrays — torch tensors so `.to(device)` moves them uniformly.
        "site_lats":  torch.as_tensor(dataset.lats,  dtype=torch.float32),
        "site_lons":  torch.as_tensor(dataset.lons,  dtype=torch.float32),
        "site_times": torch.as_tensor(dataset.times, dtype=torch.float32),
        "euclidean":  dataset.euclidean_coords,
        "blind_threshold": blind_threshold,
        "max_spatial_dist": float(dataset._max_spatial),
        "max_temporal_dist": float(dataset._max_temporal),
        "spatial_scale_km": dataset.spatial_scale_km,
        "temporal_scale_days": dataset.temporal_scale_days,
    }


def grid_block_split(x, y, n_cells=20, train_frac=0.8, test_frac=0.1, seed=42):

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xi = np.floor((x - x.min()) / (x.ptp() + 1e-9) * n_cells).clip(0, n_cells - 1).astype(int)
    yi = np.floor((y - y.min()) / (y.ptp() + 1e-9) * n_cells).clip(0, n_cells - 1).astype(int)
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


def save_splits(path: str, train_idx, val_idx, test_idx, num_rows: Optional[int] = None,
                meta: Optional[dict] = None) -> None:

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


def load_splits(path: str, expected_num_rows: Optional[int] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
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
            "h3 package required for --fold h3. Install with: pip install h3"
        )
    cells = np.array([h3lib.latlng_to_cell(float(lat), float(lon), resolution)
                      for lat, lon in zip(lats, lons)])
    unique_cells = np.unique(cells)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_cells))
    unique_cells = unique_cells[perm]

    n = len(unique_cells)
    n_test = max(1, round(n * test_frac))
    n_val  = max(1, round(n * (1 - train_frac - test_frac)))
    test_cells  = set(unique_cells[:n_test])
    val_cells   = set(unique_cells[n_test : n_test + n_val])

    train_idx = np.where(~np.isin(cells, list(test_cells | val_cells)))[0]
    val_idx   = np.where( np.isin(cells, list(val_cells)))[0]
    test_idx  = np.where( np.isin(cells, list(test_cells)))[0]

    n_cells_train = n - n_test - n_val
    print(f"  H3 res={resolution} | {n} cells → {n_cells_train} train / {n_val} val / {n_test} test cells")
    return train_idx, val_idx, test_idx


def create_dataloaders(
    csv_path, batch_size=32, num_source_sites=64,
    p=0.15,
    blind_percentile="auto",
    train_frac=0.8, test_frac=0.1, num_workers=0,
    seed=42, env_cols=None, spatial_scale_km=None, temporal_scale_days=None,
    euclidean_coords=False, no_time=False,
    fold_method="random", resolution: Optional[int] = None,
    saved_splits: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    restrict_source_pool_with_saved_splits: bool = True,
):
    
    dataset = JSDMDataset(
        csv_path=csv_path,
        num_source_sites=num_source_sites,
        env_cols=env_cols,
        spatial_scale_km=spatial_scale_km,
        temporal_scale_days=temporal_scale_days,
        euclidean_coords=euclidean_coords,
        no_time=no_time,
    )

    print("Computing distance info...")
    dist_info = compute_dist_info(dataset, blind_percentile=blind_percentile)

    
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

    collator = JSDMDataCollator(
        p=p,
        site_lats=dataset.lats,
        site_lons=dataset.lons,
        site_times=dataset.times,
        spatial_scale_km=dataset.spatial_scale_km,
        temporal_scale_days=dataset.temporal_scale_days,
        euclidean=dataset.euclidean_coords,
        blind_threshold=dist_info["blind_threshold"],
        seed=seed,
    )

    train_shuffle_gen = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True,
                               generator=train_shuffle_gen)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=collator, num_workers=num_workers, pin_memory=True) if test_dataset else None

    print(f"Split ({split_origin}): "
          f"{len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test")

    splits = {"train": train_indices, "val": val_indices, "test": test_indices}
    return train_loader, val_loader, test_loader, dataset, dist_info, splits
