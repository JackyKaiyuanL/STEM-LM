"""
Data loading and preprocessing for STEM-LM.

CSV layout: one row per (site, time) observation with columns:
    time (optional), latitude, longitude, env_*, species_*

Dataset emits per-sample dicts consumed by JSDMDataCollator:
    target_species: (S,)     — true 0/1 at target site (becomes labels)
    source_species: (S, N)   — 0/1 at N nearest source sites
    source_env:     (N, E)   — env covariates at source sites
    target_env:     (E,)     — env covariates at target site
    target_idx:     ()       — index of target row in the full CSV
    source_idx:     (N,)     — indices of sampled source rows

Collator applies masked-species augmentation:
    - mask target species with per-class rates p_pres and p_abs → state token 2
      (each can be a fixed float or "rand" for Uniform[0, 1] per row)
    - for masked species, nearby source entries are set to 2 (proximity blind)
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from typing import Optional, List, Dict, Any, Tuple


EARTH_RADIUS_KM = 6371.0


def haversine_pairs_np(lat_a: np.ndarray, lon_a: np.ndarray,
                       lat_b: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    """Broadcasted haversine in km. `a`, `b` are same-shape arrays (deg)."""
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
    """Torch analogue. Accepts degrees. Inputs must broadcast."""
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
    """Distances from a single target row `idx` to every site. Returns (spatial_km, temporal_days)."""
    if euclidean:
        dx = lats - lats[idx]; dy = lons - lons[idx]
        spatial = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    else:
        spatial = haversine_pairs_np(lats[idx], lons[idx], lats, lons)
    temporal = np.abs(times - times[idx]).astype(np.float32)
    return spatial, temporal


def _tiled_stats(lats: np.ndarray, lons: np.ndarray, times: np.ndarray,
                 euclidean: bool, device: str = "cpu", tile: int = 4096):
    """Streaming N×N pass. Returns:
        max_spatial, max_temporal,
        combined_min, combined_max  (with scales applied only *after* max is known,
                                     so we do a two-phase pass)
    This fn only computes max_spatial / max_temporal (phase 1).
    """
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


def _tiled_combined_quantile(lats: np.ndarray, lons: np.ndarray, times: np.ndarray,
                             spatial_scale: float, temporal_scale: float,
                             percentile: float, euclidean: bool,
                             device: str = "cpu", tile: int = 4096,
                             n_bins: int = 1_000_000) -> float:
    """Streaming histogram quantile of the combined normalized distance over
    off-diagonal pairs. Exact to one bin width (≈1e-6 × range at n_bins=1e6).
    """
    N = len(lats)
    lats_t  = torch.as_tensor(lats,  dtype=torch.float64, device=device)
    lons_t  = torch.as_tensor(lons,  dtype=torch.float64, device=device)
    times_t = torch.as_tensor(times, dtype=torch.float64, device=device)
    sp_scale = float(spatial_scale)
    tp_scale = float(temporal_scale)

    def compute_combined_block(r0, r1):
        la = lats_t[r0:r1, None]; lb = lats_t[None, :]
        lo_a = lons_t[r0:r1, None]; lo_b = lons_t[None, :]
        if euclidean:
            sp = torch.sqrt((la - lb) ** 2 + (lo_a - lo_b) ** 2)
        else:
            sp = haversine_pairs_torch(la, lo_a, lb, lo_b)
        tp = (times_t[r0:r1, None] - times_t[None, :]).abs()
        return torch.sqrt((sp / sp_scale) ** 2 + (tp / tp_scale) ** 2)

    # Phase A: min / max of combined (exclude diagonal zeros and strict zeros)
    cmin = torch.tensor(float("inf"), dtype=torch.float64, device=device)
    cmax = torch.tensor(0.0, dtype=torch.float64, device=device)
    for r0 in range(0, N, tile):
        r1 = min(r0 + tile, N)
        c = compute_combined_block(r0, r1)
        c_pos = c[c > 0]
        if c_pos.numel() > 0:
            cmin = torch.minimum(cmin, c_pos.min())
            cmax = torch.maximum(cmax, c_pos.max())
    cmin_v = float(cmin.item()); cmax_v = float(cmax.item())
    if not np.isfinite(cmin_v) or cmax_v <= 0:
        return 0.0

    # Phase B: accumulate histogram, then percentile from CDF
    hist = torch.zeros(n_bins, dtype=torch.int64, device=device)
    total = 0
    span = cmax_v - cmin_v
    for r0 in range(0, N, tile):
        r1 = min(r0 + tile, N)
        c = compute_combined_block(r0, r1)
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

        # Time is optional: disabled explicitly via no_time=True, or absent from CSV
        has_time = (not no_time) and (time_col in df.columns)
        if not has_time:
            reason = "--no_time" if no_time else f"column '{time_col}' not found"
            print(f"  Time ignored ({reason}) — purely spatial model")

        self.half_months: Optional[np.ndarray] = None
        if has_time:
            if df[time_col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[time_col]):
                dt = pd.to_datetime(df[time_col])
                months = dt.dt.month.values
                days   = dt.dt.day.values
                self.half_months = (2 * (months - 1) + (days > 15).astype(np.int64)).astype(np.int64)
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
                "column; the model will train without ecological signal."
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
        # Pairwise distances are never materialized. Max distances (needed for
        # scale auto-set) are computed by a streaming tiled pass on GPU when a
        # scale is not explicitly provided.
        need_max = (spatial_scale_km is None) or (has_time and temporal_scale_days is None)
        if need_max:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Computing max pairwise distance (tiled, device={device})...")
            max_sp, max_tp = _tiled_stats(self.lats, self.lons, self.times,
                                          euclidean=self.euclidean_coords, device=device)
        else:
            max_sp = max_tp = 0.0
        self.spatial_scale_km   = _resolve_scale(spatial_scale_km,   max_sp, "spatial_scale_km")
        self.temporal_scale_days = 1.0 if not has_time else _resolve_scale(
            temporal_scale_days, max_tp, "temporal_scale_days"
        )
        # Cache maxima for downstream consumers (use_temporal etc.)
        self._max_spatial = max_sp
        self._max_temporal = max_tp if has_time else 0.0
        self.source_pool = None  # None = all rows; set to train indices for h3/blocked splits
        print("Done.")

    # Row-wise on-the-fly distances: (N,) arrays from site `idx` to all sites.
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

        source_species = np.ascontiguousarray(self.species_data[source_idx].T)  # (S, N)
        source_env = self.env_data[source_idx]                                   # (N, E)
        target_env = self.env_data[idx]                                          # (E,)

        return {
            "target_species": torch.from_numpy(target_species),
            "source_species": torch.from_numpy(source_species),
            "source_env":     torch.from_numpy(source_env),
            "target_env":     torch.from_numpy(target_env),
            "target_idx":     torch.tensor(idx, dtype=torch.long),
            "source_idx":     torch.from_numpy(source_idx.astype(np.int64)),
        }


def _combined_blind_dists(site_lats, site_lons, site_times,
                          tgt_idx_np, src_idx_np,
                          spatial_scale_km, temporal_scale_days,
                          euclidean=False):
    """Compute combined normalized (B, N) distances for collator blinding.

    site_lats/lons/times are full (N_total,) numpy arrays. tgt_idx_np is (B,),
    src_idx_np is (B, N). Returns torch.FloatTensor (B, N).
    """
    la_t = site_lats[tgt_idx_np][:, None]
    lo_t = site_lons[tgt_idx_np][:, None]
    ti_t = site_times[tgt_idx_np][:, None]
    la_s = site_lats[src_idx_np]
    lo_s = site_lons[src_idx_np]
    ti_s = site_times[src_idx_np]
    if euclidean:
        sp = np.sqrt((la_t - la_s) ** 2 + (lo_t - lo_s) ** 2).astype(np.float32)
    else:
        sp = haversine_pairs_np(la_t, lo_t, la_s, lo_s)
    tp = np.abs(ti_t - ti_s).astype(np.float32)
    combined = np.sqrt((sp / spatial_scale_km) ** 2 + (tp / temporal_scale_days) ** 2)
    return torch.from_numpy(combined.astype(np.float32))


class JSDMDataCollator:
    def __init__(self, p_pres=0.15, p_abs=0.15,
                 site_lats=None, site_lons=None, site_times=None,
                 spatial_scale_km=1.0, temporal_scale_days=1.0,
                 euclidean=False,
                 blind_threshold=None, mask_token_prob=1.0, seed=None):
        # Per-class mask rates. Each is either a float in [0, 1] or a
        # 'rand[:lo,hi]' string → sample Uniform[lo, hi] per row.
        # 'rand' alone is shorthand for 'rand:0.0,1.0'. Examples:
        #   p_pres=0.15,           p_abs=0.15           → fixed 15% random mask
        #   p_pres='rand',         p_abs='rand'         → independent per-class rates per row
        #   p_pres='rand:0.3,1.0', p_abs=1.0            → presence-only, varying completeness in [0.3, 1]
        #   p_pres=1.0,            p_abs=1.0            → always 100% mask
        self.p_pres = self._canonicalize(p_pres)
        self.p_abs  = self._canonicalize(p_abs)
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
        self.blind_threshold = blind_threshold  # scalar float or None
        self._has_coords = self.site_lats is not None and self.site_lons is not None and self.site_times is not None
        # Fraction of masked positions that get the [MASK] token (id=2); the rest
        # keep their original value (BERT's 10% keep-original trick). Default 1.0:
        # BERT's rationale (pretrain/finetune gap, bidirectional conditioning at
        # non-masked positions) doesn't apply here — our train task = inference
        # task, [MASK] marks "predict here" in both, and keep-original creates an
        # eval leak at any mask rate < 1.
        self.mask_token_prob = mask_token_prob
        # Private generator isolates mask sampling from model init / dropout /
        # DataLoader shuffle, so two runs with the same seed produce byte-identical
        # mask patterns regardless of model architecture.
        self.generator = (torch.Generator().manual_seed(int(seed))
                          if seed is not None else None)

    @staticmethod
    def _canonicalize(r):
        """Return a canonical form: float in [0,1] or a 'rand:lo,hi' string."""
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
            # r is 'rand:lo,hi' (canonical form)
            lo, hi = [float(x) for x in r[len("rand:"):].split(",")]
            return torch.rand(B, generator=self.generator) * (hi - lo) + lo
        return torch.full((B,), float(r))

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

        # Per-row mask rates, then per-position probability picked by class.
        p_pres_row = self._sample_row_rates(B, self.p_pres)          # (B,)
        p_abs_row  = self._sample_row_rates(B, self.p_abs)           # (B,)
        is_pres = target_species.bool()                               # (B, S)
        probability_matrix = torch.where(
            is_pres, p_pres_row[:, None], p_abs_row[:, None]
        ).expand(B, S)
        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        for b in range(B):
            if not masked_indices[b].any():
                masked_indices[b, torch.randint(S, (1,), generator=self.generator)] = True

        # Target token ids: 0=absent, 1=present, 2=mask
        target_ids = target_species.long()  # (B, S)
        if self.mask_token_prob >= 1.0:
            indices_replaced = masked_indices
        else:
            indices_replaced = (
                torch.bernoulli(torch.full((B, S), self.mask_token_prob),
                                generator=self.generator).bool()
                & masked_indices
            )
        target_ids[indices_replaced] = 2  # mask_token_prob of masked → [MASK]; rest keep original

        # Source ids: convert float {0, 1} to long {0, 1}, then blind nearby sites to 2
        source_ids = source_species.long()  # (B, S, N)
        if self._has_coords and self.blind_threshold is not None:
            source_idx = batch["source_idx"]   # (B, N)
            target_idx = batch["target_idx"]   # (B,)
            src_idx_np = source_idx.numpy()    # (B, N)
            tgt_idx_np = target_idx.numpy()    # (B,)
            blind_dists = _combined_blind_dists(
                self.site_lats, self.site_lons, self.site_times,
                tgt_idx_np, src_idx_np,
                self.spatial_scale_km, self.temporal_scale_days,
                euclidean=self.euclidean,
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


class FixedPValCollator(JSDMDataCollator):
    """Deterministic fixed-p mask collator for validation.

    Mask rate is a single scalar p applied to both presences and absences.
    Randomness is seeded deterministically from batch content (first
    target_idx), so the same batch produces identical masks across epochs —
    the val AUC trajectory becomes a noise-free signal of model quality
    rather than drifting with per-batch mask sampling.
    """

    def __init__(self, p,
                 site_lats=None, site_lons=None, site_times=None,
                 spatial_scale_km=1.0, temporal_scale_days=1.0, euclidean=False,
                 blind_threshold=None, base_seed=0):
        super().__init__(p_pres=p, p_abs=p,
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
            blind_dists = _combined_blind_dists(
                self.site_lats, self.site_lons, self.site_times,
                tgt_idx, src_idx,
                self.spatial_scale_km, self.temporal_scale_days,
                euclidean=self.euclidean,
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
    """Return list of (p, DataLoader) using FixedPValCollator for each p.

    Used for apples-to-apples val AUC logging and checkpoint selection:
    mask-rate sampling noise is removed, so `best_model.pt` selection
    reflects true model improvements rather than lucky-p epochs.
    """
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


def compute_dist_info(
    dataset: "JSDMDataset",
    blind_percentile: float = 2.0,
    tile: int = 4096,
    hist_bins: int = 1_000_000,
) -> dict:
    """Coord-only dist_info. Carries site coords (N-vectors) + scales + the
    blind threshold percentile (computed via streaming GPU histogram).

    No N×N matrix is materialized. Consumed by collators and JSDMModel.forward,
    which compute per-batch distances on the fly.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Computing blind-threshold percentile (tiled histogram, device={device})...")
    blind_threshold = _tiled_combined_quantile(
        dataset.lats, dataset.lons, dataset.times,
        spatial_scale=dataset.spatial_scale_km,
        temporal_scale=dataset.temporal_scale_days,
        percentile=blind_percentile,
        euclidean=dataset.euclidean_coords,
        device=device, tile=tile, n_bins=hist_bins,
    )
    print(f"  Blind threshold: {blind_threshold:.4f} ({blind_percentile}th percentile)")

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
    """
    Split by a regular n_cells x n_cells grid on arbitrary 2D (x, y) coordinates.
    For Euclidean/simulated datasets where H3 is meaningless.
    Each cell assigned entirely to one split.
    """
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
    """Persist train/val/test row indices as JSON for reproducible reloading.

    `num_rows` is stored as a sanity check — load_splits rejects the file if
    the CSV it is being applied to has a different row count.
    `meta` can carry fold method, resolution, seed, etc. for provenance.
    """
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
    """Load train/val/test indices saved by `save_splits`."""
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


def h3_block_split(lats, lons, resolution=4, train_frac=0.8, test_frac=0.1, seed=42):
    """
    Split observation indices into train/val/test by H3 spatial blocks.
    Each H3 cell at the given resolution is assigned entirely to one split.
    Returns (train_indices, val_indices, test_indices) as numpy arrays.
    """
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


def h3_time_block_split(lats, lons, half_months, resolution=3,
                        train_frac=0.8, test_frac=0.1, seed=42):
    """Spatiotemporal block on (H3 cell, half-month).

    Independently partitions H3 cells and half-month bins (0..23) into
    train/val/test. A row is:
      - train if both its cell AND its half-month are train-assigned,
      - test  if either its cell OR its half-month is test-assigned,
      - val   otherwise (at least one axis is val, neither is test).
    Val therefore covers three regimes: spatial-only (new cell, seen season),
    seasonal-only (seen cell, new season), and combined (new cell, new season).
    """
    try:
        import h3 as h3lib
    except ImportError:
        raise ImportError("h3 package required for --fold h3. Install with: pip install h3")
    cells = np.array([h3lib.latlng_to_cell(float(lat), float(lon), resolution)
                      for lat, lon in zip(lats, lons)])
    hm = np.asarray(half_months, dtype=np.int64)

    def partition(uniq, seed_shift):
        rng = np.random.RandomState(seed + seed_shift)
        arr = uniq.copy(); rng.shuffle(arr)
        n = len(arr)
        n_te = max(1, round(n * test_frac))
        n_va = max(1, round(n * (1 - train_frac - test_frac)))
        return set(arr[:n_te].tolist()), set(arr[n_te:n_te+n_va].tolist()), set(arr[n_te+n_va:].tolist())

    test_c, val_c, train_c = partition(np.unique(cells), 0)
    test_h, val_h, train_h = partition(np.unique(hm),    1)

    c_tr = np.isin(cells, list(train_c)); c_va = np.isin(cells, list(val_c)); c_te = np.isin(cells, list(test_c))
    h_tr = np.isin(hm,    list(train_h)); h_va = np.isin(hm,    list(val_h)); h_te = np.isin(hm,    list(test_h))

    train_idx = np.where(c_tr & h_tr)[0]
    test_idx  = np.where(c_te | h_te)[0]
    val_idx   = np.where(~(c_tr & h_tr) & ~(c_te | h_te))[0]

    n_spatial  = int((c_va & h_tr).sum())
    n_seasonal = int((c_tr & h_va).sum())
    n_combined = int((c_va & h_va).sum())
    print(f"  H3 res={resolution} × half-month | cells {len(np.unique(cells))} "
          f"({len(train_c)}/{len(val_c)}/{len(test_c)}) | "
          f"half-months {len(np.unique(hm))} "
          f"({len(train_h)}/{len(val_h)}/{len(test_h)})")
    print(f"  rows: train {len(train_idx)} | val {len(val_idx)} "
          f"(spatial {n_spatial} + seasonal {n_seasonal} + combined {n_combined}) "
          f"| test {len(test_idx)}")
    return train_idx, val_idx, test_idx


def create_dataloaders(
    csv_path, batch_size=32, num_source_sites=64,
    p_pres=0.15, p_abs=0.15,
    blind_percentile=2.0,
    train_frac=0.8, test_frac=0.1, num_workers=0,
    seed=42, env_cols=None, spatial_scale_km=None, temporal_scale_days=None,
    euclidean_coords=False, no_time=False,
    fold_method="random", resolution: Optional[int] = None,
    saved_splits: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    restrict_source_pool_with_saved_splits: bool = True,
):
    """Build train/val/test dataloaders.

    If `saved_splits=(train_idx, val_idx, test_idx)` is given, those indices
    are used directly and fold_method is ignored. Source pool is restricted
    to train indices by default (treat as blocked CV); set
    `restrict_source_pool_with_saved_splits=False` to keep the full pool.

    Returns:
        train_loader, val_loader, test_loader, dataset, dist_info, splits
    where `splits = {"train": np.ndarray, "val": np.ndarray, "test": np.ndarray}`.
    """
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

    # Split into train / val / test
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
        if dataset.half_months is not None:
            train_indices, val_indices, test_indices = h3_time_block_split(
                dataset.lats, dataset.lons, dataset.half_months,
                resolution=resolution, train_frac=train_frac, test_frac=test_frac, seed=seed,
            )
            split_origin = "h3_time"
        elif no_time:
            train_indices, val_indices, test_indices = h3_block_split(
                dataset.lats, dataset.lons,
                resolution=resolution, train_frac=train_frac, test_frac=test_frac, seed=seed,
            )
            split_origin = "h3"
        else:
            raise ValueError(
                "--fold h3 with time enabled requires parseable dates in the 'time' column "
                "(e.g. 'YYYY-MM-DD') so month/day can be extracted for spatiotemporal blocking. "
                "Either pass --no_time or convert the time column to a standard date string."
            )
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
    else:  # random (default)
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
        p_pres=p_pres,
        p_abs=p_abs,
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
