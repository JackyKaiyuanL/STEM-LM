"""
Enrich eButterfly NA observations with environmental covariates.

Adds columns:
  ERA5 (daily, date-matched, lapse-rate corrected with DEM):
    env_t2m_min, env_t2m_max, env_t2m_mean, env_tp_sum
  MODIS phenology (16-day composite nearest to obs date):
    env_ndvi, env_evi
  SoilGrids 0–5 cm:
    env_soil_bdod, env_soil_cec, env_soil_cfvo, env_soil_clay,
    env_soil_nitrogen, env_soil_phh2o, env_soil_sand, env_soil_silt
  Elevation:
    env_dem    (Copernicus DEM COP30, m)

Output: ${REPO_ROOT}/lab/ebutterfly_na_2011_2025_jsdm_enriched.csv
(meta cols, then env_* cols, then species cols — matches JSDMDataset expectations)

Self-check the ERA5 lapse-rate correction by doing the extraction twice
(raw vs corrected) and verifying the difference equals (era5_elev - dem_elev) * 0.0065.

Requires:
  pip install xarray zarr gcsfs fsspec pyarrow rasterio pandas numpy
  conda install -c conda-forge libgdal-hdf4   # needed for MOD13Q1 HDF4 read
"""
import argparse, glob, math, os, re, subprocess, sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── paths (override any of these via env vars; defaults derive from REPO_ROOT) ──
REPO_ROOT = os.environ.get("REPO_ROOT",
                           os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OBS_CSV   = os.environ.get("OBS_CSV",   os.path.join(REPO_ROOT, "lab", "ebutterfly_na_2011_2025_jsdm.csv"))
OUT_CSV   = os.environ.get("OUT_CSV",   os.path.join(REPO_ROOT, "lab", "ebutterfly_na_2011_2025_jsdm_enriched.csv"))
ERA5_RAW  = os.environ.get("ERA5_RAW",  os.path.join(REPO_ROOT, "lab", "_ebutterfly_era5_raw.parquet"))
ERA5_CORR = os.environ.get("ERA5_CORR", os.path.join(REPO_ROOT, "lab", "_ebutterfly_era5_corr.parquet"))

ERA5_EXTRACT = os.environ.get("ERA5_EXTRACT", os.path.join(REPO_ROOT, "Examples", "env_vars", "era5_land", "extract_era5_at_points.py"))
DEM_VRT      = os.environ.get("DEM_VRT",      os.path.join(REPO_ROOT, "Examples", "env_vars", "copernicus_dem", "COP30_hh_vsicurl.vrt"))
MODIS_DIR    = os.environ.get("MODIS_DIR",    os.path.join(REPO_ROOT, "Examples", "env_vars", "modis_phenology", "raw"))
SOIL_DIR     = os.environ.get("SOIL_DIR",     os.path.join(REPO_ROOT, "Examples", "env_vars", "soilgrids", "raw"))

SOIL_VARS = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt"]

# MODIS sinusoidal projection constants (sphere radius 6371007.181 m)
R_MODIS         = 6371007.181
MAX_X_SIN       = R_MODIS * math.pi            # 20_015_109.354
MAX_Y_SIN       = R_MODIS * math.pi / 2        # 10_007_554.677
TILE_WIDTH_M    = 1_111_950.5197665            # 10° in sinusoidal metres
PIXELS_PER_TILE = 4800                         # 250 m / pixel → 10° / 4800

LAPSE_K_PER_M = 0.0065


# ── utilities ────────────────────────────────────────────────────────────────
def sep(title):
    line = "═" * 70
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def log(msg, **kw):
    print(f"  {msg}", flush=True, **kw)


def latlon_to_sin(lats, lons):
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    x = R_MODIS * lon_r * np.cos(lat_r)
    y = R_MODIS * lat_r
    return x, y


def sin_to_tile_pixel(x, y):
    h   = ((x + MAX_X_SIN) / TILE_WIDTH_M).astype(int)
    v   = ((MAX_Y_SIN - y) / TILE_WIDTH_M).astype(int)
    xin = (x + MAX_X_SIN) - h * TILE_WIDTH_M
    yin = (MAX_Y_SIN - y) - v * TILE_WIDTH_M
    px  = TILE_WIDTH_M / PIXELS_PER_TILE
    col = (xin / px).astype(int).clip(0, PIXELS_PER_TILE - 1)
    row = (yin / px).astype(int).clip(0, PIXELS_PER_TILE - 1)
    return h, v, col, row


def nearest_mod13q1_doy(dt_series):
    """MOD13Q1 composites start every 16 days: DOY 1, 17, 33, …, 353. For each
    obs date, pick the composite DOY whose centre is closest."""
    doy = dt_series.dt.dayofyear.to_numpy()
    composite_doys = np.arange(1, 354, 16)                  # (23,)
    diffs = np.abs(doy[:, None] - composite_doys[None, :])  # (N, 23)
    return composite_doys[diffs.argmin(axis=1)]


# ── ERA5 ─────────────────────────────────────────────────────────────────────
def run_era5(out_path, with_dem):
    cmd = ["python", ERA5_EXTRACT,
           "--obs_csv", OBS_CSV, "--out", out_path,
           "--id_col", "__index__", "--date_col", "time",
           "--vars", "2m_temperature", "total_precipitation",
           "--workers", "8"]
    if with_dem:
        cmd += ["--dem_vrt", DEM_VRT]
    log(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError("ERA5 extract failed")


def validate_era5():
    sep("ERA5 lapse-rate correction — self-check")
    raw  = pd.read_parquet(ERA5_RAW).set_index("id")
    corr = pd.read_parquet(ERA5_CORR).set_index("id")
    common = raw.index.intersection(corr.index)
    raw, corr = raw.loc[common], corr.loc[common]

    delta = (corr["era5_elev_m"] - corr["dem_elev_m"]) * LAPSE_K_PER_M    # expected Δ K
    log(f"DEM elev (m):   min={corr.dem_elev_m.min():.1f}  median={corr.dem_elev_m.median():.1f}  max={corr.dem_elev_m.max():.1f}")
    log(f"ERA5 orog (m):  min={corr.era5_elev_m.min():.1f}  median={corr.era5_elev_m.median():.1f}  max={corr.era5_elev_m.max():.1f}")
    log(f"Δelev = era5 − dem:  mean={(corr.era5_elev_m - corr.dem_elev_m).mean():+.2f} m   |max|={(corr.era5_elev_m - corr.dem_elev_m).abs().max():.1f} m")
    log(f"Expected correction: mean={delta.mean():+.3f} K   |max|={delta.abs().max():.3f} K\n")

    log("Arithmetic check (corrected − raw should equal Δelev × 0.0065 K/m):")
    all_ok = True
    for col in ("2m_temperature_min", "2m_temperature_max", "2m_temperature_mean"):
        diff  = corr[col] - raw[col]
        resid = (diff - delta).abs().max()
        ok = resid < 1e-3
        all_ok &= ok
        log(f"  {col:35s}  max |resid| = {resid:.2e} K   {'✓' if ok else '✗'}")
    log(f"\nArithmetic verdict: {'PASS' if all_ok else 'FAIL'}\n")

    log("Slope of t2m_mean vs DEM elev (expect corrected steeper / more negative):")
    s_raw  = np.polyfit(corr.dem_elev_m,  raw["2m_temperature_mean"],  1)[0] * 1000
    s_corr = np.polyfit(corr.dem_elev_m, corr["2m_temperature_mean"], 1)[0] * 1000
    log(f"  raw       : {s_raw:+.3f} K/km")
    log(f"  corrected : {s_corr:+.3f} K/km  (ELR reference = −6.500 K/km)")


# ── MODIS ────────────────────────────────────────────────────────────────────
def _try_hdf_backend():
    try:
        from pyhdf.SD import SD, SDC       # noqa
        return "pyhdf"
    except ImportError:
        pass
    try:
        import rasterio
        # smoke-test: open one subdataset
        sample = next(iter(glob.glob(f"{MODIS_DIR}/*.hdf")), None)
        if sample is None:
            return None
        sub = f'HDF4_EOS:EOS_GRID:"{sample}":MODIS_Grid_16DAY_250m_500m_VI:250m 16 days NDVI'
        with rasterio.open(sub) as src:
            _ = src.shape
        return "rasterio"
    except Exception as e:
        log(f"rasterio HDF4 test failed: {e}")
        return None


def _read_mod13q1(path, backend):
    if backend == "pyhdf":
        from pyhdf.SD import SD, SDC
        sd = SD(path, SDC.READ)
        ndvi = sd.select("250m 16 days NDVI").get()
        evi  = sd.select("250m 16 days EVI").get()
        sd.end()
        return ndvi, evi
    elif backend == "rasterio":
        import rasterio
        ndvi_sub = f'HDF4_EOS:EOS_GRID:"{path}":MODIS_Grid_16DAY_250m_500m_VI:250m 16 days NDVI'
        evi_sub  = f'HDF4_EOS:EOS_GRID:"{path}":MODIS_Grid_16DAY_250m_500m_VI:250m 16 days EVI'
        with rasterio.open(ndvi_sub) as src: ndvi = src.read(1)
        with rasterio.open(evi_sub)  as src: evi  = src.read(1)
        return ndvi, evi
    raise RuntimeError("no HDF backend")


def extract_modis(obs):
    sep("MODIS phenology — NDVI / EVI from MOD13Q1 (16-day, 250 m)")
    backend = _try_hdf_backend()
    if backend is None:
        print("\n  ERROR: cannot read MOD13Q1 HDF4. Install one of:")
        print("    conda install -c conda-forge libgdal-hdf4")
        print("    pip install pyhdf\n")
        sys.exit(1)
    log(f"HDF backend: {backend}")

    # Build HDF file index: (year, composite_doy, h, v) → path
    pat = re.compile(r"MOD13Q1\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.061\..*\.hdf$")
    hdf_map = {}
    for p in glob.glob(f"{MODIS_DIR}/*.hdf"):
        m = pat.search(p)
        if m:
            y, doy, hh, vv = m.groups()
            hdf_map[(int(y), int(doy), int(hh), int(vv))] = p
    log(f"Indexed {len(hdf_map):,} HDF files under {MODIS_DIR}")

    lats = obs["latitude"].to_numpy()
    lons = obs["longitude"].to_numpy()
    x, y = latlon_to_sin(lats, lons)
    h, v, col, row = sin_to_tile_pixel(x, y)
    dt   = pd.to_datetime(obs["time"])
    year = dt.dt.year.to_numpy().astype(int)
    doy  = nearest_mod13q1_doy(dt)

    keys = list(zip(year, doy, h, v))
    obs_w = obs.copy()
    obs_w["_key"] = pd.Series(keys, index=obs.index)
    obs_w["_row"] = row
    obs_w["_col"] = col

    uniq = obs_w["_key"].unique()
    log(f"Unique (year, composite, h, v) tile-keys for obs: {len(uniq):,}")
    missing_keys = [k for k in uniq if k not in hdf_map]
    if missing_keys:
        log(f"WARNING: {len(missing_keys)} tile-keys have no matching HDF (likely outside NA+Meso bbox)")
        for k in missing_keys[:5]:
            log(f"  example missing: {k}")

    ndvi = np.full(len(obs), np.nan, dtype=np.float32)
    evi  = np.full(len(obs), np.nan, dtype=np.float32)

    grp = obs_w.groupby("_key", sort=False)
    done = 0
    for key, sub in grp:
        path = hdf_map.get(key)
        if path is not None:
            try:
                ndvi_arr, evi_arr = _read_mod13q1(path, backend)
                r = sub["_row"].to_numpy()
                c = sub["_col"].to_numpy()
                v_ndvi = ndvi_arr[r, c].astype(np.float32) / 10000.0  # MODIS scale
                v_evi  = evi_arr[r,  c].astype(np.float32) / 10000.0
                # MOD13Q1 fill = −3000 raw → −0.3 after /10000; drop those
                v_ndvi = np.where(v_ndvi < -0.2, np.nan, v_ndvi)
                v_evi  = np.where(v_evi  < -0.2, np.nan, v_evi)
                ndvi[sub.index] = v_ndvi
                evi[sub.index]  = v_evi
            except Exception as e:
                log(f"  [err reading {os.path.basename(path)}] {e}")
        done += 1
        if done % 200 == 0 or done == len(grp):
            log(f"  processed {done}/{len(grp)} tile-keys")

    n_ndvi = np.isfinite(ndvi).sum()
    n_evi  = np.isfinite(evi).sum()
    log(f"NDVI: {n_ndvi}/{len(obs)} valid   mean={np.nanmean(ndvi):.3f}   range [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
    log(f"EVI : {n_evi }/{len(obs)} valid   mean={np.nanmean(evi):.3f}    range [{np.nanmin(evi):.3f},  {np.nanmax(evi):.3f}]")
    return pd.DataFrame({"env_ndvi": ndvi, "env_evi": evi}, index=obs.index)


# ── SoilGrids + DEM ──────────────────────────────────────────────────────────
def extract_soil_dem(obs):
    sep("SoilGrids (0–5 cm, NA) + Copernicus DEM — static extract")
    import rasterio
    lats = obs["latitude"].to_numpy()
    lons = obs["longitude"].to_numpy()
    coords = list(zip(lons, lats))

    out = {}
    for var in SOIL_VARS:
        path = f"{SOIL_DIR}/{var}_0-5cm_na.tif"
        if not os.path.exists(path):
            log(f"SKIP env_soil_{var}: {path} missing")
            continue
        with rasterio.open(path) as src:
            vals = np.fromiter((s[0] for s in src.sample(coords)),
                               dtype=np.float32, count=len(lats))
        # SoilGrids nodata varies; treat highly-negative as NaN
        vals = np.where(vals < -32000, np.nan, vals)
        out[f"env_soil_{var}"] = vals
        log(f"env_soil_{var:9s}: {np.isfinite(vals).sum()}/{len(lats)} valid   "
            f"mean={np.nanmean(vals):.2f}   range [{np.nanmin(vals):.1f}, {np.nanmax(vals):.1f}]")

    with rasterio.open(DEM_VRT) as src:
        dem_vals = np.fromiter((s[0] for s in src.sample(coords)),
                               dtype=np.float32, count=len(lats))
    out["env_dem"] = dem_vals
    log(f"env_dem    : {np.isfinite(dem_vals).sum()}/{len(lats)} valid   "
        f"mean={np.nanmean(dem_vals):.1f} m   range [{np.nanmin(dem_vals):.1f}, {np.nanmax(dem_vals):.1f}] m")

    return pd.DataFrame(out, index=obs.index)


# ── orchestration ────────────────────────────────────────────────────────────
def main():
    t_start = datetime.now()
    sep(f"Loading {OBS_CSV}")
    obs = pd.read_csv(OBS_CSV)
    log(f"{len(obs):,} rows × {obs.shape[1]} cols")
    log(f"time : {obs['time'].min()} → {obs['time'].max()}")
    log(f"lat  : {obs.latitude.min():.3f} to {obs.latitude.max():.3f}")
    log(f"lon  : {obs.longitude.min():.3f} to {obs.longitude.max():.3f}")

    # ERA5 raw
    sep("ERA5 extraction — raw (no lapse correction)")
    if os.path.exists(ERA5_RAW):
        log(f"{ERA5_RAW} already exists, reusing (delete to re-extract)")
    else:
        run_era5(ERA5_RAW, with_dem=False)

    # ERA5 corrected
    sep("ERA5 extraction — lapse-rate corrected via DEM")
    if os.path.exists(ERA5_CORR):
        log(f"{ERA5_CORR} already exists, reusing (delete to re-extract)")
    else:
        run_era5(ERA5_CORR, with_dem=True)

    validate_era5()

    # Load corrected ERA5 and rename for env_ convention
    era5 = pd.read_parquet(ERA5_CORR).set_index("id").sort_index()
    rename = {
        "2m_temperature_min":            "env_t2m_min",
        "2m_temperature_max":            "env_t2m_max",
        "2m_temperature_mean":           "env_t2m_mean",
        "total_precipitation_sum":       "env_tp_sum",
    }
    era5_df = era5.reindex(columns=list(rename.keys())).rename(columns=rename)

    # MODIS
    modis_df = extract_modis(obs)

    # Soil + DEM
    soil_dem_df = extract_soil_dem(obs)

    # Merge
    sep("Merging everything")
    meta_cols    = ["time", "latitude", "longitude"]
    species_cols = [c for c in obs.columns if c not in meta_cols]
    env_df = pd.concat(
        [era5_df.reset_index(drop=True),
         modis_df.reset_index(drop=True),
         soil_dem_df.reset_index(drop=True)],
        axis=1,
    )
    enriched = pd.concat(
        [obs[meta_cols].reset_index(drop=True),
         env_df,
         obs[species_cols].reset_index(drop=True)],
        axis=1,
    )
    env_cols = [c for c in enriched.columns if c.startswith("env_")]
    log(f"Final shape: {enriched.shape[0]:,} rows × {enriched.shape[1]} cols")
    log(f"env_ columns ({len(env_cols)}):")
    for c in env_cols:
        nvalid = enriched[c].notna().sum()
        log(f"  {c:25s}  {nvalid:,}/{len(enriched):,} valid")

    # NaN diagnostics
    nan_rows = enriched[env_cols].isna().any(axis=1).sum()
    log(f"\nRows with any env NaN: {nan_rows:,} / {len(enriched):,} "
        f"({100*nan_rows/len(enriched):.1f}%)")

    enriched.to_csv(OUT_CSV, index=False)
    log(f"\nWROTE {OUT_CSV}  ({os.path.getsize(OUT_CSV)/1e6:.1f} MB)")
    log(f"Total elapsed: {(datetime.now() - t_start).total_seconds():.1f} s")


if __name__ == "__main__":
    main()
