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
import os, subprocess, sys
from datetime import datetime

import numpy as np
import pandas as pd

REPO_ROOT = os.environ.get("REPO_ROOT",
                           os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
COVARIATES = os.path.join(REPO_ROOT, "data_processing", "covariates")
sys.path.insert(0, os.path.join(COVARIATES, "modis_phenology"))
from mod13q1 import extract_modis as sample_modis  # noqa: E402

ENV_DIR   = os.environ.get("ENV_DIR",   os.path.join(REPO_ROOT, "Examples", "env_vars"))
OBS_CSV   = os.environ.get("OBS_CSV",   os.path.join(REPO_ROOT, "lab", "ebutterfly_na_2011_2025_jsdm.csv"))
OUT_CSV   = os.environ.get("OUT_CSV",   os.path.join(REPO_ROOT, "lab", "ebutterfly_na_2011_2025_jsdm_enriched.csv"))
ERA5_RAW  = os.environ.get("ERA5_RAW",  os.path.join(REPO_ROOT, "lab", "_ebutterfly_era5_raw.parquet"))
ERA5_CORR = os.environ.get("ERA5_CORR", os.path.join(REPO_ROOT, "lab", "_ebutterfly_era5_corr.parquet"))

ERA5_EXTRACT = os.path.join(COVARIATES, "era5", "extract_era5_at_points.py")
DEM_VRT      = os.path.join(COVARIATES, "copernicus_dem", "COP30_hh_vsicurl.vrt")
MODIS_DIR    = os.path.join(ENV_DIR, "modis_phenology", "raw")
SOIL_DIR     = os.path.join(ENV_DIR, "soilgrids", "raw")

SOIL_VARS = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt"]

LAPSE_K_PER_M = 0.0065


def sep(title):
    line = "═" * 70
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def log(msg, **kw):
    print(f"  {msg}", flush=True, **kw)


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


def extract_modis(obs):
    sep("MODIS phenology — NDVI / EVI from MOD13Q1 (16-day, 250 m)")
    return sample_modis(obs, MODIS_DIR)


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
