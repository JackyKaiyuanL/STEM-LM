"""
Add daily ERA5 (t2m min/max/mean + tp_sum, lapse-corrected) and MODIS NDVI/EVI
to the static grid for one or more chosen dates.

For each date, writes one CSV `monarch_grid_<DATE>.csv` with the 15-col env
schema STEM-LM expects (4 ERA5 daily + 2 MODIS + 8 soil + 1 DEM).

Usage:
  python enrich_grid_daily.py \
      --static_grid static_grid_na.csv \
      --dates 2025-04-15 2025-07-15
"""
import argparse, glob, math, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = os.environ.get("REPO_ROOT",
                           os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODIS_DIR    = os.path.join(REPO_ROOT, "Examples", "env_vars", "modis_phenology", "raw")

STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BBOX_N, BBOX_S = 72.0, 7.0
BBOX_W, BBOX_E = -170.0, -50.0
LAPSE_RATE_K_PER_M = 0.0065
G_STD = 9.80665
ERA5_VARS = ["2m_temperature", "total_precipitation"]
ERA5_OROG_NC = os.path.expanduser("~/.cache/era5_orog_na.nc")

R_MODIS         = 6371007.181
MAX_X_SIN       = R_MODIS * math.pi
MAX_Y_SIN       = R_MODIS * math.pi / 2
TILE_WIDTH_M    = 1_111_950.5197665
PIXELS_PER_TILE = 4800


def latlon_to_sin(lats, lons):
    lat_r = np.radians(lats); lon_r = np.radians(lons)
    return R_MODIS * lon_r * np.cos(lat_r), R_MODIS * lat_r


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
    starts = np.array([1 + 16 * i for i in range(23)])
    out = pd.DataFrame({"year": dt_series.dt.year, "doy": dt_series.dt.dayofyear})
    out["nearest"] = out["doy"].apply(
        lambda d: int(starts[np.argmin(np.abs(starts - d))]))
    return out


def _read_mod13q1(path):
    from pyhdf.SD import SD, SDC
    sd = SD(path, SDC.READ)
    ndvi = sd.select("250m 16 days NDVI").get()
    evi  = sd.select("250m 16 days EVI").get()
    sd.end()
    return ndvi, evi


def extract_modis(grid_df, date_str):
    print(f"  MODIS NDVI/EVI for {date_str} ...", flush=True)

    obs = grid_df[["latitude", "longitude"]].copy()
    obs["date"] = pd.to_datetime(date_str)
    snap = nearest_mod13q1_doy(obs["date"])
    obs["composite_year"], obs["composite_doy"] = snap["year"], snap["nearest"]

    x, y = latlon_to_sin(obs.latitude.to_numpy(), obs.longitude.to_numpy())
    h, v, col, row = sin_to_tile_pixel(x, y)
    obs["h"], obs["v"], obs["col"], obs["row"] = h, v, col, row

    hdf_map = {}
    for p in glob.glob(f"{MODIS_DIR}/*.hdf"):
        fn = os.path.basename(p)
        try:
            year = int(fn.split(".")[1][1:5])
            doy  = int(fn.split(".")[1][5:8])
            hh   = int(fn.split(".")[2][1:3])
            vv   = int(fn.split(".")[2][4:6])
            hdf_map[(year, doy, hh, vv)] = p
        except Exception:
            continue

    obs["env_ndvi"] = np.nan
    obs["env_evi"]  = np.nan

    keys = list(set(zip(obs.composite_year, obs.composite_doy, obs.h, obs.v)))
    print(f"    {len(keys)} unique (year, doy, h, v) tiles to read", flush=True)
    for k in keys:
        path = hdf_map.get(k)
        if path is None:
            continue
        ndvi_arr, evi_arr = _read_mod13q1(path)
        m = ((obs.composite_year == k[0]) & (obs.composite_doy == k[1]) &
             (obs.h == k[2]) & (obs.v == k[3]))
        rows_local = obs.loc[m, "row"].to_numpy()
        cols_local = obs.loc[m, "col"].to_numpy()
        obs.loc[m, "env_ndvi"] = ndvi_arr[rows_local, cols_local].astype(np.float32) / 10000.0
        obs.loc[m, "env_evi"]  = evi_arr [rows_local, cols_local].astype(np.float32) / 10000.0

    return obs[["env_ndvi", "env_evi"]].reset_index(drop=True)


def _ensure_era5_orog():
    if os.path.exists(ERA5_OROG_NC) and os.path.getsize(ERA5_OROG_NC) > 0:
        return
    import cdsapi
    os.makedirs(os.path.dirname(ERA5_OROG_NC), exist_ok=True)
    tmp = ERA5_OROG_NC + ".part"
    print(f"Downloading ERA5 invariant orography → {ERA5_OROG_NC}")
    cdsapi.Client(quiet=True, progress=False).retrieve(
        "reanalysis-era5-single-levels",
        {"variable": "geopotential", "year": "2020", "month": "01",
         "day": "01", "time": "00:00", "product_type": "reanalysis",
         "data_format": "netcdf", "download_format": "unarchived",
         "area": [BBOX_N, BBOX_W, BBOX_S, BBOX_E]}, tmp)
    os.replace(tmp, ERA5_OROG_NC)


def sample_era5_orog(lats, lons):
    _ensure_era5_orog()
    ds = xr.open_dataset(ERA5_OROG_NC)
    z = "z" if "z" in ds.data_vars else list(ds.data_vars)[0]
    elev = (ds[z] / G_STD).astype(np.float32)
    for d in list(elev.dims):
        if d not in ("latitude", "longitude"):
            elev = elev.isel({d: 0})
    if float(elev.longitude.min()) >= 0:
        lons_q = np.where(lons < 0, lons + 360.0, lons)
    else:
        lons_q = lons
    return elev.sel(
        latitude=xr.DataArray(lats, dims="obs"),
        longitude=xr.DataArray(lons_q, dims="obs"),
        method="nearest",
    ).to_numpy().astype(np.float32)


def fetch_era5_daily(store, lats, lons, date_str, dem_elev, era5_elev):
    is_0_360 = float(store.longitude.values.min()) >= 0
    W = BBOX_W + 360 if (is_0_360 and BBOX_W < 0) else BBOX_W
    E = BBOX_E + 360 if (is_0_360 and BBOX_E < 0) else BBOX_E
    lats_native = store.latitude.values
    lat_slice = slice(BBOX_N, BBOX_S) if lats_native[0] > lats_native[-1] else slice(BBOX_S, BBOX_N)

    d = pd.Timestamp(date_str)
    month_start = pd.Timestamp(year=d.year, month=d.month, day=1)
    month_end   = month_start + pd.offsets.MonthEnd(1)
    ds = store[ERA5_VARS].sel(time=slice(str(month_start.date()), str(month_end.date())))
    ds = ds.sel(latitude=lat_slice, longitude=slice(W, E))

    daily = xr.Dataset()
    t2m = ds["2m_temperature"]
    daily["2m_temperature_min"]  = t2m.resample(time="1D").min()
    daily["2m_temperature_max"]  = t2m.resample(time="1D").max()
    daily["2m_temperature_mean"] = t2m.resample(time="1D").mean()
    daily["total_precipitation_sum"] = ds["total_precipitation"].resample(time="1D").sum()
    daily = daily.compute()

    lons_q = np.where(is_0_360 & (lons < 0), lons + 360.0, lons).astype(np.float64)
    sampled = daily.sel(
        latitude=xr.DataArray(lats, dims="obs"),
        longitude=xr.DataArray(lons_q, dims="obs"),
        time=xr.DataArray(np.full(len(lats), np.datetime64(date_str), "datetime64[ns]"), dims="obs"),
        method="nearest",
    )

    delta = (era5_elev - dem_elev).astype(np.float32) * LAPSE_RATE_K_PER_M
    out = pd.DataFrame({
        "env_t2m_min":  sampled["2m_temperature_min"].to_numpy().astype(np.float32) + delta,
        "env_t2m_max":  sampled["2m_temperature_max"].to_numpy().astype(np.float32) + delta,
        "env_t2m_mean": sampled["2m_temperature_mean"].to_numpy().astype(np.float32) + delta,
        "env_tp_sum":   sampled["total_precipitation_sum"].to_numpy().astype(np.float32),
    })
    return out


def process_one(date_str, grid, store, lats, lons, dem_elev, era5_elev, output_dir):
    print(f"\n=== {date_str} ===", flush=True)
    era5 = fetch_era5_daily(store, lats, lons, date_str, dem_elev, era5_elev)
    modis = extract_modis(grid, date_str)

    out = grid.copy()
    for c in ["env_t2m_min", "env_t2m_max", "env_t2m_mean", "env_tp_sum"]:
        out[c] = era5[c].to_numpy()
    out["env_ndvi"] = modis["env_ndvi"].to_numpy()
    out["env_evi"]  = modis["env_evi"].to_numpy()
    out["time"] = date_str

    env_order = ["env_t2m_min", "env_t2m_max", "env_t2m_mean", "env_tp_sum",
                 "env_ndvi", "env_evi",
                 "env_soil_bdod", "env_soil_cec", "env_soil_cfvo", "env_soil_clay",
                 "env_soil_nitrogen", "env_soil_phh2o", "env_soil_sand", "env_soil_silt",
                 "env_dem"]
    out = out[["time", "latitude", "longitude"] + env_order]

    out_path = os.path.join(output_dir, f"monarch_grid_{date_str}.csv")
    out.to_csv(out_path, index=False)
    n_nan = out[env_order].isna().sum().sum()
    print(f"  → {out_path}  ({len(out):,} rows, {n_nan} NaN cells)", flush=True)


def main(static_grid, dates, output_dir, workers):
    grid = pd.read_csv(static_grid)
    print(f"Loaded static grid: {len(grid):,} cells × {grid.shape[1]} cols")

    os.makedirs(output_dir, exist_ok=True)
    lats = grid["latitude"].to_numpy()
    lons = grid["longitude"].to_numpy()
    dem_elev = grid["env_dem"].to_numpy().astype(np.float32)

    print(f"Sampling ERA5 orography once at {len(grid):,} grid cells...")
    era5_elev = sample_era5_orog(lats, lons)
    delta = era5_elev - dem_elev
    print(f"  Δelev (era5 − site): mean={np.nanmean(delta):+.1f}  "
          f"|max|={np.nanmax(np.abs(delta)):.1f} m  "
          f"→ lapse range [{np.nanmin(delta)*LAPSE_RATE_K_PER_M:+.2f}, "
          f"{np.nanmax(delta)*LAPSE_RATE_K_PER_M:+.2f}] K")

    print("Opening ARCO-ERA5 Zarr...")
    store = xr.open_zarr(STORE, consolidated=True, storage_options={"token": "anon"})
    missing = [v for v in ERA5_VARS if v not in store.data_vars]
    if missing:
        raise SystemExit(f"ERROR: variables not in store: {missing}")
    print(f"  vars OK: {ERA5_VARS}")

    n_workers = min(workers, len(dates))
    print(f"Dispatching {len(dates)} dates across {n_workers} workers")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(lambda d: process_one(d, grid, store, lats, lons,
                                           dem_elev, era5_elev, output_dir),
                    dates))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--static_grid", default="static_grid_na.csv")
    p.add_argument("--dates", nargs="+", required=True,
                   help="One or more YYYY-MM-DD")
    p.add_argument("--output_dir", default=".")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent date-processing threads")
    a = p.parse_args()
    main(a.static_grid, a.dates, a.output_dir, a.workers)
