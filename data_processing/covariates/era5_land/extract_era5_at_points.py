"""
Extract daily ERA5 climate stats at observation (lat, lon, date) points from
Google Cloud ARCO-ERA5 — public Zarr mirror, no auth, no CDS queue, 0.25° grid.

Outputs one parquet row per input observation:
  id, lat, lon, date, <var>_min/_max/_mean/_sum...

Strategy: obs are grouped by (year, month). For each month, one Zarr subset is
loaded (NA+Meso bbox, hourly), resampled to daily, then vectorised-sampled at
every obs point in that month via xarray nearest-neighbour indexing. Months run
concurrently via ThreadPoolExecutor (gcsfs + zarr are thread-safe).

Requires:
  pip install xarray zarr gcsfs fsspec pandas pyarrow rasterio

Usage:
  python extract_era5_at_points.py \\
    --obs_csv ${REPO_ROOT}/lab/eButterfly/ebutterfly_na_2011_2025_jsdm.csv \\
    --out ebutterfly_na_era5.parquet \\
    --id_col __index__ --lat_col latitude --lon_col longitude --date_col time \\
    --dem_vrt ${REPO_ROOT}/Examples/env_vars/dem/COP30_hh.vrt \\
    --workers 12

With --dem_vrt set, 2m temperature and dewpoint are corrected by the dry-adiabatic
lapse rate (6.5 K/km) using true site elevation from the Copernicus DEM vs ERA5's
coarse-grid orography:
    t_corrected = t_raw + (elev_era5_orog - elev_dem_site) * 0.0065
"""
import argparse, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

# ARCO-ERA5 (Analysis-Ready, Cloud-Optimized ERA5) — Google Research public mirror.
# 0.25° × 0.25°, hourly, 1940-present. See https://github.com/google-research/arco-era5
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# NA + Mesoamerica bbox (matches eButterfly / eBird clip)
BBOX_N, BBOX_S = 72.0, 7.0
BBOX_W, BBOX_E = -170.0, -50.0

DEFAULT_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
]

LAPSE_RATE_K_PER_M = 0.0065       # dry adiabatic, standard meteorological default
G_STD              = 9.80665      # m/s², for geopotential → elevation conversion
ORO_CANDIDATE_NAMES = ("geopotential_at_surface", "surface_geopotential", "z")


def _open_store():
    # Public bucket — no credentials needed. gcsfs is auto-detected by xarray.
    return xr.open_zarr(STORE, consolidated=True,
                        storage_options={"token": "anon"})


def _lon_convention(store_ds):
    """Return ('is_0_360', west, east, lat_slice) for the ARCO store."""
    lons = store_ds.longitude.values
    lats = store_ds.latitude.values
    is_0_360 = float(lons.min()) >= 0
    W = BBOX_W + 360 if (is_0_360 and BBOX_W < 0) else BBOX_W
    E = BBOX_E + 360 if (is_0_360 and BBOX_E < 0) else BBOX_E
    lat_slice = slice(BBOX_N, BBOX_S) if lats[0] > lats[-1] else slice(BBOX_S, BBOX_N)
    return is_0_360, W, E, lat_slice


def _ensure_era5_orog_nc(path):
    """Download ERA5 invariant orography for NA+Meso bbox if not cached.

    ARCO-ERA5's in-store geopotential_at_surface is an ocean-only diagnostic
    (NaN over land), so we fetch the real invariant field from CDS once.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run `pip install cdsapi` and configure "
              "~/.cdsapirc per https://cds.climate.copernicus.eu/how-to-api",
              file=sys.stderr)
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    print(f"Downloading ERA5 invariant orography (one-off, ~1 MB) → {path}")
    cdsapi.Client(quiet=True, progress=False).retrieve(
        "reanalysis-era5-single-levels",
        {
            "variable": "geopotential",
            "year": "2020", "month": "01", "day": "01", "time": "00:00",
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [BBOX_N, BBOX_W, BBOX_S, BBOX_E],
        },
        tmp,
    )
    os.replace(tmp, path)


def _era5_orography_elev(store_ds):
    """Return ERA5 surface elevation (metres) as a 2D (lat, lon) DataArray.

    Reduces any non-(lat, lon) dims by selecting index 0, then converts
    geopotential (m²/s²) → metres via /g. Picks the first matching candidate
    name that exists in the store.
    """
    for name in ORO_CANDIDATE_NAMES:
        if name not in store_ds.data_vars and name not in store_ds.coords:
            continue
        da = store_ds[name]
        # Reduce any extra dims (time, level, pressure_level, etc.) to 2D.
        keep = {"latitude", "longitude"}
        for d in list(da.dims):
            if d not in keep:
                da = da.isel({d: 0})
        print(f"  ERA5 orography variable: {name}  dims={dict(da.sizes)}")
        elev = (da / G_STD).astype(np.float32)
        elev.name = "era5_elev_m"
        return elev
    raise KeyError(
        f"No ERA5 orography variable found; tried {ORO_CANDIDATE_NAMES}. "
        f"Available (first 30): {sorted(list(store_ds.data_vars))[:30]}"
    )


def _sample_dem(dem_path, lats, lons):
    """Bulk-sample a DEM at (lat, lon) pairs via rasterio.

    - Dedups coords (rounded to 6 decimals, lossless within DEM's 30 m cell) so
      we only call `src.sample` once per unique location. eBird-scale datasets
      collapse 10–100× on this step.
    - Tile-sorts the uniques by 1° bin so GDAL's tile cache stays warm.
    - Single-threaded; rasterio handles are not thread-safe and VSICURL cache
      is per-handle, so multi-threading loses more than it gains.
    """
    import os as _os
    _os.environ.setdefault("GDAL_CACHEMAX", "2048")
    _os.environ.setdefault("VSI_CACHE",     "TRUE")
    _os.environ.setdefault("VSI_CACHE_SIZE",                   str(256 * 1024 * 1024))
    _os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE",         str(1024 * 1024 * 1024))
    _os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.vrt")
    _os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN",     "EMPTY_DIR")
    import rasterio
    N = len(lats)

    # ── Dedup to unique coords (6-decimal round, ~11 cm, lossless in 30 m DEM)
    lat_r = np.round(np.asarray(lats, dtype=np.float64), 6)
    lon_r = np.round(np.asarray(lons, dtype=np.float64), 6)
    packed = lat_r + 1j * lon_r
    uniq_packed, inv_idx = np.unique(packed, return_inverse=True)
    U = len(uniq_packed)
    uniq_lat = uniq_packed.real.astype(np.float64)
    uniq_lon = uniq_packed.imag.astype(np.float64)
    print(f"  DEM dedup: {N:,} obs → {U:,} unique coords "
          f"({100*(1 - U/max(N,1)):.2f}% dedup)", flush=True)

    # Tile-sort the uniques.
    tile_key = np.floor(uniq_lat).astype(int) * 1000 + np.floor(uniq_lon).astype(int)
    order = np.argsort(tile_key, kind="stable")

    uniq_vals = np.empty(U, dtype=np.float32)
    chunk = 500
    with rasterio.open(dem_path) as src:
        done = 0
        for i in range(0, U, chunk):
            j = min(U, i + chunk)
            idx = order[i:j]
            samples = src.sample(list(zip(uniq_lon[idx], uniq_lat[idx])))
            for k, v in enumerate(samples):
                uniq_vals[idx[k]] = v[0] if v is not None else np.nan
            done = j
            if (j // chunk) % 4 == 0 or j == U:
                print(f"  DEM sampled {done}/{U} uniques", flush=True)

    # Broadcast back to full N-row array.
    return uniq_vals[inv_idx]


def _apply_lapse(out_df, vars_, dem_elev, era5_elev):
    """In-place lapse-rate correction: 6.5 K/km × (era5_elev − dem_elev)."""
    delta = (era5_elev - dem_elev).astype(np.float32) * LAPSE_RATE_K_PER_M
    out_df["dem_elev_m"]  = dem_elev
    out_df["era5_elev_m"] = era5_elev
    for v in vars_:
        if v == "2m_temperature":
            for stat in ("min", "max", "mean"):
                col = f"{v}_{stat}"
                if col in out_df.columns:
                    out_df[col] = out_df[col].astype(np.float32) + delta
        elif v == "2m_dewpoint_temperature":
            col = f"{v}_mean"
            if col in out_df.columns:
                out_df[col] = out_df[col].astype(np.float32) + delta


def _daily_stats(ds_month, vars_):
    """Hourly → daily. Temp: min/max/mean. Precip & solar: sum. Others: mean."""
    daily = xr.Dataset()
    for v in vars_:
        if v not in ds_month.data_vars:
            raise KeyError(f"variable {v!r} not in store; available: {list(ds_month.data_vars)[:10]}...")
        da = ds_month[v]
        if v == "2m_temperature":
            daily[f"{v}_min"]  = da.resample(time="1D").min()
            daily[f"{v}_max"]  = da.resample(time="1D").max()
            daily[f"{v}_mean"] = da.resample(time="1D").mean()
        elif v in ("total_precipitation", "surface_solar_radiation_downwards"):
            daily[f"{v}_sum"]  = da.resample(time="1D").sum()
        else:
            daily[f"{v}_mean"] = da.resample(time="1D").mean()
    return daily.compute()


def process_month(year, month, obs_chunk, vars_, store_ds,
                  era5_elev_full=None, dem_elev_full=None):
    """era5_elev_full, dem_elev_full: 1D np.ndarray aligned to the *full* obs
    DataFrame index. process_month slices them by obs_chunk.index itself — no
    alignment work at the call site. Pass both=None to skip lapse correction."""
    is_0_360, W, E, lat_slice = _lon_convention(store_ds)
    print(f"  [start] {year}-{month:02d}  ({len(obs_chunk)} obs)", flush=True)
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end   = (month_start + pd.offsets.MonthEnd(1))
    ds = store_ds[vars_].sel(time=slice(str(month_start.date()), str(month_end.date())))
    ds = ds.sel(latitude=lat_slice, longitude=slice(W, E))
    daily = _daily_stats(ds, vars_)

    obs_lons_native = obs_chunk["lon"].to_numpy().astype(np.float64)
    if is_0_360:
        obs_lons_native = np.where(obs_lons_native < 0,
                                   obs_lons_native + 360.0, obs_lons_native)
    obs_lats_native = obs_chunk["lat"].to_numpy()

    lats_q  = xr.DataArray(obs_lats_native,  dims="obs")
    lons_q  = xr.DataArray(obs_lons_native,  dims="obs")
    dates_q = xr.DataArray(obs_chunk["date"].to_numpy().astype("datetime64[ns]"),
                           dims="obs")
    sampled = daily.sel(latitude=lats_q, longitude=lons_q, time=dates_q,
                        method="nearest")
    out = obs_chunk.reset_index(drop=True).copy()
    for v in sampled.data_vars:
        out[v] = sampled[v].to_numpy()

    if era5_elev_full is not None and dem_elev_full is not None:
        idx = obs_chunk.index.to_numpy()
        _apply_lapse(out, vars_,
                     dem_elev=dem_elev_full[idx],
                     era5_elev=era5_elev_full[idx])

    print(f"  [done ] {year}-{month:02d}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs_csv", required=True)
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--id_col",  default="id",
                    help='Observation id column; "__index__" uses the row index')
    ap.add_argument("--lat_col", default="latitude")
    ap.add_argument("--lon_col", default="longitude")
    ap.add_argument("--date_col", default="date",
                    help='Column with date/datetime; parsed by pd.to_datetime')
    ap.add_argument("--vars", nargs="+", default=DEFAULT_VARS,
                    help="ARCO-ERA5 variable long names")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent month-processing threads")
    ap.add_argument("--years", type=int, nargs=2, default=None,
                    help="Inclusive year range filter on input obs, e.g. 2011 2025")
    ap.add_argument("--dem_vrt", default=None,
                    help="Path to Copernicus DEM VRT (COP30_hh.vrt). If set, "
                         "t2m and d2m are lapse-rate corrected to site elevation.")
    ap.add_argument("--era5_orog_nc", default=os.path.expanduser("~/.cache/era5_orog_na.nc"),
                    help="Path to ERA5 invariant orography NetCDF (auto-downloaded via "
                         "cdsapi if missing; ~1 MB, NA+Meso bbox).")
    args = ap.parse_args()

    usecols = [args.lat_col, args.lon_col, args.date_col]
    if args.id_col != "__index__":
        usecols = [args.id_col] + usecols
    obs = pd.read_csv(args.obs_csv, usecols=usecols)
    if args.id_col == "__index__":
        obs["id"] = np.arange(len(obs), dtype=np.int64)
    else:
        obs = obs.rename(columns={args.id_col: "id"})
    obs = obs.rename(columns={args.lat_col: "lat", args.lon_col: "lon",
                              args.date_col: "date"})
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce", utc=False).dt.floor("D")
    obs = obs.dropna(subset=["date", "lat", "lon"])
    if args.years is not None:
        obs = obs[(obs.date.dt.year >= args.years[0]) & (obs.date.dt.year <= args.years[1])]
    obs["year"]  = obs.date.dt.year.astype(int)
    obs["month"] = obs.date.dt.month.astype(int)
    print(f"{len(obs):,} observations | {obs.date.min().date()} → {obs.date.max().date()}")
    print(f"  {obs.groupby(['year','month']).ngroups} year-months to process")

    print("Opening ARCO-ERA5 Zarr...")
    store = _open_store()
    missing = [v for v in args.vars if v not in store.data_vars]
    if missing:
        print(f"ERROR: variables not in store: {missing}")
        print(f"Sample of available: {sorted(list(store.data_vars))[:20]}")
        sys.exit(1)
    print(f"  vars OK: {args.vars}")

    # Pre-sample DEM + ERA5 orography at every obs, ONCE, before the thread pool
    # (rasterio/xarray are not thread-safe on shared handles; also avoids repeat I/O).
    # ERA5 orography is loaded from a one-off CDS-downloaded NetCDF; if missing,
    # it is auto-downloaded via cdsapi the first time.
    obs_dem_elev  = None
    obs_era5_elev = None
    if args.dem_vrt is not None:
        if not os.path.exists(args.dem_vrt):
            print(f"ERROR: DEM not found at {args.dem_vrt}", file=sys.stderr)
            sys.exit(1)

        _ensure_era5_orog_nc(args.era5_orog_nc)
        print(f"Loading ERA5 invariant orography from {args.era5_orog_nc}")
        orog_ds = xr.open_dataset(args.era5_orog_nc)
        # CDS NetCDF uses 'z' as geopotential variable name
        z_name = "z" if "z" in orog_ds.data_vars else list(orog_ds.data_vars)[0]
        elev_da = (orog_ds[z_name] / G_STD).astype(np.float32)
        for d in list(elev_da.dims):
            if d not in ("latitude", "longitude"):
                elev_da = elev_da.isel({d: 0})
        nvalid = int(np.isfinite(elev_da.values).sum())
        print(f"  ERA5 orography: dims={dict(elev_da.sizes)}  finite={nvalid}  "
              f"range [{float(np.nanmin(elev_da.values)):.1f}, "
              f"{float(np.nanmax(elev_da.values)):.1f}] m")
        if nvalid == 0:
            print("ERROR: ERA5 orography NetCDF is entirely NaN", file=sys.stderr)
            sys.exit(1)

        lats_raw = obs["lat"].to_numpy()
        lons_raw = obs["lon"].to_numpy()
        # CDS orog NetCDF uses -180..180 longitude by default, but if 0..360 detected, wrap.
        if float(elev_da.longitude.min()) >= 0:
            lons_q = np.where(lons_raw < 0, lons_raw + 360.0, lons_raw)
        else:
            lons_q = lons_raw
        print(f"Pre-sampling ERA5 orography at {len(obs)} obs...")
        obs_era5_elev = elev_da.sel(
            latitude=xr.DataArray(lats_raw, dims="obs"),
            longitude=xr.DataArray(lons_q, dims="obs"),
            method="nearest",
        ).to_numpy().astype(np.float32)
        print(f"  obs ERA5 orog: range [{np.nanmin(obs_era5_elev):.1f}, "
              f"{np.nanmax(obs_era5_elev):.1f}] m")

        print(f"Pre-sampling DEM at {len(obs)} obs points (single-threaded)...")
        obs_dem_elev = _sample_dem(args.dem_vrt, lats_raw, lons_raw)
        print(f"  obs DEM (site): range [{np.nanmin(obs_dem_elev):.1f}, "
              f"{np.nanmax(obs_dem_elev):.1f}] m")

        delta = obs_era5_elev - obs_dem_elev
        print(f"  Δelev (era5 − site): mean={np.nanmean(delta):+.1f}  "
              f"|max|={np.nanmax(np.abs(delta)):.1f} m  "
              f"→ lapse correction range [{np.nanmin(delta)*0.0065:+.2f}, "
              f"{np.nanmax(delta)*0.0065:+.2f}] K")

    # Dispatch per (year, month); threads overlap GCS fetch latency well.
    groups = [((yr, mo), chunk) for (yr, mo), chunk in obs.groupby(["year", "month"])]
    print(f"Dispatching {len(groups)} month-tasks across {args.workers} workers")

    def run(key_chunk):
        (yr, mo), chunk = key_chunk
        try:
            return process_month(yr, mo, chunk, args.vars, store,
                                 era5_elev_full=obs_era5_elev,
                                 dem_elev_full=obs_dem_elev)
        except Exception as e:
            print(f"  [err] {yr}-{mo:02d}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return None

    parts = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, g) for g in groups]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                parts.append(r)
            done += 1
            rows_so_far = sum(len(p) for p in parts)
            print(f"  {done}/{len(futs)} months | {rows_so_far:,} rows extracted",
                  flush=True)

    if not parts:
        print("No rows extracted — aborting.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.drop(columns=["year", "month"])
    out_df.to_parquet(args.out, index=False)
    print(f"\nWrote {len(out_df):,} rows × {len(out_df.columns)} cols → {args.out}")
    print(f"  size: {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
