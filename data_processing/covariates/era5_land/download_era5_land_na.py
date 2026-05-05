"""
Download ERA5-Land hourly data for NA+Mesoamerica, one (variable, year, month) per file.

Optionally aggregates each month-file to daily stats in place, reducing ~300MB hourly
files to ~12MB daily files. Temp: daily min/max/mean. Dewpoint: daily mean.
Precip & solar: daily sum.

Requires:
  pip install "cdsapi>=0.7.7" xarray netCDF4
  ~/.cdsapirc with:
    url: https://cds.climate.copernicus.eu/api
    key: <PAT-from-https://cds.climate.copernicus.eu/profile>
  Accept ERA5-Land T&C once at:
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land

Usage:
  python download_era5_land_na.py --outdir raw --aggregate_after   # download + inline daily
  python download_era5_land_na.py --outdir raw --aggregate_only    # aggregate existing files, no download
"""
import argparse, os, sys, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]
# NA + Mesoamerica bbox (N, W, S, E)
AREA = [72, -170, 7, -50]


def to_daily(path):
    """Aggregate an hourly ERA5-Land month-file to daily stats; delete hourly on success."""
    import xarray as xr
    if path.endswith("_daily.nc"):
        return path
    out = path[:-3] + "_daily.nc"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        os.remove(path)
        return out
    tmp = out + ".part"
    ds = xr.open_dataset(path)
    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
    daily = xr.Dataset()
    for var in ds.data_vars:
        da = ds[var]
        kw = {time_coord: "1D"}
        if var == "t2m":
            daily[f"{var}_min"]  = da.resample(**kw).min()
            daily[f"{var}_max"]  = da.resample(**kw).max()
            daily[f"{var}_mean"] = da.resample(**kw).mean()
        elif var in ("tp", "ssrd"):
            daily[f"{var}_sum"]  = da.resample(**kw).sum()
        else:
            daily[f"{var}_mean"] = da.resample(**kw).mean()
    enc = {v: {"zlib": True, "complevel": 4} for v in daily.data_vars}
    daily.to_netcdf(tmp, engine="netcdf4", encoding=enc)
    ds.close()
    os.replace(tmp, out)
    os.remove(path)
    return out


def retrieve_one(var, year, month, outdir, aggregate_after=False):
    out_hourly = os.path.join(outdir, f"era5_land_{var}_{year}_{month:02d}.nc")
    out_daily  = out_hourly[:-3] + "_daily.nc"
    if os.path.exists(out_daily) and os.path.getsize(out_daily) > 0:
        print(f"[skip] {out_daily}", flush=True)
        return out_daily
    if os.path.exists(out_hourly) and os.path.getsize(out_hourly) > 0:
        if aggregate_after:
            print(f"[aggr] {out_hourly}", flush=True)
            return to_daily(out_hourly)
        print(f"[skip] {out_hourly}", flush=True)
        return out_hourly
    tmp = out_hourly + ".part"
    client = cdsapi.Client(quiet=True, progress=False)
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": var,
            "year": str(year),
            "month": f"{month:02d}",
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        tmp,
    )
    os.replace(tmp, out_hourly)
    print(f"[done] {out_hourly}", flush=True)
    if aggregate_after:
        print(f"[aggr] {out_hourly}", flush=True)
        return to_daily(out_hourly)
    return out_hourly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2010, 2025],
                    help="inclusive range, e.g. 2010 2026")
    ap.add_argument("--outdir", default="raw")
    ap.add_argument("--workers", type=int, default=16,
                    help="CDS throttles ~20 active requests; 4-8 is polite")
    ap.add_argument("--variables", nargs="+", default=VARIABLES)
    ap.add_argument("--aggregate_after", action="store_true",
                    help="After each download, aggregate hourly to daily and remove hourly")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip downloads; aggregate existing hourly files in outdir")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.aggregate_only:
        paths = sorted(p for p in glob.glob(os.path.join(args.outdir, "era5_land_*.nc"))
                       if not p.endswith("_daily.nc"))
        print(f"Aggregating {len(paths)} hourly files to daily, {args.workers} concurrent")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(to_daily, p) for p in paths]
            done = 0
            for f in as_completed(futs):
                try:
                    out = f.result()
                    done += 1
                    if done % 10 == 0:
                        print(f"({done}/{len(paths)}) {out}", flush=True)
                except Exception as e:
                    print(f"[err] {e}", file=sys.stderr, flush=True)
        return

    years = list(range(args.years[0], args.years[1] + 1))
    jobs = [(v, y, m) for v in args.variables for y in years for m in range(1, 13)]
    print(f"Submitting {len(jobs)} (var, year, month) jobs to CDS, {args.workers} concurrent"
          + (f"  (aggregate_after=True)" if args.aggregate_after else ""))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(retrieve_one, v, y, m, args.outdir, args.aggregate_after)
                for v, y, m in jobs]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"[err] {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
