import argparse
import os
import subprocess
import sys

import pandas as pd

REPO_ROOT = os.environ.get("REPO_ROOT",
                           os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
COVARIATES = os.path.join(REPO_ROOT, "data_processing", "covariates")
sys.path.insert(0, os.path.join(COVARIATES, "modis_phenology"))
from mod13q1 import extract_modis  # noqa: E402

ENV_DIR = os.environ.get("ENV_DIR", os.path.join(REPO_ROOT, "Examples", "env_vars"))
MODIS_DIR = os.path.join(ENV_DIR, "modis_phenology", "raw")
ERA5_EXTRACT = os.path.join(COVARIATES, "era5", "extract_era5_at_points.py")
DEM_VRT = os.path.join(COVARIATES, "copernicus_dem", "COP30_hh_vsicurl.vrt")
ERA5_RENAME = {"2m_temperature_min": "env_t2m_min", "2m_temperature_max": "env_t2m_max",
               "2m_temperature_mean": "env_t2m_mean", "total_precipitation_sum": "env_tp_sum"}
ENV_ORDER = [*ERA5_RENAME.values(), "env_ndvi", "env_evi",
             "env_soil_bdod", "env_soil_cec", "env_soil_cfvo", "env_soil_clay",
             "env_soil_nitrogen", "env_soil_phh2o", "env_soil_sand", "env_soil_silt", "env_dem"]


def main():
    parser = argparse.ArgumentParser(description="Add date-matched ERA5 and MODIS covariates to the static grid.")
    parser.add_argument("--static_grid", default="static_grid_na.csv")
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    grid = pd.read_csv(args.static_grid)
    os.makedirs(args.output_dir, exist_ok=True)
    for date in args.dates:
        points = grid.assign(time=date)
        points_csv = os.path.join(args.output_dir, f"_grid_points_{date}.csv")
        era5_path = os.path.join(args.output_dir, f"_grid_era5_{date}.parquet")
        points[["time", "latitude", "longitude"]].to_csv(points_csv, index=False)
        subprocess.run([sys.executable, ERA5_EXTRACT, "--obs_csv", points_csv, "--out", era5_path,
                        "--id_col", "__index__", "--date_col", "time",
                        "--vars", "2m_temperature", "total_precipitation",
                        "--workers", str(args.workers), "--dem_vrt", DEM_VRT], check=True)
        era5 = pd.read_parquet(era5_path).set_index("id").sort_index().reindex(range(len(points)))
        out = points.assign(**{dst: era5[src].to_numpy() for src, dst in ERA5_RENAME.items()})
        out = pd.concat([out, extract_modis(points, MODIS_DIR)], axis=1)
        out_path = os.path.join(args.output_dir, f"monarch_grid_{date}.csv")
        out[["time", "latitude", "longitude", *ENV_ORDER]].to_csv(out_path, index=False)
        print(f"{date}: {len(out):,} cells -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
