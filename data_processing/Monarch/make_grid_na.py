"""
0.5° land-only NA grid for monarch migration visualization.

Output columns: latitude, longitude, env_dem,
                env_soil_bdod, env_soil_cec, env_soil_cfvo, env_soil_clay,
                env_soil_nitrogen, env_soil_phh2o, env_soil_sand, env_soil_silt
(static layers only; daily ERA5 + MODIS added by enrich_grid_daily.py)

Land mask:
  1. Natural Earth admin_0 country polygons (US/Canada/Mexico/Mesoamerica).
  2. Raster nodata (cells where any soil var or DEM is nodata are dropped).

Usage:
  python make_grid_na.py --resolution 0.5 --output static_grid_na.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import rasterio

REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
ENV_DIR = os.path.join(REPO_ROOT, "Examples", "env_vars")
SG_DIR  = os.path.join(ENV_DIR, "soilgrids", "raw")
DEM_VRT = os.path.join(ENV_DIR, "copernicus_dem", "COP30_hh_vsicurl.vrt")

LON_MIN, LON_MAX = -170.0, -50.0
LAT_MIN, LAT_MAX =    7.0,  72.0

INCLUDE_COUNTRIES = {
    "United States of America", "Canada",
    "Mexico", "Guatemala", "Belize", "Honduras", "El Salvador",
    "Nicaragua", "Costa Rica", "Panama",
}

SOIL_VARS = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt"]
ENV_COLS  = ["env_dem"] + [f"env_soil_{v}" for v in SOIL_VARS]


def get_country_geom():
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union
    shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                  name="admin_0_countries")
    geoms = []
    for rec in shpreader.Reader(shp).records():
        name = rec.attributes.get("NAME_LONG", "") or rec.attributes.get("NAME", "")
        if any(inc.lower() in name.lower() or name.lower() in inc.lower()
               for inc in INCLUDE_COUNTRIES):
            geoms.append(rec.geometry)
    print(f"  Country mask: {len(geoms)} polygons")
    return unary_union(geoms)


def make_grid(resolution):
    lons = np.arange(LON_MIN, LON_MAX + resolution, resolution)
    lats = np.arange(LAT_MIN, LAT_MAX + resolution, resolution)
    lon_g, lat_g = np.meshgrid(lons, lats)
    return lat_g.ravel(), lon_g.ravel()


def sample_raster(path, coords):
    with rasterio.open(path) as src:
        vals = np.array(list(src.sample(coords, masked=True)), dtype=np.float32)
    v = vals[:, 0] if vals.ndim == 2 else vals
    if hasattr(v, "filled"):
        v = v.filled(np.nan)
    return v


def sample_soil(lons, lats):
    coords = list(zip(lons.tolist(), lats.tolist()))
    out = np.full((len(lons), len(SOIL_VARS)), np.nan, dtype=np.float32)
    for i, var in enumerate(SOIL_VARS):
        path = os.path.join(SG_DIR, f"{var}_0-5cm_na.tif")
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        print(f"  SoilGrids {var} ...", end="\r", flush=True)
        vals = sample_raster(path, coords)
        vals[vals > 1e6] = np.nan
        out[:, i] = vals
    print()
    return out


def sample_dem(lons, lats):
    coords = list(zip(lons.tolist(), lats.tolist()))
    print("  DEM via VRT ...", flush=True)
    return sample_raster(DEM_VRT, coords)


def main(resolution, output):
    print(f"Building {resolution}° grid over NA "
          f"({LON_MIN}–{LON_MAX}°E, {LAT_MIN}–{LAT_MAX}°N)")
    lats, lons = make_grid(resolution)
    print(f"Grid points before land masking: {len(lats):,}")

    print("Building country mask ...")
    from shapely.geometry import Point
    geom = get_country_geom()
    in_country = np.array([geom.contains(Point(x, y))
                           for x, y in zip(lons.tolist(), lats.tolist())])
    keep_mask = in_country & ~((lats >= 18) & (lats <= 23) &
                               (lons >= -161) & (lons <= -154))
    lats, lons = lats[keep_mask], lons[keep_mask]
    print(f"After country mask: {len(lats):,} points")

    print("\nExtracting SoilGrids ...")
    sg = sample_soil(lons, lats)
    print("Extracting DEM ...")
    dem = sample_dem(lons, lats)

    env = np.concatenate([dem.reshape(-1, 1), sg], axis=1)
    sg_zero = (sg == 0)
    sg = np.where(sg_zero, np.nan, sg)
    for j in range(sg.shape[1]):
        col = sg[:, j]
        col_mean = np.nanmean(col)
        col[np.isnan(col)] = col_mean
        sg[:, j] = col
    env = np.concatenate([dem.reshape(-1, 1), sg], axis=1)
    keep = np.isfinite(env).all(axis=1)
    print(f"\nFinal land mask: {keep.sum():,} / {len(lats):,} kept "
          f"({keep.mean()*100:.1f}%)")

    df = pd.DataFrame(env[keep], columns=ENV_COLS)
    df.insert(0, "longitude", lons[keep])
    df.insert(0, "latitude",  lats[keep])
    df.to_csv(output, index=False)
    print(f"\nSaved {len(df):,} land grid points → {output}")
    print(f"  Lat: {df.latitude.min():.2f}–{df.latitude.max():.2f}")
    print(f"  Lon: {df.longitude.min():.2f}–{df.longitude.max():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--output", type=str, default="static_grid_na.csv")
    args = p.parse_args()
    main(args.resolution, args.output)
