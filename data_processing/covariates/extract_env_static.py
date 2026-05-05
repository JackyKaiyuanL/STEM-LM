"""
Extract static environmental variables and append to jSDM CSVs.

Variables (28 total):
  env_bio01..env_bio19  WorldClim v2.1 bioclimatic (19 vars)
  env_bdod..env_silt    SoilGrids v2.0 soil 0-5cm  (8 vars)
  env_dem               Copernicus DEM GLO-30        (1 var)

Sources:
  WorldClim: ${REPO_ROOT}/Examples/env_vars/worldclim/raw/wc2.1_30s_bio.zip
             Read via /vsizip/ — no unzip required
  SoilGrids: ${REPO_ROOT}/Examples/env_vars/soilgrids/raw/{var}_0-5cm_{region}.tif
  DEM:       https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/
             Read via /vsicurl/ — tiles fetched on demand

Usage:
  python extract_env_static.py --dataset splotopen_europe
  python extract_env_static.py --dataset splotopen_australia
  python extract_env_static.py --dataset ebutterfly_na
  python extract_env_static.py --dataset ebutterfly_us
  python extract_env_static.py --csv /path/to/file.csv --no_soilgrids

Datasets and their SoilGrids region key:
  splotopen_europe    → soilgrids region: europe
  splotopen_australia → soilgrids region: australia
  ebutterfly_na       → soilgrids region: na  (covers US + Canada + Mesoamerica)
  ebutterfly_us       → soilgrids region: na  (US is a spatial subset of NA tiles)
"""

import argparse, math, os
import numpy as np
import pandas as pd
import rasterio
from collections import defaultdict

REPO_ROOT  = os.environ.get("REPO_ROOT",
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ENV_DIR    = os.environ.get("ENV_DIR", os.path.join(REPO_ROOT, "Examples", "env_vars"))
WC_ZIP     = f"{ENV_DIR}/worldclim/raw/wc2.1_30s_bio.zip"
SG_DIR     = f"{ENV_DIR}/soilgrids/raw"
COP_BASE   = "https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh"

WORLDCLIM_VARS = [f"bio{i:02d}" for i in range(1, 20)]   # bio01..bio19
SOILGRIDS_VARS = ["bdod", "cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt"]

WC_ENV_COLS = [f"env_bio{i:02d}" for i in range(1, 20)]
SG_ENV_COLS = [f"env_{v}" for v in SOILGRIDS_VARS]
DEM_ENV_COLS = ["env_dem"]
CANONICAL_ENV_ORDER = WC_ENV_COLS + SG_ENV_COLS + DEM_ENV_COLS

_LAB = os.path.join(REPO_ROOT, "lab")
DATASET_CONFIG = {
    "splotopen_europe": {
        "csv": os.path.join(_LAB, "sPlotOpen", "splotopen_europe_jsdm.csv"),
        "sg_region": "europe",
    },
    "splotopen_australia": {
        "csv": os.path.join(_LAB, "sPlotOpen", "splotopen_australia_jsdm.csv"),
        "sg_region": "australia",
    },
    "ebutterfly_na": {
        "csv": os.path.join(_LAB, "eButterfly", "ebutterfly_na_static_jsdm.csv"),
        "sg_region": "na",
    },
    "ebutterfly_us": {
        "csv": os.path.join(_LAB, "eButterfly", "ebutterfly_us_static_jsdm.csv"),
        "sg_region": "na",
    },
}

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"


def sample_raster(path, coords_xy):
    """Sample a raster at list of (lon, lat) tuples. Returns np.array."""
    with rasterio.open(path) as src:
        vals = np.array(list(src.sample(coords_xy, masked=True)), dtype=np.float32)
    return vals[:, 0]  # single band


def sample_worldclim(lons, lats):
    """Sample all 19 WorldClim bands. Returns (N, 19) array."""
    coords = list(zip(lons, lats))
    out = np.full((len(lons), 19), np.nan, dtype=np.float32)
    for i, bio in enumerate(WORLDCLIM_VARS):
        inner = f"wc2.1_30s_bio_{int(bio[3:])}.tif"
        path = f"/vsizip/{WC_ZIP}/{inner}"
        print(f"  WorldClim {bio} ...", flush=True)
        vals = sample_raster(path, coords)
        out[:, i] = vals
    return out


def sample_soilgrids(lons, lats, region):
    """Sample all 8 SoilGrids bands. Returns (N, 8) array."""
    coords = list(zip(lons, lats))
    out = np.full((len(lons), 8), np.nan, dtype=np.float32)
    for i, var in enumerate(SOILGRIDS_VARS):
        tif_path = os.path.join(SG_DIR, f"{var}_0-5cm_{region}.tif")
        vrt_path = os.path.join(SG_DIR, f"{var}_0-5cm_{region}.vrt")
        path = tif_path if os.path.exists(tif_path) else vrt_path
        if not os.path.exists(path):
            print(f"  MISSING: {tif_path} (or {vrt_path}) — skipping")
            continue
        print(f"  SoilGrids {var} ...", flush=True)
        vals = sample_raster(path, coords)
        out[:, i] = vals
    return out


def sample_dem(lons, lats):
    """Sample Copernicus DEM via vsicurl, grouping by tile for efficiency."""
    out = np.full(len(lons), np.nan, dtype=np.float32)

    # Group point indices by their 1°x1° tile
    tile_groups = defaultdict(list)
    for idx, (lon, lat) in enumerate(zip(lons, lats)):
        tlat = math.floor(lat)
        tlon = math.floor(lon)
        ns = "N" if tlat >= 0 else "S"
        ew = "E" if tlon >= 0 else "W"
        tile = f"Copernicus_DSM_10_{ns}{abs(tlat):02d}_00_{ew}{abs(tlon):03d}_00_DEM.tif"
        tile_groups[tile].append(idx)

    n_tiles = len(tile_groups)
    print(f"  DEM: {n_tiles} unique tiles for {len(lons)} points ...", flush=True)
    for t, (tile_name, idxs) in enumerate(tile_groups.items()):
        url = f"/vsicurl/{COP_BASE}/{tile_name}"
        coords = [(lons[i], lats[i]) for i in idxs]
        try:
            with rasterio.open(url) as src:
                vals = np.array(list(src.sample(coords, masked=True)),
                                dtype=np.float32)[:, 0]
            for i, idx in enumerate(idxs):
                out[idx] = vals[i]
        except Exception as e:
            print(f"    WARN tile {tile_name}: {e}")
        if (t + 1) % 50 == 0:
            print(f"    {t+1}/{n_tiles} tiles done", flush=True)

    n_nan = np.isnan(out).sum()
    if n_nan:
        print(f"  DEM: {n_nan} NaN values (ocean/water tiles → set to 0)")
        out = np.nan_to_num(out, nan=0.0)
    return out


def enrich_csv(
    csv_path,
    *,
    out_path=None,
    sg_region=None,
    include_worldclim: bool = True,
    include_soilgrids: bool = True,
    include_dem: bool = True,
    min_presences: int = 100,
):
    if not (include_worldclim or include_soilgrids or include_dem):
        raise ValueError("Nothing to do: all extraction steps disabled.")
    if include_soilgrids and not sg_region:
        raise ValueError("Missing sg_region (required unless include_soilgrids=False).")

    env_cols = (
        (WC_ENV_COLS if include_worldclim else []) +
        (SG_ENV_COLS if include_soilgrids else []) +
        (DEM_ENV_COLS if include_dem else [])
    )
    out_path = out_path or csv_path

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Skip if requested env cols already present and complete
    if all(c in df.columns for c in env_cols) and not df[env_cols].isna().any().any():
        print("Requested env columns already present and complete — nothing to do.")
        if out_path != csv_path:
            df.to_csv(out_path, index=False)
            print(f"Saved copy → {out_path}")
        return

    lons = df["longitude"].values.astype(float)
    lats = df["latitude"].values.astype(float)
    print(f"{len(df)} plots, soilgrids_region={sg_region!r}")

    parts = []
    if include_worldclim:
        print("\nExtracting WorldClim ...")
        wc = sample_worldclim(lons, lats)
        parts.append(pd.DataFrame(wc, columns=WC_ENV_COLS))
    if include_soilgrids:
        print("\nExtracting SoilGrids ...")
        sg = sample_soilgrids(lons, lats, sg_region)
        parts.append(pd.DataFrame(sg, columns=SG_ENV_COLS))
    if include_dem:
        print("\nExtracting DEM ...")
        dem = sample_dem(lons, lats)
        parts.append(pd.DataFrame(dem.reshape(-1, 1), columns=DEM_ENV_COLS))

    env_df = pd.concat(parts, axis=1)
    assert list(env_df.columns) == env_cols

    # Insert env cols after longitude (before species). If the input already
    # has env_* columns, preserve them and overwrite/append the requested set.
    meta_cols = ["time", "latitude", "longitude"]
    existing_env_cols = [c for c in df.columns if c.startswith("env_")]
    species_cols = [c for c in df.columns if c not in meta_cols + existing_env_cols]

    # Preserve any existing env columns not being recomputed.
    preserved_cols = [c for c in CANONICAL_ENV_ORDER if c in existing_env_cols and c not in env_cols]
    preserved_extra = [c for c in existing_env_cols if c not in CANONICAL_ENV_ORDER and c not in env_cols]
    preserved_env = df[preserved_cols + preserved_extra] if (preserved_cols or preserved_extra) else None

    merged_env = env_df if preserved_env is None else pd.concat([preserved_env, env_df], axis=1)
    # Reorder env columns canonically when possible (unknown env_* columns at end).
    merged_env_cols = [c for c in CANONICAL_ENV_ORDER if c in merged_env.columns] + [
        c for c in merged_env.columns if c not in CANONICAL_ENV_ORDER
    ]
    merged_env = merged_env[merged_env_cols]

    out = pd.concat([df[meta_cols], merged_env, df[species_cols]], axis=1)

    # Drop rows with any NaN in env columns present — incomplete coverage is unacceptable
    final_env_cols = [c for c in out.columns if c.startswith("env_")]
    before = len(out)
    out = out.dropna(subset=final_env_cols).reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"\nDropped {dropped} rows with NaN env values "
              f"({dropped/before:.1%} of plots)")
        # Re-apply min_presences filter on remaining rows
        sp_remaining = [c for c in out.columns if c not in meta_cols + env_cols]
        keep2 = [c for c in sp_remaining if out[c].sum() >= min_presences]
        dropped_sp = len(sp_remaining) - len(keep2)
        if dropped_sp:
            print(f"  Re-filtered species: {len(keep2)} kept, {dropped_sp} dropped "
                  f"(fell below {min_presences} presences after row drop)")
        out = out[meta_cols + env_cols + keep2]
    else:
        print(f"\nNo NaN values — all {len(out)} rows complete.")

    assert not out[env_cols].isna().any().any(), "NaN still present after drop!"

    out.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(f"Shape: {out.shape}  ({len(env_cols)} env columns, "
          f"{len([c for c in out.columns if c not in meta_cols+env_cols])} species)")


def main(dataset, min_presences=100):
    cfg = DATASET_CONFIG[dataset]
    enrich_csv(
        cfg["csv"],
        out_path=cfg["csv"],
        sg_region=cfg["sg_region"],
        include_worldclim=True,
        include_soilgrids=True,
        include_dem=True,
        min_presences=min_presences,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()))
    src.add_argument("--csv", type=str, help="Input jSDM CSV path to enrich in-place")
    p.add_argument("--out_csv", type=str, default=None,
                   help="Optional output path (default: overwrite input CSV)")
    p.add_argument("--sg_region", type=str, default=None,
                   help="SoilGrids region key (e.g., europe/australia/na); required unless --no_soilgrids")
    p.add_argument("--no_worldclim", action="store_true",
                   help="Skip WorldClim env_bio01..19 extraction")
    p.add_argument("--no_soilgrids", action="store_true",
                   help="Skip SoilGrids env_{bdod..silt} extraction")
    p.add_argument("--no_dem", action="store_true",
                   help="Skip DEM env_dem extraction")
    p.add_argument("--min_presences", type=int, default=100)
    a = p.parse_args()

    if a.dataset:
        cfg = DATASET_CONFIG[a.dataset]
        csv_path = cfg["csv"]
        sg_region = cfg["sg_region"]
    else:
        csv_path = a.csv
        sg_region = a.sg_region

    try:
        enrich_csv(
            csv_path,
            out_path=a.out_csv or csv_path,
            sg_region=sg_region,
            include_worldclim=not a.no_worldclim,
            include_soilgrids=not a.no_soilgrids,
            include_dem=not a.no_dem,
            min_presences=a.min_presences,
        )
    except ValueError as e:
        raise SystemExit(str(e))
