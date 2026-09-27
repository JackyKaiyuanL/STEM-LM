import glob
import math
import re

import numpy as np
import pandas as pd

R_MODIS = 6371007.181
MAX_X_SIN = R_MODIS * math.pi
MAX_Y_SIN = R_MODIS * math.pi / 2
TILE_WIDTH_M = 1_111_950.5197665
PIXELS_PER_TILE = 4800
HDF_NAME = re.compile(r"MOD13Q1\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.061\..*\.hdf$")


def latlon_to_sin(lats, lons):
    lat_r = np.radians(lats)
    return R_MODIS * np.radians(lons) * np.cos(lat_r), R_MODIS * lat_r


def sin_to_tile_pixel(x, y):
    h = ((x + MAX_X_SIN) / TILE_WIDTH_M).astype(int)
    v = ((MAX_Y_SIN - y) / TILE_WIDTH_M).astype(int)
    px = TILE_WIDTH_M / PIXELS_PER_TILE
    col = (((x + MAX_X_SIN) - h * TILE_WIDTH_M) / px).astype(int).clip(0, PIXELS_PER_TILE - 1)
    row = (((MAX_Y_SIN - y) - v * TILE_WIDTH_M) / px).astype(int).clip(0, PIXELS_PER_TILE - 1)
    return h, v, col, row


def nearest_mod13q1_doy(dates):
    doy = dates.dt.dayofyear.to_numpy()
    composite_doys = np.arange(1, 354, 16)
    return composite_doys[np.abs(doy[:, None] - composite_doys[None, :]).argmin(axis=1)]


def read_mod13q1(path):
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        import rasterio
        grid = f'HDF4_EOS:EOS_GRID:"{path}":MODIS_Grid_16DAY_250m_500m_VI:250m 16 days'
        with rasterio.open(f"{grid} NDVI") as src:
            ndvi = src.read(1)
        with rasterio.open(f"{grid} EVI") as src:
            evi = src.read(1)
        return ndvi, evi
    sd = SD(path, SDC.READ)
    ndvi, evi = sd.select("250m 16 days NDVI").get(), sd.select("250m 16 days EVI").get()
    sd.end()
    return ndvi, evi


def extract_modis(df, modis_dir):
    hdf_map = {}
    for path in glob.glob(f"{modis_dir}/*.hdf"):
        m = HDF_NAME.search(path)
        if m:
            hdf_map[tuple(int(g) for g in m.groups())] = path
    dates = pd.to_datetime(df["time"])
    h, v, col, row = sin_to_tile_pixel(*latlon_to_sin(df["latitude"].to_numpy(), df["longitude"].to_numpy()))
    keys = pd.Series(list(zip(dates.dt.year.to_numpy(), nearest_mod13q1_doy(dates), h, v)))
    ndvi = np.full(len(df), np.nan, dtype=np.float32)
    evi = np.full(len(df), np.nan, dtype=np.float32)
    missing = 0
    for key, positions in keys.groupby(keys).groups.items():
        path = hdf_map.get(key)
        if path is None:
            missing += 1
            continue
        ndvi_arr, evi_arr = read_mod13q1(path)
        pos = np.asarray(positions)
        ndvi[pos] = ndvi_arr[row[pos], col[pos]].astype(np.float32) / 10000.0
        evi[pos] = evi_arr[row[pos], col[pos]].astype(np.float32) / 10000.0
    ndvi[ndvi < -0.2] = np.nan
    evi[evi < -0.2] = np.nan
    print(f"  MODIS: {len(hdf_map)} HDF files, {keys.nunique()} tile keys ({missing} without a file), "
          f"NDVI valid {np.isfinite(ndvi).sum()}/{len(df)}", flush=True)
    return pd.DataFrame({"env_ndvi": ndvi, "env_evi": evi}, index=df.index)
