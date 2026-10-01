# Monarch grid — Data Processing

Builds the 0.5° North America grid used to render the monarch butterfly (*Danaus plexippus*) suitability maps shown in the paper, snapshotted on the four phenology dates: **2025-05-15, 2025-07-15, 2025-09-15, 2025-11-15**.

## Pipeline (run in order)

1. **`make_grid_na.py`** — builds the land-only static grid over North America with SoilGrids (8 properties, 0–5 cm) and Copernicus GLO-30 DEM elevation. Mexico → southern Canada; Hawaii excluded. SoilGrids nodata is written as NaN, as in the observation datasets; cells outside every DEM tile are dropped.
   - Output: `static_grid_na.csv`.

2. **`enrich_grid_daily.py`** — for each requested date, runs the grid cells through the same covariate code as the eButterfly observations: `covariates/era5/extract_era5_at_points.py` for ARCO-ERA5 daily 2 m temperature (min/max/mean) and total precipitation (lapse-rate corrected to the DEM), and `covariates/modis_phenology/mod13q1.py` for NDVI/EVI from the 16-day composite nearest the date.
   - Output: one `monarch_grid_<DATE>.csv` per date with the 15-column env schema STEM-LM consumes.

3. **`monarch_inference.py`** — loads a trained STEM-LM run, predicts the species at every grid cell with all species masked ($p=1$) and every eButterfly observation as the source pool, applies the run's temperature $T^\star$ (`temperature.json`, fitted on validation by `stemlm train --temperature_scaling`), and renders one panel per date in a single row.
   - Output (in `--output_dir`): `monarch_pred_<DATE>.csv` per date (`suitability` with $T^\star$, `suitability_raw` without), `monarch_pred_full.png`, `monarch_pred_full_raw.png`.

## Reproducing the four panels

```bash
cd data_processing/Monarch

# 1. static grid (run once)
python make_grid_na.py --resolution 0.5 --output static_grid_na.csv

# 2. four phenology snapshots
python enrich_grid_daily.py \
    --static_grid static_grid_na.csv \
    --dates 2025-05-15 2025-07-15 2025-09-15 2025-11-15 \
    --workers 32

# 3. inference + 1×4 plot
python monarch_inference.py /path/to/ebutterfly_na_2011_2025.parquet \
    --run_dir /path/to/stemlm_run --grid_dir . --output_dir output
```

Use `--workers` up to your core count; ARCO-ERA5 months are fetched concurrently.

## Environmental data acquisition

Same sources and code as `data_processing/eButterfly`; the extraction code lives under `data_processing/covariates/<source>/` and raw rasters under `ENV_DIR` (default `${REPO_ROOT}/Examples/env_vars`).

| Source | Product | Access | License | Citation |
|---|---|---|---|---|
| ARCO-ERA5 | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (Zarr) | Public GCS bucket, no account. Read directly via `xarray` + `gcsfs`. | Copernicus License | Hersbach et al. (2020); ARCO mirror by Google Public Datasets |
| Copernicus DEM | GLO-30 (30 m) | `/vsicurl/` from OpenTopography S3; no account. | Copernicus DEM License | ESA / Airbus (2021), DOI: 10.5270/ESA-c5d3d65 |
| MODIS MOD13Q1 v6.1 | Terra 16-day NDVI/EVI 250 m | NASA Earthdata HDF4; NA tiles under `${ENV_DIR}/modis_phenology/raw/`, downloaded by `covariates/modis_phenology/download_mod13q1_na.py`. | NASA open data | Didan (2021), DOI: 10.5067/MODIS/MOD13Q1.061 |
| SoilGrids v2.0 | 8 properties at 0–5 cm | WCS GetCoverage; download scripts under `covariates/soilgrids/`. | CC BY 4.0 | Poggio et al. (2021), SOIL 7:217–240 |

## Notes

- The grid CSVs (~10,500 land cells × 15 covariates per date) are not committed; re-run the pipeline to regenerate.
- ERA5 orography is cached at `~/.cache/era5_orog_na.nc` after the first run.
- Hawaii is excluded via a longitude/latitude rectangle in `make_grid_na.py`; Great Lakes are masked at render time, not in the grid.
