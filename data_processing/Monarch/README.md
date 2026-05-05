# Monarch grid — Data Processing

Builds the 0.5° North America grid used to render the monarch butterfly (*Danaus plexippus*) suitability maps shown in the paper, snapshotted on the four phenology dates: **2025-05-15, 2025-07-15, 2025-09-15, 2025-11-15**.

## Pipeline (run in order)

1. **`make_grid_na.py`** — builds the land-only static grid over North America with SoilGrids (8 properties, 0–5 cm) and Copernicus GLO-30 DEM elevation. Mexico → southern Canada; Hawaii excluded. Soil-zero water cells are mean-imputed rather than dropped.
   - Output: `static_grid_na.csv`.

2. **`enrich_grid_daily.py`** — for each requested date, joins the static grid with ARCO-ERA5 daily 2 m temperature (min/max/mean) and total precipitation (lapse-rate corrected with the static DEM at $-6.5\,^\circ\mathrm{C}\,\mathrm{km}^{-1}$), plus MODIS MOD13Q1 NDVI/EVI from the 16-day composite nearest the date.
   - Output: one `monarch_grid_<DATE>.csv` per date with the 15-column env schema STEM-LM consumes.

3. **`monarch_grid_inference.ipynb`** — loads the trained STEM-LM full model, runs bagged inference ($K=10$) at each of the four dates, fits the post-hoc temperature $T^\star$ on validation logits via NLL minimization (the load-bearing calibration step that drops test ECE from $0.0554$ → $0.0063$), then writes one `monarch_pred_<DATE>.csv` per date and renders the 1×4 figure.
   - Inputs (must be present in `WORK_DIR`): the four `monarch_grid_<DATE>.csv` files from step 2, the trained model directory `out_ebutterfly_ablation_seed41_full/`, `ebutterfly_na_2011_2025.csv`, and `ebutterfly_splits.json`.
   - Output: `monarch_pred_<DATE>.csv` per date and `monarch_pred_full_1by4.png`.

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

# 3. inference + temperature scaling + 1×4 plot
export WORK_DIR=$(pwd)        # dir containing grid CSVs + trained model dir + train CSV + splits
export REPO_ROOT=$(cd ../.. && pwd)
jupyter notebook monarch_grid_inference.ipynb
```

Use `--workers` up to your core count; ARCO-ERA5 fetches and MODIS sampling parallelize across dates.

The inference notebook (`monarch_grid_inference.ipynb`) loads the trained STEM-LM full model, runs bagged inference at each of the four dates, fits the post-hoc temperature $T^\star$ on validation logits by NLL minimization, then writes one `monarch_pred_<DATE>.csv` per date and renders the figure. **`WORK_DIR` must contain** the four `monarch_grid_<DATE>.csv` files (from step 2), the trained model directory `out_ebutterfly_ablation_seed41_full/`, `ebutterfly_na_2011_2025.csv`, and `ebutterfly_splits.json`.

## Environmental data acquisition

Same sources as `data_processing/eButterfly` — the extraction routines referenced below live under `Examples/env_vars/<source>/` in the repo's data directory. `REPO_ROOT` defaults to two levels above this folder; override with `export REPO_ROOT=/path/to/STEM-LM` if running from elsewhere.

| Source | Product | Access | License | Citation |
|---|---|---|---|---|
| ARCO-ERA5 | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (Zarr) | Public GCS bucket, no account. Read directly via `xarray` + `gcsfs`. | Copernicus License | Hersbach et al. (2020); ARCO mirror by Google Public Datasets |
| Copernicus DEM | GLO-30 (30 m) | `/vsicurl/` from OpenTopography S3; no account. | Copernicus DEM License | ESA / Airbus (2021), DOI: 10.5270/ESA-c5d3d65 |
| MODIS MOD13Q1 v6.1 | Terra 16-day NDVI/EVI 250 m | NASA Earthdata HDF4; pre-downloaded NA tiles under `Examples/env_vars/modis_phenology/raw/`. | NASA open data | Didan (2021), DOI: 10.5067/MODIS/MOD13Q1.061 |
| SoilGrids v2.0 | 8 properties at 0–5 cm | WCS GetCoverage; download script under `Examples/env_vars/soilgrids/`. | CC BY 4.0 | Poggio et al. (2021), SOIL 7:217–240 |

## Notes

- The grid CSVs (~10,500 land cells × 15 covariates per date) are not committed; re-run the pipeline to regenerate.
- ERA5 orography is cached at `~/.cache/era5_orog_na.nc` after the first run.
- Hawaii is excluded via a longitude/latitude rectangle in `make_grid_na.py`; Great Lakes are masked at render time, not in the grid.
