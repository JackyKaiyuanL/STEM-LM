# eButterfly — Data Processing

Reproduce the eButterfly dataset used for STEM-LM training and benchmarking, starting from the raw GBIF DwC-A archive.

## Source

GBIF DwC-A `cf3bdc30-370c-48d3-8fff-b587a39d72d6` (eButterfly), accessed 2026-04-13. Drop the unzipped archive at `<RAW>/event.txt` and `<RAW>/occurrence.txt`.

## Pipeline (run in order)

1. **`prepare_ebutterfly.py`** — protocol filter (structured checklists only), geographic filter (US + Canada + Mesoamerica, 2011–2025), species filter (≥100 presences), wide presence/absence CSV.
   - Output: `ebutterfly_na_2011_2025_jsdm.csv` (presence/absence + lat/lon/time only).

2. **`enrich_ebutterfly_na.py`** — adds 15 environmental covariates per observation date:
   - **ARCO-ERA5 daily** 2 m temperature (min/max/mean) and total precipitation at 0.25°, lapse-rate corrected with Copernicus DEM at $-6.5\,^\circ\mathrm{C}\,\mathrm{km}^{-1}$.
   - **MODIS MOD13Q1** NDVI/EVI 16-day composite nearest the observation date.
   - **SoilGrids v2.0** 0–5 cm: bdod, cec, cfvo, clay, nitrogen, phh2o, sand, silt.
   - **Copernicus GLO-30 DEM** elevation.
   - Output: `ebutterfly_na_2011_2025.csv` (the file STEM-LM consumes).

3. **`../regen_splits.py`** — H3 spatial-block split (resolution 2, 80/10/10), one file per split seed (41, 42, 43).
   - `python ../regen_splits.py ebutterfly_na_2011_2025.csv --splits_path ebutterfly_splits_seed41.json --resolution 2 --seed 41`
   - Output: `ebutterfly_splits_seed{41,42,43}.json`.

## Survey protocols kept

Traveling Survey, Area Survey, Timed Count, Point Count, Atlas Square, Pollard Walk, Pollard Transect.
Excluded: Incidental Observation(s) (presence-only), Historical (no reliable date).

## Environmental data acquisition

All four sources are public; the download and extraction scripts live under `data_processing/covariates/<source>/` and are invoked by `enrich_ebutterfly_na.py`.

| Source | Product | Access | License | Citation |
|---|---|---|---|---|
| ARCO-ERA5 | ERA5 hourly, 0.25° (public Zarr mirror), aggregated to daily | Anonymous Google Cloud read; the one-off orography download for the lapse-rate correction needs a CDS account and `~/.cdsapirc`. | Copernicus License | Hersbach et al. (2020), Q. J. R. Meteorol. Soc. 146:1999–2049 |
| Copernicus DEM | GLO-30 (30 m) | `/vsicurl/` from `https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/`; no account required. Tiles fetched on demand. | Copernicus DEM License | ESA / Airbus (2021), DOI: 10.5270/ESA-c5d3d65 |
| MODIS MOD13Q1 v6.1 | Terra 16-day NDVI/EVI 250 m | NASA Earthdata; requires free Earthdata account. Tiles downloaded as HDF4 (NA tiles only) via `covariates/modis_phenology/download_mod13q1_na.py`. | NASA open data | Didan (2021), DOI: 10.5067/MODIS/MOD13Q1.061 |
| SoilGrids v2.0 | 8 properties at 0–5 cm | WCS GetCoverage from `https://maps.isric.org` per variable, regional bbox; download scripts under `covariates/soilgrids/`. | CC BY 4.0 | Poggio et al. (2021), SOIL 7:217–240 |

## Notes

- Intermediate CSVs are not committed to the repo. Re-run the scripts to regenerate.
- Raw rasters are read from `ENV_DIR` (default `${REPO_ROOT}/Examples/env_vars`); set it to where the downloads live.
- ERA5 is read from the public Zarr store, one month per request; DEM tiles are immediate via `/vsicurl/`.
