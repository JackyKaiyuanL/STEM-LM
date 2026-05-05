# eButterfly — Data Processing

Reproduce the eButterfly dataset used for STEM-LM training and benchmarking, starting from the raw GBIF DwC-A archive.

## Source

GBIF DwC-A `cf3bdc30-370c-48d3-8fff-b587a39d72d6` (eButterfly), accessed 2026-04-13. Drop the unzipped archive at `<RAW>/event.txt` and `<RAW>/occurrence.txt`.

## Pipeline (run in order)

1. **`prepare_ebutterfly.py`** — protocol filter (structured checklists only), geographic filter (US + Canada + Mesoamerica, 2011–2025), species filter (≥100 presences), wide presence/absence CSV.
   - Output: `ebutterfly_na_2011_2025_jsdm.csv` (presence/absence + lat/lon/time only).

2. **`enrich_ebutterfly_na.py`** — adds 15 environmental covariates per observation date:
   - **ERA5-Land daily** 2 m temperature (min/max/mean) and total precipitation, lapse-rate corrected with Copernicus DEM at $-6.5\,^\circ\mathrm{C}\,\mathrm{km}^{-1}$.
   - **MODIS MOD13Q1** NDVI/EVI 16-day composite nearest the observation date.
   - **SoilGrids v2.0** 0–5 cm: bdod, cec, cfvo, clay, nitrogen, phh2o, sand, silt.
   - **Copernicus GLO-30 DEM** elevation.
   - Output: `ebutterfly_na_2011_2025.csv` (the file STEM-LM consumes).

3. **`regen_splits.py`** — H3 spatial-block split (resolution 2, seed 42, 80/10/10).
   - Output: `ebutterfly_splits.json`.

## Survey protocols kept

Traveling Survey, Area Survey, Timed Count, Point Count, Atlas Square, Pollard Walk, Pollard Transect.
Excluded: Incidental Observation(s) (presence-only), Historical (no reliable date).

## Environmental data acquisition

All four sources are public; the extraction scripts referenced below live under `Examples/env_vars/<source>/` in the repo's data directory and are invoked by `enrich_ebutterfly_na.py`.

| Source | Product | Access | License | Citation |
|---|---|---|---|---|
| ERA5-Land | `reanalysis-era5-land` (hourly → daily-aggregated) | Copernicus CDS API; requires free CDS account and `~/.cdsapirc` PAT. `pip install cdsapi`. | Copernicus License | Muñoz Sabater (2019), DOI: 10.24381/cds.68d2bb30 |
| Copernicus DEM | GLO-30 (30 m) | `/vsicurl/` from `https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/`; no account required. Tiles fetched on demand. | Copernicus DEM License | ESA / Airbus (2021), DOI: 10.5270/ESA-c5d3d65 |
| MODIS MOD13Q1 v6.1 | Terra 16-day NDVI/EVI 250 m | NASA Earthdata; requires free Earthdata account. Tiles downloaded as HDF4 (NA tiles only) via the script under `Examples/env_vars/modis_phenology/`. | NASA open data | Didan (2021), DOI: 10.5067/MODIS/MOD13Q1.061 |
| SoilGrids v2.0 | 8 properties at 0–5 cm | WCS GetCoverage from `https://maps.isric.org` per variable, regional bbox; download script under `Examples/env_vars/soilgrids/`. | CC BY 4.0 | Poggio et al. (2021), SOIL 7:217–240 |

## Notes

- Intermediate CSVs are not committed to the repo. Re-run the scripts to regenerate.
- Raster paths inside the scripts are absolute; edit the path constants at the top of each file to match your environment.
- ERA5-Land downloads are queue-bound on the CDS side; expect 10 min – 24 h per variable depending on demand. DEM tiles are immediate via `/vsicurl/`.
