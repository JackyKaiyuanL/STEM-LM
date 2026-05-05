# sPlotOpen — Data Processing

Reproduce the sPlotOpen dataset used for STEM-LM training and benchmarking, starting from the raw v76 archive.

## Source

iDiv sPlotOpen v76 (Sabatini et al. 2021). Drop the unzipped archive under `Examples/sPlotOpen/raw/iDiv Data Repository_3474_v76__20260408/` (header, DT, species CSVs).

## Pipeline (run in order)

1. **`prepare_splotopen.py`** — extracts species occurrences (≥100 presences) and plot metadata (lat/lon, no time). Global, no climate-balanced subsample filter.
   - Output: `splotopen_global_jsdm.csv` (presence/absence + lat/lon only; `time` column 0 for all rows).

2. **`../covariates/extract_env_static.py`** — adds 28 static environmental covariates per plot:
   - **WorldClim v2.1** 19 bioclimatic variables (`env_bio01..env_bio19`).
   - **SoilGrids v2.0** 0–5 cm: 8 soil variables (`env_soil_*`).
   - **Copernicus GLO-30 DEM** elevation (`env_dem`).
   - Output: `splotopen_global.csv` (the file STEM-LM consumes).

3. **`regen_splits.py`** — H3 spatial-block split (resolution 2, seed 42, 80/10/10).
   - Output: `splotopen_global_splits.json`.

## Environmental data acquisition

All three sources are public; the extraction is read-only via rasterio (no per-plot API calls).

| Source | Product | Access | License | Citation |
|---|---|---|---|---|
| WorldClim v2.1 | 19 bioclimatic variables, 30 arcsec (~1 km), 1970–2000 normals | Direct download `wc2.1_30s_bio.zip` from worldclim.org; read in-place via `/vsizip/`. | CC BY 4.0 | Fick & Hijmans (2017), Int. J. Climatol. 37:4302–4315 |
| SoilGrids v2.0 | 8 properties at 0–5 cm | WCS GetCoverage from `https://maps.isric.org` per variable, regional bbox; download script under `Examples/env_vars/soilgrids/`. | CC BY 4.0 | Poggio et al. (2021), SOIL 7:217–240 |
| Copernicus DEM | GLO-30 (30 m) | `/vsicurl/` from `https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/`; no account required. Tiles fetched on demand. | Copernicus DEM License | ESA / Airbus (2021), DOI: 10.5270/ESA-c5d3d65 |

## Notes

- Plant distributions are treated as approximately time-invariant; no temporal covariates are extracted.
- Intermediate CSVs are not committed. Re-run to regenerate.
- Raster paths in `extract_env_static.py` are absolute; edit the constants if your raw data lives elsewhere.
