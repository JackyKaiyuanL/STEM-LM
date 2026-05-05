# Environmental covariate acquisition

Scripts and protocol notes for the public data sources used by the eButterfly and sPlotOpen pipelines. Per-source subfolders contain the download script(s) and the original `PROTOCOL.txt` from `Examples/env_vars/`.

## Layout

```
covariates/
├── extract_env_static.py        # Joins WorldClim + SoilGrids + DEM onto a (lat, lon) CSV.
├── era5_land/
│   ├── download_era5_land_na.py # Hourly NetCDF download via Copernicus CDS API.
│   ├── extract_era5_at_points.py # Per-(lat, lon, date) extractor used by eButterfly enrichment.
│   └── PROTOCOL.txt
├── modis_phenology/
│   ├── download_mod13q1_na.py   # Bulk HDF4 download from NASA LP DAAC for NA tiles.
│   └── PROTOCOL.txt
├── soilgrids/
│   ├── download_soilgrids_na.sh
│   ├── download_soilgrids_global_tiles.sh
│   └── PROTOCOL.txt
├── worldclim/
│   └── PROTOCOL.txt             # No download script: grab wc2.1_30s_bio.zip directly.
└── copernicus_dem/
    ├── COP30_hh_vsicurl.vrt     # GDAL VRT pointing at the public S3 bucket — no local tiles needed.
    └── PROTOCOL.txt
```

## Source summary

| Source | Product | Used by | License | Citation |
|---|---|---|---|---|
| ERA5-Land | `reanalysis-era5-land` (hourly → daily-aggregated) | eButterfly | Copernicus License | Muñoz Sabater (2019), DOI: 10.24381/cds.68d2bb30 |
| MODIS MOD13Q1 v6.1 | Terra 16-day NDVI/EVI 250 m | eButterfly | NASA open data | Didan (2021), DOI: 10.5067/MODIS/MOD13Q1.061 |
| SoilGrids v2.0 | 8 properties, 0–5 cm | eButterfly, sPlotOpen | CC BY 4.0 | Poggio et al. (2021), SOIL 7:217–240 |
| WorldClim v2.1 | 19 bioclimatic variables, 30 arcsec | sPlotOpen | CC BY 4.0 | Fick & Hijmans (2017), Int. J. Climatol. 37:4302–4315 |
| Copernicus DEM | GLO-30 (30 m) | eButterfly, sPlotOpen | Copernicus DEM License | ESA / Airbus (2021), DOI: 10.5270/ESA-c5d3d65 |

## Access notes

- **ERA5-Land** requires a free Copernicus CDS account and `~/.cdsapirc` PAT. `pip install cdsapi`. Downloads are queue-bound (10 min – 24 h per variable depending on demand).
- **MODIS** requires a free NASA Earthdata account. Tiles are HDF4 (read with rasterio + libgdal-hdf4). Bulk download for the NA tile set is ~30 GB.
- **SoilGrids** is fetched per-variable per-region via WCS GetCoverage. The shell scripts pull the regional GeoTIFFs needed for NA / Europe / Australia.
- **WorldClim** has no API: download `wc2.1_30s_bio.zip` from worldclim.org and read in-place via `/vsizip/`.
- **Copernicus DEM** tiles are public on `https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/`. The `/vsicurl/` VRT lets rasterio fetch tiles on demand without storing any locally.

## What each pipeline calls

- **eButterfly** (`../eButterfly/enrich_ebutterfly_na.py`) calls `era5_land/extract_era5_at_points.py` per row, joins MODIS HDF4 tiles by date, and samples SoilGrids + DEM via rasterio.
- **sPlotOpen** (`../sPlotOpen/prepare_splotopen.py` then `extract_env_static.py`) samples WorldClim + SoilGrids + DEM only; no temporal layer is needed.
