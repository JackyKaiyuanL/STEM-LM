# Environmental covariate acquisition

Scripts and protocol notes for the public data sources used by the eButterfly, sPlotOpen and Monarch-grid pipelines. Raw downloads live under `ENV_DIR` (environment variable, default `${REPO_ROOT}/Examples/env_vars`), one subfolder per source.

## Layout

```
covariates/
├── extract_env_static.py         # Joins WorldClim + SoilGrids + DEM onto a (lat, lon) CSV.
├── era5/
│   ├── extract_era5_at_points.py # Per-(lat, lon, date) daily ERA5 extractor with lapse-rate correction.
│   └── PROTOCOL.txt
├── modis_phenology/
│   ├── download_mod13q1_na.py    # Bulk HDF4 download from NASA LP DAAC for NA tiles.
│   ├── mod13q1.py                # Nearest-composite NDVI/EVI sampling at (lat, lon, date).
│   └── PROTOCOL.txt
├── soilgrids/
│   ├── download_soilgrids_na.sh
│   ├── download_soilgrids_global_tiles.sh
│   └── PROTOCOL.txt
├── worldclim/
│   └── PROTOCOL.txt              # No download script: grab wc2.1_30s_bio.zip directly.
└── copernicus_dem/
    ├── COP30_hh_vsicurl.vrt      # GDAL VRT pointing at the public S3 bucket; no local tiles needed.
    └── PROTOCOL.txt
```

## Source summary

| Source | Product | Used by | License | Citation |
|---|---|---|---|---|
| ARCO-ERA5 | ERA5 hourly, 0.25°, aggregated to daily | eButterfly, Monarch grid | Copernicus License | Hersbach et al. (2020), Q. J. R. Meteorol. Soc. 146:1999–2049 |
| MODIS MOD13Q1 v6.1 | Terra 16-day NDVI/EVI 250 m | eButterfly, Monarch grid | NASA open data | Didan (2021), DOI: 10.5067/MODIS/MOD13Q1.061 |
| SoilGrids v2.0 | 8 properties, 0–5 cm | eButterfly, sPlotOpen, Monarch grid | CC BY 4.0 | Poggio et al. (2021), SOIL 7:217–240 |
| WorldClim v2.1 | 19 bioclimatic variables, 30 arcsec | sPlotOpen | CC BY 4.0 | Fick & Hijmans (2017), Int. J. Climatol. 37:4302–4315 |
| Copernicus DEM | GLO-30 (30 m) | eButterfly, sPlotOpen, Monarch grid | Copernicus DEM License | ESA / Airbus (2021), DOI: 10.5270/ESA-c5d3d65 |

## Access notes

- **ARCO-ERA5** is a public Google Cloud Zarr store read anonymously; the one-off ERA5 orography download for the lapse-rate correction needs a Copernicus CDS account and `~/.cdsapirc` (see `era5/PROTOCOL.txt`).
- **MODIS** requires a free NASA Earthdata account. Tiles are HDF4 (read with pyhdf, or rasterio + libgdal-hdf4). Bulk download for the NA tile set is ~30 GB.
- **SoilGrids** is fetched per-variable per-region via WCS GetCoverage. The shell scripts pull the regional GeoTIFFs needed for NA / Europe / Australia.
- **WorldClim** has no API: download `wc2.1_30s_bio.zip` from worldclim.org and read in-place via `/vsizip/`.
- **Copernicus DEM** tiles are public on `https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh/`. The `/vsicurl/` VRT lets rasterio fetch tiles on demand without storing any locally.

## What each pipeline calls

- **eButterfly** (`../eButterfly/enrich_ebutterfly_na.py`) calls `era5/extract_era5_at_points.py` on the observation rows, `modis_phenology/mod13q1.py` for NDVI/EVI, and samples SoilGrids + DEM via rasterio.
- **Monarch grid** (`../Monarch/make_grid_na.py`, then `../Monarch/enrich_grid_daily.py`) samples SoilGrids + DEM once, then calls the same ERA5 extractor and MODIS sampler on the grid cells for each map date.
- **sPlotOpen** (`../sPlotOpen/prepare_splotopen.py` then `extract_env_static.py`) samples WorldClim + SoilGrids + DEM only; no temporal layer is needed.
