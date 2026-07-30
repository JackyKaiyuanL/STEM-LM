# NEUS — Data Processing

Reproduce the NEUS bottom-trawl dataset used for STEM-LM training and benchmarking, starting from the FISHGLOB compilation.

## Source

FISHGLOB v1 (Maureaud et al. 2024, *Sci. Data* 11:24, DOI 10.1038/s41597-023-02866-w), survey `NEUS` — the NOAA NEFSC Northeast US bottom-trawl survey, Cape Hatteras to the Gulf of Maine, 1963–2020. Clone `https://github.com/fishglob/FishGlob_data` and point `--fishglob` at `outputs/Compiled_data/FishGlob_public_clean.RData`.

Only NEUS is used. The four other NW Atlantic surveys in FISHGLOB (SCS, GSL-N, GSL-S, SEUS) are excluded on two grounds: the five share only 165 of 848 taxa, so zero-filling a pooled list manufactures absences for taxa a survey never resolves; and month is nearly collinear with survey (every July haul is SCS, every May haul is NEUS), so pooling to close the calendar would confound season with region, depth and gear.

## Pipeline (run in order)

1. **`prepare_neus.py`** — extracts hauls and occurrences from the RData compilation via Rscript, filters years, drops hauls with incomplete date or covariates, zero-fills to presence/absence over taxa with ≥100 hauls, and attaches the three in-survey covariates.
   - Output: `neus_nefsc.csv` (the file STEM-LM consumes).

2. **`regen_splits.py`** — H3 spatial-block split, seed 42, 80/10/10.
   - Output: `neus_nefsc_splits.json`.
   - Use `--resolution 4`, not 2. The domain is 18.1° × 16.0°, about a tenth of the continental datasets: at resolution 2 the data falls in 15 cells with one cell holding 34.7% of hauls and a 10% test split drawn from 2 cells. Resolution 4 gives 77 cells, largest 3.9%, test drawn from 8.

## Environmental covariates

All three are measured at the haul by the survey itself, so `env_sbt` and `env_sst` are contemporaneous with the observation rather than interpolated from a reanalysis. No external extraction step is required.

| Column | Variable | Units | Coverage |
|---|---|---|---|
| `env_depth` | Bottom depth | m | 99.9% |
| `env_sbt` | Bottom temperature | °C | 87.3% |
| `env_sst` | Surface temperature | °C | 89.1% |

FISHGLOB passes NOAA's `BOTTEMP` and `SURFTEMP` through unchanged (`cleaning_codes/get_neus.R:211-212`); there is no documented fill convention. 104 hauls carry `sst` exactly 0.00 and 2 carry `sbt` exactly 0.00, against 113 hauls below 1 °C in total. These are left as-is rather than recoded.

Salinity is the covariate most obviously missing and is not obtainable in situ here; GLORYS12V1 would supply it from 1993 onward.

## Survey caveats to declare

- **2009 gear change.** The *Albatross IV* was replaced by the *Henry B. Bigelow*, with tow duration dropping from 30 to 20 minutes and stratum count from ~114 to 81. Mean richness per haul steps from 10.56 (1963–2008) to 14.54 (2009–2019). FISHGLOB applies the published calibration factors, but those correct abundance, not detection: a species undetected before 2009 and detected after flips 0→1 for gear reasons. Trim with `--year_min`/`--year_max` to stay inside one gear era if this matters for the analysis at hand.
- **Effort is not constant.** Hauls per year range 132 to 1,063 (median 647, CV 0.28); stations run 183 → 552 → ~340 → ~400 across the record; the 1960s cover 38–42 strata against 110+ later.
- **Absence is gear-conditional.** A bottom trawl undersamples pelagic, small-bodied and burrowing species, so a zero means not caught by this gear at this station.
- **Calendar gap.** NEUS runs Spring (Feb–Jun) and Fall (Sep–Dec); January, July and August are absent.

## Notes
- Intermediate CSVs are written to `--workdir` and are not committed. Re-run to regenerate.
- Presence is defined by the existence of an occurrence row, never by `num` or `wgt`: 5.1% of records across the NW Atlantic surveys carry `wgt == 0` from pre-1997 dial-scale rounding and are genuine presences.
