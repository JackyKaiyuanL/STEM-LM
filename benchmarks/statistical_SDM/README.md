# Statistical-SDM benchmarks

Per-species Logistic regression, Maxnet, and GAM baselines for STEM-LM.

## Software versions

- R 4.5.0
- mgcv 1.9.1 (provides `bam` for GAM)
- maxnet (CRAN)
- pROC, PRROC, jsonlite, parallel

## Pipeline

1. **`run_logisticReg.r`** — per-species GLM with logistic link. Three covariate sets per species: `env`, `spatiotemporal`, `full`. Fourier(DOY) basis at 365/182/122/91-day periods.
2. **`run_gam.r`** — same three covariate sets via `mgcv::bam`. Spatial: thin-plate `s(lat, lon, k=50)`; temporal: cyclic cubic `s(doy, bs="cc", k=12)`.
3. **`run_maxnet.r`** — environmental covariates only. Two-pass: (1) optimize `reg_mult` on 20 random species by max val AUROC; (2) refit all species at the selected reg_mult.

Each writes combined per-split prediction CSVs (`predictions_{train,val,test}_all.csv`) under `<DATASET>/results/<method>/<cov_set>/`.

## Eval

After the runs above, compute STEM-LM-faithful metrics (AUROC mean + q25/q50/q75, AUPRC, CBI, Brier, ECE):

- **`eval_logisticReg.r`**, **`eval_gam.r`**, **`eval_maxnet.r`** — read combined predictions and write `metrics_summary.csv` per method.
- **`stemlm_metrics.R`** — sourced helper. CBI matches `STEMLM_metric.safe_cbi` (background-only "expected" denominator, `min_per_window=10` floor).

## Defaults

- `DATASET=ebutterfly` (override via env var).
- `DATA_FILE` and `SPLITS_FILE` default to repo-relative paths inferred from script location.
- `N_CORES=32` for parallel mclapply.
