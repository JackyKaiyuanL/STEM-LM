# CISO benchmark on eButterfly

Reproduces the CISO-SDM baseline on the eButterfly NA dataset for seeds 1337/1338/1339.

## What's here
- `ciso_train_benchmark_final_seed{1337,1338,1339}.ipynb` — train + evaluate at five `eval_known_ratio` levels.
- `results_CISO_butterfly_100_epochs_seed*/results_ratio_*.csv` — pre-computed per-mask metrics.
- `results_CISO_butterfly_100_epochs_seed*/ciso_benchmark_summary.csv` — aggregated summary.

## Required inputs (not bundled)
Stage these files inside `WORK_DIR` (defaults to the directory you launch the notebook from; override via `export WORK_DIR=/path/to/data`):
- `ebutterfly_na_2011_2025.csv` — eButterfly NA presence/absence + covariates.
- `ebutterfly_splits.json` — train/val/test split indices (from the data folder of this study).

The notebook auto-clones the upstream repo (`https://github.com/RolnickLab/CISO-SDM`) to `../CISO-SDM`; no manual clone needed.

## Running
```bash
cd benchmarks/CISO/eButterfly
export WORK_DIR=/path/to/dir/containing/csv_and_splits
jupyter notebook ciso_train_benchmark_final_seed1337.ipynb
```
The notebook will:
1. Generate the prep `.npy` / CSV files and `configs/config_ciso_*.yaml` from the input CSV + splits.
2. Train CISO for 100 epochs.
3. Evaluate at `eval_known_ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` and write metrics to `results_CISO_butterfly_100_epochs_seed*/`.

Checkpoints (`*.ckpt`) and per-hotspot prediction shards (`preds_ratio_*/`) are written to the working directory but excluded from this folder.
