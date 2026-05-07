# CISO benchmark on sPlotOpen

Reproduces the CISO-SDM baseline on the sPlotOpen v2.0 (global) dataset for seeds 1337/1338/1339.

## What's here
- `ciso_train_benchmark_final.ipynb` — train + evaluate at four `eval_known_ratio` levels. Set the `'seed'` field in the YAML config block (and the seed-suffixed `CKPT_DIR` / `RESULTS_DIR` paths) to switch between 1337/1338/1339.
- `main.py` — modified upstream `main.py` with `CSVLogger` instead of `CometLogger`. Drop into the cloned `CISO-SDM/` repo before training.

## Required inputs (not bundled)
Stage these inside `WORK_DIR` (defaults to the launch directory; override via `export WORK_DIR=/path/to/data`):
- `splotopen_global.csv` — sPlotOpen global presence/absence + covariates.
- `splotopen_global_splits.json` — train/val/test split indices (from the data folder of this study).

The notebook auto-clones the upstream repo (`https://github.com/RolnickLab/CISO-SDM`) to `../CISO-SDM`; no manual clone needed.

## Running
```bash
cd benchmarks/CISO/sPlotOpen
export WORK_DIR=/path/to/dir/containing/csv_and_splits
jupyter notebook ciso_train_benchmark_final.ipynb
```
The notebook will:
1. Generate the prep `.npy` / CSV files and `configs/config_ciso_*.yaml` from the input CSV + splits.
2. Train CISO for 50 epochs.
3. Evaluate at `eval_known_ratio ∈ {0.0, 0.25, 0.5, 0.75}` and write metrics to `RESULTS_DIR`.

Checkpoints (`*.ckpt`) and per-hotspot prediction shards (`preds_ratio_*/`) are written to the working directory but excluded from this folder.
