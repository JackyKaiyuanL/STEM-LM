# MaskSDM benchmark on eButterfly

Reproduces the MaskSDM-MEE baseline on the eButterfly NA dataset for seeds 1337/1338/1339.

## What's here
- `masksdm_benchmark_seed{1337,1338,1339}.ipynb` — train + evaluate at five masking levels.
- `results_masksdm_butterfly_1000_epochs_seed*/masksdm_benchmark_summary.{csv,json}` — pre-computed summaries.

## Required inputs (not bundled)
Stage these inside `WORK_DIR` (defaults to the launch directory; override via `export WORK_DIR=/path/to/data`):
- `ebutterfly_na_2011_2025.csv`
- `ebutterfly_splits.json`

The notebook auto-clones the upstream repo (`https://github.com/zbirobin/MaskSDM-MEE`) to `../MaskSDM-MEE`; no manual clone needed.

## Running
```bash
cd benchmarks/MaskSDM/eButterfly
export WORK_DIR=/path/to/dir/containing/csv_and_splits
jupyter notebook masksdm_benchmark_seed1337.ipynb
```
The notebook will:
1. Build numpy tensors from the CSV + splits.
2. Train an FT-Transformer for 1000 epochs.
3. Evaluate at masking fractions matching CISO's `eval_known_ratio` schedule and write metrics.

Per-epoch checkpoints (`epoch_*.pt`) are written to the working directory and excluded from this folder.
