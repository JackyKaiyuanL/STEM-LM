# MaskSDM benchmark on sPlotOpen

Reproduces the MaskSDM-MEE baseline on the sPlotOpen v2.0 (global) dataset for seeds 1337/1338/1339.

## What's here
- `masksdm_benchmark_v2.ipynb` — train + evaluate at $p{=}1.0$ (fully unconditioned, matching the upstream MaskSDM evaluation protocol). Switch between seeds 1337/1338/1339 via the seed-suffixed `CKPT_DIR` / `RESULTS_DIR` paths in the setup cell.

## Required inputs (not bundled)
Stage these inside `WORK_DIR` (defaults to the launch directory; override via `export WORK_DIR=/path/to/data`):
- `splotopen_global.csv` — sPlotOpen global presence/absence + covariates.
- `splotopen_global_splits.json` — train/val/test split indices (from the data folder of this study).

The notebook auto-clones the upstream repo (`https://github.com/zbirobin/MaskSDM-MEE`) to `../MaskSDM-MEE`; no manual clone needed.

## Running
```bash
cd benchmarks/MaskSDM/sPlotOpen
export WORK_DIR=/path/to/dir/containing/csv_and_splits
jupyter notebook masksdm_benchmark_v2.ipynb
```
The notebook will:
1. Build numpy tensors from the CSV + splits.
2. Train an FT-Transformer for 1000 epochs.
3. Evaluate at $p{=}1.0$ and write the STEM-LM metric set (AUROC / AUPRC / CBI / Brier / ECE).

Per-epoch checkpoints (`epoch_*.pt`) are written to the working directory and excluded from this folder.
