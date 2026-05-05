"""
Regenerate ebutterfly_splits_v2.json from the v2 CSV using the same h3
spatial-block fold as the original (resolution=2, seed=42, train/val/test
0.8/0.1/0.1).

Required because phase 1 / 3 can drop a handful of rows whose nearest valid
WorldClim or MODIS pixel is too far away.

Usage:
  python regen_splits_v2.py \\
      --in_csv  ${REPO_ROOT}/lab/ebutterfly_na_2011_2025_jsdm_v2.csv \\
      --out_json ${REPO_ROOT}/lab/ebutterfly_splits_v2.json \\
      --resolution 2 --seed 42

Self-contained — no model code needed. Mirrors jsdm_data.py:h3_block_split.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

T0 = time.monotonic()
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}  +{time.monotonic()-T0:7.1f}s]  {msg}",
          flush=True)


def h3_block_split(lats, lons, resolution=2, train_frac=0.8, test_frac=0.1, seed=42):
    """Mirror of jsdm_data.py's h3_block_split."""
    try:
        import h3
    except ImportError:
        sys.exit("ERROR: pip install h3")

    n = len(lats)
    cells = np.array([h3.latlng_to_cell(float(lats[i]), float(lons[i]), resolution)
                      for i in range(n)])
    uniq_cells = np.unique(cells)
    n_cells = len(uniq_cells)
    log(f"  {n} obs across {n_cells} unique h3 res-{resolution} cells")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cells)
    shuffled = uniq_cells[perm]

    n_train_cells = int(round(n_cells * train_frac))
    n_test_cells  = int(round(n_cells * test_frac))
    n_val_cells   = n_cells - n_train_cells - n_test_cells

    train_cells = set(shuffled[:n_train_cells])
    val_cells   = set(shuffled[n_train_cells:n_train_cells + n_val_cells])
    test_cells  = set(shuffled[n_train_cells + n_val_cells:])

    train_idx = np.array([i for i in range(n) if cells[i] in train_cells], dtype=np.int64)
    val_idx   = np.array([i for i in range(n) if cells[i] in val_cells],   dtype=np.int64)
    test_idx  = np.array([i for i in range(n) if cells[i] in test_cells],  dtype=np.int64)
    return train_idx, val_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log(f"Loading {args.in_csv}")
    df = pd.read_csv(args.in_csv, usecols=["latitude", "longitude"])
    log(f"  {len(df):,} rows")

    log(f"h3 res={args.resolution} block split (seed={args.seed}, "
        f"train={args.train_frac}, test={args.test_frac})")
    train_idx, val_idx, test_idx = h3_block_split(
        df["latitude"].values, df["longitude"].values,
        resolution=args.resolution,
        train_frac=args.train_frac, test_frac=args.test_frac, seed=args.seed,
    )
    log(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    out = {
        "num_rows": len(df),
        "meta": {
            "fold": "h3",
            "resolution": args.resolution,
            "train_frac": args.train_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "csv_path": os.path.basename(args.in_csv),
            "generated_without_full_run": True,
        },
        "train": train_idx.tolist(),
        "val":   val_idx.tolist(),
        "test":  test_idx.tolist(),
    }
    log(f"Writing {args.out_json}")
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    log(f"  {os.path.getsize(args.out_json)/1e6:.2f} MB")
    log("Done.")


if __name__ == "__main__":
    main()
