"""Generate an H3 spatial-block split for the processed sPlotOpen CSV."""

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
    """Mirror STEMLM_data.h3_block_split."""
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

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_cells)
    shuffled = uniq_cells[perm]

    n_test_cells = max(1, round(n_cells * test_frac))
    n_val_cells = max(1, round(n_cells * (1 - train_frac - test_frac)))

    test_cells = set(shuffled[:n_test_cells])
    val_cells = set(shuffled[n_test_cells:n_test_cells + n_val_cells])
    train_cells = set(shuffled) - test_cells - val_cells

    train_idx = np.where(np.isin(cells, list(train_cells)))[0]
    val_idx = np.where(np.isin(cells, list(val_cells)))[0]
    test_idx = np.where(np.isin(cells, list(test_cells)))[0]
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
