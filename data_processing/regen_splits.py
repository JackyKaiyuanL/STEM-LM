#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemlm.data import h3_block_split, save_splits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="H3 spatial-block train/val/test split of a processed dataset CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--test_frac", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, usecols=["latitude", "longitude"])
    train, val, test = h3_block_split(df["latitude"].values, df["longitude"].values,
                                      resolution=args.resolution, train_frac=args.train_frac,
                                      test_frac=args.test_frac, seed=args.seed)
    save_splits(args.splits_path, train, val, test, num_rows=len(df),
                meta={"fold": "h3", "resolution": args.resolution, "train_frac": args.train_frac,
                      "test_frac": args.test_frac, "seed": args.seed,
                      "csv_path": os.path.basename(args.csv_path)})
    print(f"{len(df)} rows -> train {len(train)} / val {len(val)} / test {len(test)} -> {args.splits_path}")


if __name__ == "__main__":
    main()
