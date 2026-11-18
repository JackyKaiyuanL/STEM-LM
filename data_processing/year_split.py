#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemlm.data import read_table, save_splits  # noqa: E402


def latest_years(counts, rows, frac):
    share = counts.cumsum() / rows
    return [int(y) for y in counts.index[: int(np.argmax(share.to_numpy() >= frac)) + 1]]


def main():
    parser = argparse.ArgumentParser(description="Train/val/test split by year: the latest whole years are held out.")
    parser.add_argument("csv_path")
    parser.add_argument("--splits_path", required=True)
    parser.add_argument("--frac", type=float, default=0.1,
                        help="Test takes the latest whole years until they hold this share of the rows; "
                             "validation then takes the next latest years by the same rule.")
    args = parser.parse_args()

    year = pd.to_datetime(read_table(args.csv_path)["time"]).dt.year
    counts = year.value_counts().sort_index(ascending=False)
    test_years = latest_years(counts, len(year), args.frac)
    val_years = latest_years(counts.drop(test_years), len(year), args.frac)
    test, val = year.isin(test_years), year.isin(val_years)
    train = ~test & ~val
    save_splits(args.splits_path, np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test), num_rows=len(year),
                meta={"fold": "year", "frac": args.frac, "test_years": test_years, "val_years": val_years,
                      "table": os.path.basename(args.csv_path)})
    print(f"{len(year)} rows -> train {train.sum()} / val {val.sum()} (years {val_years}) / "
          f"test {test.sum()} (years {test_years}) -> {args.splits_path}")


if __name__ == "__main__":
    main()
