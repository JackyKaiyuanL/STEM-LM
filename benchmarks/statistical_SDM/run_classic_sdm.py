#!/usr/bin/env python3
import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_species_arg, load_dataset, resolve_output_dir, timed_phase, write_metrics  # noqa: E402
from stemlm.data import create_dataloaders  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
METHODS = {
    "logistic": ("run_logistic.r", ("env", "spatiotemporal", "full")),
    "autologistic": ("run_autologistic.r", ("env", "spatiotemporal", "full")),
    "gam": ("run_gam.r", ("env", "spatiotemporal", "full")),
    "maxnet": ("run_maxnet.r", ("env",)),
}


def write_autocovariate(dataset, out_path):
    share = np.empty((len(dataset), dataset.num_species), dtype=np.float32)
    for start in range(0, len(dataset), 2048):
        items = dataset.__getitems__(list(range(start, min(start + 2048, len(dataset)))))
        share[start:start + len(items)] = np.stack([it["source_species"].numpy().mean(1) for it in items])
    columns = ["auto_" + re.sub(r"[^A-Za-z0-9]", "_", s) for s in dataset.species_cols]
    pd.DataFrame(share, columns=columns).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate one classic SDM on one dataset and split.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--splits_path", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--n_cores", type=int, default=8)
    parser.add_argument("--source_cell_resolution", type=int, default=7,
                        help="As in stemlm train: held-out rows outside the target's H3 cell at this "
                             "resolution join the training rows as autologistic neighbours.")
    add_species_arg(parser)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args, __file__, prefix=args.method)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, _, species_cols, splits = load_dataset(args.csv_path.resolve(), args.splits_path.resolve(),
                                               args.min_train_presences)
    species_file = output_dir / "species.txt"
    species_file.write_text("\n".join(species_cols) + "\n")
    script, cov_sets = METHODS[args.method]
    data_file = args.csv_path.resolve()
    if data_file.suffix == ".parquet":
        data_file = output_dir / "data.csv"
        df.to_csv(data_file, index=False)
    environment = {**os.environ, "DATA_FILE": str(data_file),
                   "SPLITS_FILE": str(args.splits_path.resolve()),
                   "SPECIES_FILE": str(species_file),
                   "RESULTS_DIR": str(output_dir), "N_CORES": str(args.n_cores),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    if args.method == "autologistic":
        _, dataset, _, _ = create_dataloaders(str(args.csv_path.resolve()), splits_path=str(args.splits_path.resolve()),
                                              min_train_presences=args.min_train_presences,
                                              source_cell_resolution=args.source_cell_resolution)
        environment["AUTOCOV_FILE"] = str(output_dir / "autocovariate.csv")
        write_autocovariate(dataset, environment["AUTOCOV_FILE"])
        dataset.heldout_split = None
        environment["AUTOCOV_TRAIN_FILE"] = str(output_dir / "autocovariate_train_sources.csv")
        write_autocovariate(dataset, environment["AUTOCOV_TRAIN_FILE"])
    with timed_phase(output_dir, "training"):
        subprocess.run([args.rscript, str(SCRIPT_DIR / script)], check=True, env=environment)
    if data_file != args.csv_path.resolve():
        data_file.unlink()

    test_idx = np.sort(splits["test"])
    labels_all = df[species_cols].to_numpy(dtype=np.int64)[test_idx]
    results = []
    sources = ("heldout", "train") if args.method == "autologistic" else ("none",)
    for cov_set in cov_sets:
        for src in sources:
            name = "predictions_test_all.csv" if src == "none" else f"predictions_test_{src}_all.csv"
            pred = pd.read_csv(output_dir / cov_set / name)
            z = (pred.pivot(index="row_index", columns="species", values="logit")
                 .reindex(index=test_idx, columns=species_cols).to_numpy(dtype=np.float64))
            fitted = ~np.isnan(z).all(axis=0)
            if not np.isfinite(z[:, fitted]).all():
                raise ValueError(f"{cov_set}: non-finite logits in {output_dir / cov_set / name}")
            results.append(({"cov_set": cov_set, "masking_p": 1.0, "sources": src},
                            np.where(fitted, z, 0.0),
                            np.where(fitted, labels_all, -100)))
    write_metrics(output_dir, args.method, results, species_cols,
                  df[species_cols].to_numpy()[splits["train"]].sum(0))
    for f in [*output_dir.glob("*/predictions_test_*.csv"), *output_dir.glob("autocovariate*.csv")]:
        with open(f, "rb") as src, gzip.open(f"{f}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        f.unlink()


if __name__ == "__main__":
    main()
