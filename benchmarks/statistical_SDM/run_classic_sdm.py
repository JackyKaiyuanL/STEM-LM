#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_species_arg, load_dataset, resolve_output_dir, timed_phase, write_metrics  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
METHODS = {
    "logistic": ("run_logistic.r", ("env", "spatiotemporal", "full")),
    "gam": ("run_gam.r", ("env", "spatiotemporal", "full")),
    "maxnet": ("run_maxnet.r", ("env",)),
}


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate one classic SDM on one dataset and split.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--splits_path", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--n_cores", type=int, default=8)
    add_species_arg(parser)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args, __file__, prefix=args.method)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, _, species_cols, splits = load_dataset(args.csv_path.resolve(), args.splits_path.resolve(),
                                               args.min_train_presences)
    species_file = output_dir / "species.txt"
    species_file.write_text("\n".join(species_cols) + "\n")
    script, cov_sets = METHODS[args.method]
    environment = {**os.environ, "DATA_FILE": str(args.csv_path.resolve()),
                   "SPLITS_FILE": str(args.splits_path.resolve()),
                   "SPECIES_FILE": str(species_file),
                   "RESULTS_DIR": str(output_dir), "N_CORES": str(args.n_cores),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with timed_phase(output_dir, "training"):
        subprocess.run([args.rscript, str(SCRIPT_DIR / script)], check=True, env=environment)

    test_idx = np.sort(splits["test"])
    labels_all = df[species_cols].to_numpy(dtype=np.int64)[test_idx]
    results = []
    for cov_set in cov_sets:
        pred = pd.read_csv(output_dir / cov_set / "predictions_test_all.csv")
        z = (pred.pivot(index="row_index", columns="species", values="logit")
             .reindex(index=test_idx, columns=species_cols).to_numpy(dtype=np.float64))
        fitted = ~np.isnan(z).all(axis=0)
        if not np.isfinite(z[:, fitted]).all():
            raise ValueError(f"{cov_set}: non-finite logits in {output_dir / cov_set}")
        results.append(({"cov_set": cov_set, "masking_p": 1.0},
                        np.where(fitted, z, 0.0),
                        np.where(fitted, labels_all, -100)))
    write_metrics(output_dir, args.method, results, species_cols,
                  df[species_cols].to_numpy()[splits["train"]].sum(0))


if __name__ == "__main__":
    main()
