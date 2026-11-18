#!/usr/bin/env python3
import argparse
import gc
import gzip
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_species_arg, load_dataset, resolve_output_dir, timed_phase, write_metrics  # noqa: E402
from stemlm.data import create_dataloaders, haversine_pairs_np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
METHODS = {
    "logistic": ("run_logistic.r", ("env", "spatiotemporal", "full")),
    "autologistic": ("run_autologistic.r", ("env", "spatiotemporal", "full")),
    "gam": ("run_gam.r", ("env", "spatiotemporal", "full")),
    "maxnet": ("run_maxnet.r", ("env",)),
    "rfsi": ("run_rfsi.r", ("env", "full")),
}
SOURCE_METHODS = ("autologistic", "rfsi")


def write_neighbors(dataset, prefix):
    shape = (len(dataset), dataset.num_source_sites)
    idx, dist, gap = np.empty(shape, np.int32), np.empty(shape, np.float32), np.empty(shape, np.float32)
    for start in range(0, len(dataset), 2048):
        chunk = list(range(start, min(start + 2048, len(dataset))))
        for i, item in zip(chunk, dataset.__getitems__(chunk), strict=True):
            src = item["source_idx"].numpy()
            idx[i] = src
            dist[i] = haversine_pairs_np(dataset.lats[i], dataset.lons[i], dataset.lats[src], dataset.lons[src])
            gap[i] = np.abs(dataset.times[i] - dataset.times[src])
    for name, values in (("idx", idx), ("dist", dist), ("gap", gap)):
        pd.DataFrame(values).to_csv(f"{prefix}_{name}.csv", index=False, header=False)


def write_autocovariate(dataset, out_path, idw):
    share = np.empty((len(dataset), dataset.num_species), dtype=np.float32)
    for start in range(0, len(dataset), 2048):
        chunk = list(range(start, min(start + 2048, len(dataset))))
        for i, item in zip(chunk, dataset.__getitems__(chunk), strict=True):
            species = item["source_species"].numpy()
            if idw:
                src = item["source_idx"].numpy()
                d = haversine_pairs_np(dataset.lats[i], dataset.lons[i], dataset.lats[src], dataset.lons[src])
                w = (d == 0).astype(np.float32) if (d == 0).any() else 1.0 / d
                share[i] = species @ (w / w.sum())
            else:
                share[i] = species.mean(1)
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
                        help="As in stemlm train: a training row's autologistic neighbors are its nearest "
                             "training rows; a validation or test row's are the training rows and the other "
                             "rows of its own split outside its H3 cell at this resolution.")
    parser.add_argument("--training_rows_exclude_cell", action="store_true",
                        help="A training row's autologistic neighbors also exclude its own H3 cell at "
                             "--source_cell_resolution.")
    parser.add_argument("--train_exclusion", action="store_true",
                        help="As in stemlm train: per training row, remove autologistic neighbors within a "
                             "random radius and time window.")
    parser.add_argument("--train_exclusion_seed", type=int, default=42)
    parser.add_argument("--train_sources_only", action="store_true",
                        help="Autologistic neighbors of every row are its nearest training rows, with no "
                             "cell exclusion. --source_cell_resolution is then unused.")
    parser.add_argument("--n_obs", type=int, default=5,
                        help="rfsi: number of nearest sources entered one by one (observation, distance and "
                             "time gap); all sources also enter as an equal-weight, an inverse-distance-weighted "
                             "and an inverse-time-gap-weighted share of sources with the species present.")
    parser.add_argument("--idw", action="store_true",
                        help="autologistic: weight the sources by inverse distance in the autocovariate; a source "
                             "at distance zero takes all the weight.")
    parser.add_argument("--no_shares", action="store_true",
                        help="rfsi: leave out the three shares over all sources.")
    parser.add_argument("--no_time", action="store_true",
                        help="As in stemlm train: sources are chosen without the time column, and rfsi gets "
                             "no time gaps.")
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
    if args.method in SOURCE_METHODS:
        loaders = create_dataloaders(str(args.csv_path.resolve()), splits_path=str(args.splits_path.resolve()),
                                     min_train_presences=args.min_train_presences,
                                     source_cell_resolution=args.source_cell_resolution,
                                     train_sources_only=args.train_sources_only,
                                     train_exclusion=args.train_exclusion, no_time=args.no_time)
        environment["AUTOCOV_SOURCES"] = "train" if args.train_sources_only else "heldout"
        loaders[1].training_rows_exclude_cell = args.training_rows_exclude_cell
        np.random.seed(args.train_exclusion_seed)
        if args.method == "autologistic":
            environment["AUTOCOV_FILE"] = str(output_dir / "autocovariate.csv")
            write_autocovariate(loaders[1], environment["AUTOCOV_FILE"], args.idw)
        else:
            environment.update(NEIGHBOR_PREFIX=str(output_dir / "neighbors"), N_OBS=str(args.n_obs),
                               HAS_TIME=str(int(loaders[1].has_time)), SHARES=str(int(not args.no_shares)))
            write_neighbors(loaders[1], environment["NEIGHBOR_PREFIX"])
        del loaders
    test_idx = np.sort(splits["test"])
    labels_all = df[species_cols].to_numpy(dtype=np.int64)[test_idx]
    train_presences = df[species_cols].to_numpy()[splits["train"]].sum(0)
    del df
    gc.collect()
    pyarrow.default_memory_pool().release_unused()
    with timed_phase(output_dir, "training"):
        subprocess.run([args.rscript, str(SCRIPT_DIR / script)], check=True, env=environment)
    if data_file != args.csv_path.resolve():
        data_file.unlink()

    results = []
    sources = (environment["AUTOCOV_SOURCES"],) if args.method in SOURCE_METHODS else ("none",)
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
    write_metrics(output_dir, args.method, results, species_cols, train_presences)
    for f in [*output_dir.glob("*/predictions_test_*.csv"), *output_dir.glob("autocovariate*.csv"),
              *output_dir.glob("neighbors_*.csv")]:
        with open(f, "rb") as src, gzip.open(f"{f}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        f.unlink()


if __name__ == "__main__":
    main()
