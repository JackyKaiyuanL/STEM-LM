import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemlm.data import MIN_TRAIN_PRESENCES, load_splits, species_with_presences  # noqa: E402
from stemlm.metric import compute_per_species_metrics, summarize_per_species_metrics  # noqa: E402

META_COLS = ("time", "latitude", "longitude")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_output_dir(args, script_file, prefix=None):
    if args.output_dir is not None:
        return args.output_dir.resolve()
    parts = [prefix, Path(args.csv_path).stem, Path(args.splits_path).stem]
    if getattr(args, "seed", None) is not None:
        parts.append(f"seed{args.seed}")
    return Path(script_file).resolve().parent / "output" / "__".join(p for p in parts if p)


def add_species_arg(parser):
    parser.add_argument("--min_train_presences", type=int, default=MIN_TRAIN_PRESENCES,
                        help="Keep the species with at least this many presences in the training rows.")


def load_dataset(csv_path, split_path, min_train_presences):
    df = pd.read_csv(csv_path)
    env_cols = [c for c in df.columns if c.startswith("env_")]
    species_cols = [c for c in df.columns if c not in META_COLS and not c.startswith("env_")]
    train, val, test = load_splits(str(split_path), expected_num_rows=len(df))
    keep = species_with_presences(df[species_cols].to_numpy(dtype=np.float32), train, min_train_presences)
    return df, env_cols, [species_cols[i] for i in keep], {"train": train, "val": val, "test": test}


@contextmanager
def timed_phase(output_dir, name):
    path = Path(output_dir) / "phase_times.json"
    times = json.loads(path.read_text()) if path.exists() else {}
    started_utc, started = utc_now(), time.monotonic()
    status = "failed"
    try:
        yield
        status = "succeeded"
    finally:
        times[name] = {"started_utc": started_utc, "ended_utc": utc_now(),
                       "wall_seconds": time.monotonic() - started, "status": status}
        path.write_text(json.dumps(times, indent=2) + "\n")


def write_metrics(output_dir, model, results, species, train_counts):
    summary_rows, species_rows = [], []
    for setting, logits, labels in results:
        per_sp = compute_per_species_metrics(logits, labels)
        summary_rows.append({"model": model, **setting, **summarize_per_species_metrics(per_sp)})
        for s in sorted(per_sp["auc_roc"]):
            species_rows.append({"model": model, **setting, "species": species[s],
                                 "train_presence_count": int(train_counts[s]),
                                 **{name: values.get(s, np.nan) for name, values in per_sp.items()}})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(Path(output_dir) / "summary.csv", index=False)
    pd.DataFrame(species_rows).to_csv(Path(output_dir) / "per_species_metrics.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return summary
