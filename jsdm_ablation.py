import argparse
import json
import os
import subprocess
import sys
from typing import List

ABLATION_MODES = ("full", "no_st", "no_env", "no_st_env")

JSDM_TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jsdm_train.py")


def main():
    ap = argparse.ArgumentParser(
        description="Ablation master — runs jsdm_train.py once per mode.",
        allow_abbrev=False,
    )
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--output_dir", type=str, default="./ablation_output",
                    help="Parent dir. Each mode gets a subdirectory named after it.")
    ap.add_argument("--modes", nargs="+", default=list(ABLATION_MODES),
                    choices=ABLATION_MODES,
                    help="Which modes to run (default: all four).")
    ap.add_argument("--shared_splits_from", type=str, default=None,
                    help="Optional: explicit splits.json to use for all modes. "
                         "If omitted, the first mode's splits.json is reused by "
                         "the rest for a self-consistent comparison.")
    
    args, passthrough = ap.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shared_splits = args.shared_splits_from

    for i, mode in enumerate(args.modes):
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        cmd: List[str] = [
            sys.executable, JSDM_TRAIN, args.csv_path,
            "--ablation", mode,
            "--output_dir", mode_dir,
        ]
        if shared_splits is not None:
            cmd += ["--splits_path", shared_splits]
        cmd += passthrough

        print("\n" + "=" * 70)
        print(f"[ablation] mode {i+1}/{len(args.modes)} = {mode}")
        print(f"[ablation] cmd: {' '.join(cmd)}")
        print("=" * 70, flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"[ablation] mode {mode!r} failed (exit {r.returncode})")
            

        if shared_splits is None:
            splits_path = os.path.join(mode_dir, "splits.json")
            if os.path.exists(splits_path):
                shared_splits = splits_path
                print(f"[ablation] subsequent modes will use splits from {shared_splits}")

    # Aggregate summaries
    rows = []
    for mode in args.modes:
        sj = os.path.join(args.output_dir, mode, "ablation_summary.json")
        if os.path.exists(sj):
            with open(sj) as f:
                rows.append(json.load(f))
    if rows:
        out = os.path.join(args.output_dir, "ablation_comparison.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[ablation] comparison written → {out}")
        print("\n  mode        test_auc   test_auprc  val_auc    val_auprc   num_params")
        for r in rows:
            print(f"  {r['ablation']:10s}  "
                  f"{r.get('test_mean_auc',   float('nan')):>8.4f}   "
                  f"{r.get('test_mean_auprc', float('nan')):>10.4f}  "
                  f"{r.get('best_val_auc_mean',   float('nan')):>8.4f}   "
                  f"{r.get('best_val_auprc_mean', float('nan')):>10.4f}  "
                  f"{r['num_params']:>10,}")


if __name__ == "__main__":
    main()
