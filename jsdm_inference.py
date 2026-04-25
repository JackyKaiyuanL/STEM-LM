"""
Inference for STEM-LM.

Commands:
  predict       - Predict all species at val/test sites (fully masked, train-only sources)
  interactions  - Extract species interaction matrices from attention weights
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset

from jsdm_model import JSDMConfig, JSDMForMaskedSpeciesPrediction, extract_interaction_matrix
from jsdm_data import (
    JSDMDataset,
    JSDMDataCollator,
    FixedPValCollator,
    compute_dist_info,
    grid_block_split,
    h3_block_split,
    load_splits,
)


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json")) as f:
        config_dict = json.load(f)
    import inspect
    cfg_params = set(inspect.signature(JSDMConfig).parameters)
    cfg_d = {k: v for k, v in config_dict.items() if k in cfg_params}
    config = JSDMConfig(**cfg_d)
    model  = JSDMForMaskedSpeciesPrediction(config)
    sd = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, config


def run_forward(model, batch, dist_info, device, output_attentions=False):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return model(
        input_ids=batch["input_ids"],
        source_ids=batch["source_ids"],
        source_idx=batch["source_idx"],
        target_site_idx=batch["target_site_idx"],
        env_data=batch["env_data"],
        target_env=batch["target_env"],
        labels=batch["labels"],
        site_lats=dist_info["site_lats"],
        site_lons=dist_info["site_lons"],
        site_times=dist_info["site_times"],
        euclidean=dist_info.get("euclidean", False),
        output_attentions=output_attentions,
    )


def build_dataset_and_dist(args, config):
    no_time = args.no_time or (not config.use_temporal)
    dataset = JSDMDataset(
        csv_path=args.csv_path,
        num_source_sites=config.num_source_sites,
        no_time=no_time,
        euclidean_coords=args.euclidean_coords,
    )
    dist_info = compute_dist_info(dataset, blind_percentile=args.blind_percentile)
    return dataset, dist_info


@torch.no_grad()
def predict(model, loader, dist_info, device, species_names):
    dist_info = dict(dist_info)
    for k in ("site_lats", "site_lons", "site_times"):
        dist_info[k] = dist_info[k].to(device)

    all_probs, all_labels = [], []
    n_batches = len(loader)
    for b, batch in enumerate(loader):
        output = run_forward(model, batch, dist_info, device)
        probs = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()   # (B, S)
        labels = batch["labels"].squeeze(-1).cpu().numpy()               # (B, S)
        all_probs.append(probs)
        all_labels.append(labels)
        if (b + 1) % 50 == 0 or (b + 1) == n_batches:
            print(f"    batch {b+1}/{n_batches}")

    probs  = np.concatenate(all_probs,  axis=0)  # (N_eval, S)
    labels = np.concatenate(all_labels, axis=0)  # (N_eval, S)
    return probs, labels


def compute_metrics(probs, labels, species_names):
    aucs, auprcs = {}, {}
    for s, name in enumerate(species_names):
        mask = labels[:, s] != -100
        if mask.sum() > 0 and len(np.unique(labels[mask, s])) == 2:
            aucs[name]   = roc_auc_score(labels[mask, s], probs[mask, s])
            auprcs[name] = average_precision_score(labels[mask, s], probs[mask, s])
    mean_auc   = float(np.mean(list(aucs.values())))   if aucs   else float("nan")
    mean_auprc = float(np.mean(list(auprcs.values()))) if auprcs else float("nan")
    return aucs, auprcs, mean_auc, mean_auprc


@torch.no_grad()
def extract_interactions(model, loader, dist_info, device, species_names, num_batches=50):
    dist_info = dict(dist_info)
    for k in ("site_lats", "site_lons", "site_times"):
        dist_info[k] = dist_info[k].to(device)
    per_layer = {i: [] for i in range(model.config.num_hidden_layers)}

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        output = run_forward(model, batch, dist_info, device, output_attentions=True)
        for layer_idx in range(model.config.num_hidden_layers):
            mat = extract_interaction_matrix(output, layer_idx=layer_idx)
            per_layer[layer_idx].append(mat.cpu())

    result = {}
    for layer_idx, matrices in per_layer.items():
        avg = torch.cat(matrices, dim=0).mean(dim=0).numpy()
        result[layer_idx] = pd.DataFrame(avg, index=species_names, columns=species_names)
    return result


def add_common_args(p):
    p.add_argument("model_dir",   type=str)
    p.add_argument("csv_path",    type=str)
    p.add_argument("output_path", type=str)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--num_workers",      type=int,   default=0)
    p.add_argument(
        "--blind_percentile",
        type=lambda s: "auto" if isinstance(s, str) and s.lower() == "auto" else float(s),
        default="auto",
    )
    p.add_argument("--no_time",          action="store_true")
    p.add_argument("--euclidean_coords", action="store_true")
    p.add_argument("--splits_path",      type=str,   default=None,
                   help="Path to splits.json from training; bypasses fold recomputation.")
    p.add_argument("--fold", choices=["random", "h3", "grid"], default="h3",
                   help="Train/val/test split strategy. Prefer passing --splits_path to "
                        "exactly match training; otherwise use the same --fold/--resolution "
                        "as training.")
    p.add_argument("--resolution", type=int, default=None,
                   help="Block resolution for spatial splits. For --fold h3, this is the H3 "
                        "resolution in [0, 15] (default 2). For --fold grid, this is the grid "
                        "side length (default 20). Not valid with --fold random.")
    p.add_argument("--train_frac",       type=float, default=0.8)
    p.add_argument("--test_frac",        type=float, default=0.1)
    p.add_argument("--seed",             type=int,   default=42)


def main():
    parser = argparse.ArgumentParser(description="STEM-LM Inference")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Per-p deterministic eval on val/test split")
    add_common_args(p_pred)
    p_pred.add_argument("--eval_split", choices=["val", "test"], default="test")
    p_pred.add_argument("--p", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                        help="Fixed mask rate(s). One or more floats in [0,1]. "
                             "FixedPValCollator gives per-batch-deterministic masks at each p. "
                             "Output parquet/CSV carry a suffix per p.")

    # ── interactions ──────────────────────────────────────────────────────────
    p_int = sub.add_parser("interactions", help="Extract species interaction matrices")
    add_common_args(p_int)
    p_int.add_argument("--num_batches", type=int, default=50)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.model_dir, device)

    dataset, dist_info = build_dataset_and_dist(args, config)

    species_names_path = os.path.join(args.model_dir, "species_names.json")
    if os.path.exists(species_names_path):
        with open(species_names_path) as f:
            species_names = json.load(f)
        if list(dataset.species_cols) != list(species_names):
            raise ValueError(
                "Species mismatch between training and inference data.\n"
                f"  Trained on {len(species_names)} species; CSV has {len(dataset.species_cols)}.\n"
                f"  First divergence: "
                f"{next((i for i, (a, b) in enumerate(zip(dataset.species_cols, species_names)) if a != b), 'order/length differs')}"
            )
    else:
        print(f"  No species_names.json in {args.model_dir} — falling back to CSV species_cols")
        species_names = list(dataset.species_cols)

    # Split — restrict sources to training observations only. Prefer a saved
    # splits.json (from training) so eval matches exactly; otherwise recompute
    # the split with the same flags the model was trained on.
    splits_path = getattr(args, "splits_path", None)
    if splits_path is not None:
        print(f"Loading splits from {splits_path}")
        train_idx, val_idx, test_idx = load_splits(
            splits_path, expected_num_rows=len(dataset)
        )
    else:
        if args.fold == "random":
            if args.resolution is not None:
                raise ValueError("--resolution is not valid with --fold random.")
            rng = np.random.RandomState(args.seed)
            indices = rng.permutation(len(dataset))
            n_train = int(len(dataset) * args.train_frac)
            n_test = int(len(dataset) * args.test_frac)
            train_idx = indices[:n_train]
            val_idx = indices[n_train : len(dataset) - n_test if n_test > 0 else len(dataset)]
            test_idx = indices[len(dataset) - n_test :] if n_test > 0 else np.array([], dtype=int)
        elif args.fold == "h3":
            if args.euclidean_coords:
                raise ValueError("--fold h3 requires real lat/lon coordinates. Use --fold grid for euclidean datasets.")
            resolution = 2 if args.resolution is None else args.resolution
            if not (0 <= int(resolution) <= 15):
                raise ValueError("--resolution for --fold h3 must be an integer in [0, 15].")
            train_idx, val_idx, test_idx = h3_block_split(
                dataset.lats, dataset.lons,
                resolution=int(resolution),
                train_frac=args.train_frac, test_frac=args.test_frac, seed=args.seed,
            )
        else:  # grid
            if not args.euclidean_coords:
                raise ValueError("--fold grid is for euclidean/simulated datasets. Use --fold h3 for real lat/lon.")
            resolution = 20 if args.resolution is None else args.resolution
            if int(resolution) < 1:
                raise ValueError("--resolution for --fold grid must be a positive integer.")
            train_idx, val_idx, test_idx = grid_block_split(
                dataset.lats, dataset.lons,
                n_cells=int(resolution),
                train_frac=args.train_frac, test_frac=args.test_frac, seed=args.seed,
            )
    dataset.source_pool = train_idx
    print(f"Source pool: {len(train_idx)} training observations")

    if args.command == "predict":
        eval_idx = val_idx if args.eval_split == "val" else test_idx
        print(f"Predicting on {args.eval_split} set: {len(eval_idx)} sites")

        per_p_species_auc   = {}
        per_p_species_auprc = {}
        per_p_mean_auc      = {}
        per_p_mean_auprc    = {}
        last_probs = None
        for i, p in enumerate(args.p):
            print(f"\n── p = {p:.2f} ──")
            collator = FixedPValCollator(
                p=p,
                site_lats=dataset.lats, site_lons=dataset.lons, site_times=dataset.times,
                spatial_scale_km=dataset.spatial_scale_km,
                temporal_scale_days=dataset.temporal_scale_days,
                euclidean=dataset.euclidean_coords,
                blind_threshold=dist_info["blind_threshold"],
                base_seed=args.seed + 10_000 + 1000 * i,
            )
            loader = DataLoader(
                Subset(dataset, eval_idx), batch_size=args.batch_size,
                shuffle=False, collate_fn=collator, num_workers=args.num_workers,
            )
            probs, labels = predict(model, loader, dist_info, device, species_names)
            aucs, auprcs, mean_auc, mean_auprc = compute_metrics(probs, labels, species_names)
            per_p_species_auc[p]   = aucs
            per_p_species_auprc[p] = auprcs
            per_p_mean_auc[p]      = mean_auc
            per_p_mean_auprc[p]    = mean_auprc
            print(f"  mean AUPRC = {mean_auprc:.4f}  mean AUC = {mean_auc:.4f}  "
                  f"({len(aucs)}/{len(species_names)} species)")
            last_probs = probs

        overall_auc   = float(np.mean(list(per_p_mean_auc.values())))
        overall_auprc = float(np.mean(list(per_p_mean_auprc.values())))
        print(f"\n============================================================")
        print(f"{args.eval_split} per-p (deterministic):")
        for p in args.p:
            print(f"  p={p:.2f}  AUPRC={per_p_mean_auprc[p]:.4f}  AUC={per_p_mean_auc[p]:.4f}")
        print(f"  mean over p   AUPRC={overall_auprc:.4f}  AUC={overall_auc:.4f}")
        print(f"============================================================")

        # Save predictions (parquet) using the LAST p's output — matches legacy p=1.0 use
        result_df = pd.DataFrame(last_probs, columns=species_names)
        result_df.insert(0, "latitude",  dataset.lats[eval_idx])
        result_df.insert(1, "longitude", dataset.lons[eval_idx])
        result_df.to_parquet(args.output_path, index=False)
        print(f"Predictions saved to {args.output_path}  (at p={args.p[-1]:.2f})")

        # Per-species AUC/AUPRC table with one column per p
        all_species = sorted({s for a in per_p_species_auc.values() for s in a})
        rows = []
        for s in all_species:
            row = {"species": s}
            for p in args.p:
                row[f"auc_p{p:.2f}"]   = per_p_species_auc[p].get(s, float("nan"))
                row[f"auprc_p{p:.2f}"] = per_p_species_auprc[p].get(s, float("nan"))
            row["auc_mean"]   = float(np.nanmean([row[f"auc_p{p:.2f}"]   for p in args.p]))
            row["auprc_mean"] = float(np.nanmean([row[f"auprc_p{p:.2f}"] for p in args.p]))
            rows.append(row)
        auc_df = pd.DataFrame(rows).sort_values("auprc_mean", ascending=False)
        auc_path = os.path.join(args.model_dir, f"per_species_auc_{args.eval_split}_by_p.csv")
        auc_df.to_csv(auc_path, index=False)
        print(f"Per-species AUC/AUPRC (per-p) saved to {auc_path}")

        # JSON summary
        summary = {
            "eval_split": args.eval_split,
            "n_eval_rows": int(len(eval_idx)),
            "p_list": [float(p) for p in args.p],
            "per_p_mean_auprc": {f"{p:.2f}": float(per_p_mean_auprc[p]) for p in args.p},
            "per_p_mean_auc":   {f"{p:.2f}": float(per_p_mean_auc[p])   for p in args.p},
            "overall_mean_auprc": overall_auprc,
            "overall_mean_auc":   overall_auc,
        }
        js_path = os.path.join(args.model_dir, f"test_per_p_summary.json")
        with open(js_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {js_path}")

    elif args.command == "interactions":
        collator = JSDMDataCollator(
            p=getattr(config, "p", 0.15),
            site_lats=dataset.lats, site_lons=dataset.lons, site_times=dataset.times,
            spatial_scale_km=dataset.spatial_scale_km,
            temporal_scale_days=dataset.temporal_scale_days,
            euclidean=dataset.euclidean_coords,
            blind_threshold=dist_info["blind_threshold"],
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=args.num_workers,
        )

        interaction_dfs = extract_interactions(
            model, loader, dist_info, device, species_names, args.num_batches
        )
        last = max(interaction_dfs.keys())
        interaction_dfs[last].to_parquet(args.output_path)
        npy_dir = os.path.splitext(args.output_path)[0] + "_all_layers"
        os.makedirs(npy_dir, exist_ok=True)
        for layer_idx, df in interaction_dfs.items():
            np.save(os.path.join(npy_dir, f"layer_{layer_idx}.npy"), df.values)
        print(f"Interaction matrices saved to {args.output_path}")


if __name__ == "__main__":
    main()
