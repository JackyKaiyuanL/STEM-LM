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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from jsdm_model import JSDMConfig, JSDMForMaskedSpeciesPrediction, extract_interaction_matrix
from jsdm_data import JSDMDataset, JSDMDataCollator, compute_dist_info, h3_block_split, load_splits


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json")) as f:
        config_dict = json.load(f)
    config = JSDMConfig(**config_dict)
    model = JSDMForMaskedSpeciesPrediction(config)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    )
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
        spatial_dist_pairwise=dist_info["spatial_dist_pairwise"],
        temporal_dist_pairwise=dist_info["temporal_dist_pairwise"],
        output_attentions=output_attentions,
    )


def build_dataset_and_dist(args, config):
    no_time = args.no_time or (not config.use_temporal)
    dataset = JSDMDataset(
        csv_path=args.csv_path,
        num_source_sites=config.num_source_sites,
        no_time=no_time,
    )
    dist_info = compute_dist_info(
        spatial_dist=dataset.spatial_dists,
        temporal_dist=dataset.temporal_dists,
        spatial_scale_km=dataset.spatial_scale_km,
        temporal_scale_days=dataset.temporal_scale_days,
        blind_percentile=args.blind_percentile,
    )
    return dataset, dist_info


@torch.no_grad()
def predict(model, loader, dist_info, device, species_names):
    dist_info = dict(dist_info)
    dist_info["spatial_dist_pairwise"] = dist_info["spatial_dist_pairwise"].to(device)
    dist_info["temporal_dist_pairwise"] = dist_info["temporal_dist_pairwise"].to(device)

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


def compute_auc(probs, labels, species_names):
    aucs = {}
    for s, name in enumerate(species_names):
        mask = labels[:, s] != -100
        if mask.sum() > 0 and len(np.unique(labels[mask, s])) == 2:
            aucs[name] = roc_auc_score(labels[mask, s], probs[mask, s])
    mean_auc = np.mean(list(aucs.values())) if aucs else float("nan")
    return aucs, mean_auc


@torch.no_grad()
def extract_interactions(model, loader, dist_info, device, species_names, num_batches=50):
    dist_info = dict(dist_info)
    dist_info["spatial_dist_pairwise"] = dist_info["spatial_dist_pairwise"].to(device)
    dist_info["temporal_dist_pairwise"] = dist_info["temporal_dist_pairwise"].to(device)
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
    p.add_argument("--blind_percentile", type=float, default=2.0)
    p.add_argument("--no_time",          action="store_true")
    p.add_argument("--euclidean_coords", action="store_true")
    p.add_argument("--splits_path",      type=str,   default=None,
                   help="Path to splits.json from training; bypasses fold recomputation.")
    p.add_argument("--h3_resolution",    type=int,   default=2)
    p.add_argument("--train_frac",       type=float, default=0.8)
    p.add_argument("--test_frac",        type=float, default=0.1)
    p.add_argument("--seed",             type=int,   default=42)


def main():
    parser = argparse.ArgumentParser(description="STEM-LM Inference")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Fully masked prediction on eval split")
    add_common_args(p_pred)
    p_pred.add_argument("--eval_split", choices=["val", "test"], default="test")

    # ── interactions ──────────────────────────────────────────────────────────
    p_int = sub.add_parser("interactions", help="Extract species interaction matrices")
    add_common_args(p_int)
    p_int.add_argument("--num_batches", type=int, default=50)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.model_dir, device)

    with open(os.path.join(args.model_dir, "species_names.json")) as f:
        species_names = json.load(f)

    dataset, dist_info = build_dataset_and_dist(args, config)

    # Consistency check: the CSV's species columns must match the trained model's
    # species (same names, same order). Silent misalignment would give valid-looking
    # but wrong per-species AUCs.
    if list(dataset.species_cols) != list(species_names):
        raise ValueError(
            "Species mismatch between training and inference data.\n"
            f"  Trained on {len(species_names)} species; CSV has {len(dataset.species_cols)}.\n"
            f"  First divergence: "
            f"{next((i for i, (a, b) in enumerate(zip(dataset.species_cols, species_names)) if a != b), 'order/length differs')}"
        )

    # Split — restrict sources to training observations only. Prefer a saved
    # splits.json (from training) so eval matches exactly; otherwise recompute
    # H3 split with the same flags the model was trained on.
    splits_path = getattr(args, "splits_path", None)
    if splits_path is not None:
        print(f"Loading splits from {splits_path}")
        train_idx, val_idx, test_idx = load_splits(
            splits_path, expected_num_rows=len(dataset)
        )
    else:
        train_idx, val_idx, test_idx = h3_block_split(
            dataset.lats, dataset.lons,
            resolution=args.h3_resolution,
            train_frac=args.train_frac, test_frac=args.test_frac, seed=args.seed,
        )
    dataset.source_pool = train_idx
    print(f"Source pool: {len(train_idx)} training observations")

    if args.command == "predict":
        eval_idx = val_idx if args.eval_split == "val" else test_idx
        print(f"Predicting on {args.eval_split} set: {len(eval_idx)} sites")

        collator = JSDMDataCollator(
            mlm_probability=1.0,
            combined_dist=dist_info["combined_dist"],
            blind_threshold=dist_info["blind_threshold"],
        )
        loader = DataLoader(
            Subset(dataset, eval_idx), batch_size=args.batch_size,
            shuffle=False, collate_fn=collator, num_workers=args.num_workers,
        )

        probs, labels = predict(model, loader, dist_info, device, species_names)
        aucs, mean_auc = compute_auc(probs, labels, species_names)

        print(f"\nMean AUC ({args.eval_split}): {mean_auc:.4f}  ({len(aucs)}/{len(species_names)} species)")

        # Save predictions with coordinates
        result_df = pd.DataFrame(probs, columns=species_names)
        result_df.insert(0, "latitude",  dataset.lats[eval_idx])
        result_df.insert(1, "longitude", dataset.lons[eval_idx])
        result_df.to_parquet(args.output_path, index=False)
        print(f"Predictions saved to {args.output_path}")

        # Save per-species AUC
        auc_path = os.path.join(args.model_dir, f"per_species_auc_{args.eval_split}.csv")
        auc_df = pd.DataFrame(
            sorted(aucs.items(), key=lambda x: -x[1]), columns=["species", "auc"]
        )
        auc_df.to_csv(auc_path, index=False)
        print(f"Per-species AUC saved to {auc_path}")

        print(f"\nTop 10:")
        for _, row in auc_df.head(10).iterrows():
            print(f"  {row['species']:<50} {row['auc']:.4f}")
        print(f"Bottom 10:")
        for _, row in auc_df.tail(10).iterrows():
            print(f"  {row['species']:<50} {row['auc']:.4f}")

    elif args.command == "interactions":
        collator = JSDMDataCollator(
            mlm_probability=config.mlm_probability,
            combined_dist=dist_info["combined_dist"],
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
