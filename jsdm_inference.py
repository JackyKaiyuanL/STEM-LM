"""
Inference for ST-JSDM.

Commands:
  predict       - Predict masked species at sites
  interactions  - Extract species interaction matrices from attention weights
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from jsdm_model import JSDMConfig, JSDMForMaskedSpeciesPrediction, extract_interaction_matrix
from jsdm_data import JSDMDataset, JSDMDataCollator, compute_dist_info, h3_block_split, grid_block_split


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
        labels=batch["labels"],
        spatial_dist_pairwise=dist_info["spatial_dist_pairwise"],
        temporal_dist_pairwise=dist_info["temporal_dist_pairwise"],
        output_attentions=output_attentions,
    )


def build_dataset_and_loader(args, config):
    no_time = args.no_time or (not config.use_temporal)
    dataset = JSDMDataset(
        csv_path=args.csv_path,
        num_source_sites=config.num_source_sites,
        spatial_scale_km=args.spatial_scale_km,
        temporal_scale_days=args.temporal_scale_days,
        euclidean_coords=args.euclidean_coords,
        no_time=no_time,
    )
    dist_info = compute_dist_info(
        spatial_dist=dataset.spatial_dists,
        temporal_dist=dataset.temporal_dists,
        spatial_scale_km=dataset.spatial_scale_km,
        temporal_scale_days=dataset.temporal_scale_days,
        blind_percentile=args.blind_percentile,
    )
    collator = JSDMDataCollator(
        mlm_probability=config.mlm_probability,
        combined_dist=dist_info["combined_dist"],
        blind_threshold=dist_info["blind_threshold"],
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )
    return dataset, dist_info, loader


@torch.no_grad()
def predict_masked(model, loader, dist_info, device, species_names):
    dist_info = dict(dist_info)
    dist_info["spatial_dist_pairwise"] = dist_info["spatial_dist_pairwise"].to(device)
    dist_info["temporal_dist_pairwise"] = dist_info["temporal_dist_pairwise"].to(device)
    all_preds = []
    n_batches = len(loader)
    for b, batch in enumerate(loader):
        output = run_forward(model, batch, dist_info, device)
        probs = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()
        all_preds.append(probs)
        if (b + 1) % 50 == 0 or (b + 1) == n_batches:
            print(f"    batch {b+1}/{n_batches}")
    return pd.DataFrame(np.concatenate(all_preds, axis=0), columns=species_names)


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
    p.add_argument("model_dir", type=str)
    p.add_argument("csv_path", type=str,
                   help="CSV with species observations.")
    p.add_argument("output_path", type=str)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--blind_percentile", type=float, default=2.0)
    p.add_argument("--spatial_scale_km", type=float, default=None)
    p.add_argument("--temporal_scale_days", type=float, default=None)
    p.add_argument("--no_time", action="store_true",
                   help="Ignore time column. Auto-set if model was trained without time.")
    p.add_argument("--euclidean_coords", action="store_true",
                   help="Use Euclidean distance instead of haversine (for simulated data).")


def main():
    parser = argparse.ArgumentParser(description="ST-JSDM Inference")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── predict ───────────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Predict masked species at sites")
    add_common_args(p_pred)
    p_pred.add_argument("--fold", choices=["random", "h3", "grid"], default="random",
                        help="Split method used during training. Restricts source pool to "
                             "training observations so inference matches training setup.")
    p_pred.add_argument("--h3_resolution", type=int, default=2,
                        help="H3 resolution used during training (--fold h3 only).")
    p_pred.add_argument("--train_frac", type=float, default=0.8)
    p_pred.add_argument("--test_frac", type=float, default=0.1)
    p_pred.add_argument("--seed", type=int, default=42)
    p_pred.add_argument("--grid_cells", type=int, default=20,
                        help="Grid cell count used during training (--fold grid only).")

    # ── interactions ──────────────────────────────────────────────────────────
    p_int = sub.add_parser("interactions", help="Extract species interaction matrices")
    add_common_args(p_int)
    p_int.add_argument("--num_batches", type=int, default=50)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.model_dir, device)

    with open(os.path.join(args.model_dir, "species_names.json")) as f:
        species_names = json.load(f)

    dataset, dist_info, loader = build_dataset_and_loader(args, config)

    if args.command == "predict":
        if args.fold == "h3":
            train_idx, _, _ = h3_block_split(
                dataset.lats, dataset.lons,
                resolution=args.h3_resolution,
                train_frac=args.train_frac, test_frac=args.test_frac, seed=args.seed,
            )
            dataset.source_pool = train_idx
            print(f"Source pool: {len(train_idx)} training observations (h3 res={args.h3_resolution})")
        elif args.fold == "grid":
            train_idx, _, _ = grid_block_split(
                dataset.lats, dataset.lons,
                n_cells=args.grid_cells,
                train_frac=args.train_frac, test_frac=args.test_frac, seed=args.seed,
            )
            dataset.source_pool = train_idx
            print(f"Source pool: {len(train_idx)} training observations (grid {args.grid_cells}×{args.grid_cells})")
        else:
            print("Source pool: all observations")

        result = predict_masked(model, loader, dist_info, device, species_names)
        result.to_parquet(args.output_path, index=False)
        print(f"Predictions saved to {args.output_path}")

    elif args.command == "interactions":
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
