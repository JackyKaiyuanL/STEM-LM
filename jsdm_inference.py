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
from jsdm_data import JSDMDataset, JSDMDataCollator, build_st_clusters


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


def run_forward(model, batch, cluster_info, device, output_attentions=False):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return model(
        input_ids=batch["input_ids"],
        source_ids=batch["source_ids"],
        source_idx=batch["source_idx"],
        target_site_idx=batch["target_site_idx"],
        env_data=batch["env_data"],
        target_env=batch["target_env"],
        labels=batch["labels"],
        cluster_dict=cluster_info["cluster_dict"],
        cluster_labels=cluster_info["cluster_labels"],
        in_cluster_spatial_dist=cluster_info["in_cluster_spatial_dist"],
        in_cluster_temporal_dist=cluster_info["in_cluster_temporal_dist"],
        spatial_dist_pairwise=cluster_info["spatial_dist_pairwise"],
        temporal_dist_pairwise=cluster_info["temporal_dist_pairwise"],
        output_attentions=output_attentions,
    )


@torch.no_grad()
def predict_masked(model, loader, cluster_info, device, species_names):
    all_preds = []
    for batch in loader:
        output = run_forward(model, batch, cluster_info, device)
        probs = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()
        all_preds.append(probs)
    return pd.DataFrame(np.concatenate(all_preds, axis=0), columns=species_names)


@torch.no_grad()
def extract_interactions(model, loader, cluster_info, device, species_names, num_batches=50):
    per_layer = {i: [] for i in range(model.config.num_hidden_layers)}

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        output = run_forward(model, batch, cluster_info, device, output_attentions=True)
        for layer_idx in range(model.config.num_hidden_layers):
            mat = extract_interaction_matrix(output, layer_idx=layer_idx)
            per_layer[layer_idx].append(mat.cpu())

    result = {}
    for layer_idx, matrices in per_layer.items():
        avg = torch.cat(matrices, dim=0).mean(dim=0).numpy()
        result[layer_idx] = pd.DataFrame(avg, index=species_names, columns=species_names)
    return result


def main():
    parser = argparse.ArgumentParser(description="ST-JSDM Inference")
    parser.add_argument("command", choices=["predict", "interactions"])
    parser.add_argument("model_dir", type=str)
    parser.add_argument("csv_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--cluster_threshold", type=float, default=5.0)
    parser.add_argument("--spatial_scale_km", type=float, default=None)
    parser.add_argument("--temporal_scale_days", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.model_dir, device)

    with open(os.path.join(args.model_dir, "species_names.json")) as f:
        species_names = json.load(f)

    dataset = JSDMDataset(
        csv_path=args.csv_path,
        num_source_sites=config.num_source_sites,
        spatial_scale_km=args.spatial_scale_km,
        temporal_scale_days=args.temporal_scale_days,
    )
    cluster_info = build_st_clusters(
        dataset.coords,
        dataset.times,
        threshold=args.cluster_threshold,
        spatial_scale_km=dataset.spatial_scale_km,
        temporal_scale_days=dataset.temporal_scale_days,
        spatial_dist=dataset.spatial_dists,
        temporal_dist=dataset.temporal_dists,
    )
    collator = JSDMDataCollator(
        mlm_probability=config.mlm_probability,
        mask_value=config.mask_value_init,
        cluster_labels=cluster_info["cluster_labels"],
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )

    if args.command == "predict":
        result = predict_masked(model, loader, cluster_info, device, species_names)
        result.to_parquet(args.output_path, index=False)
        print(f"Predictions saved to {args.output_path}")

    elif args.command == "interactions":
        interaction_dfs = extract_interactions(
            model, loader, cluster_info, device, species_names, args.num_batches
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
