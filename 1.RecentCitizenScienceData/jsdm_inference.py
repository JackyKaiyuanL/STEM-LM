"""
Inference for ST-JSDM (citizen-science variant).

Commands:
  predict       - Predict species presence probabilities at sites
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
from jsdm_data import JSDMDataset, JSDMDataCollator


def load_model(model_dir, device):
    with open(os.path.join(model_dir, "config.json")) as f:
        config_dict = json.load(f)
    config = JSDMConfig(**config_dict)
    model  = JSDMForMaskedSpeciesPrediction(config)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    )
    return model.to(device).eval()


def _build_forecast_batch(example, num_species):
    source_ids = example["source_species"].unsqueeze(0)
    source_idx = example["source_idx"].unsqueeze(0)
    target_site_idx = example["target_idx"].view(1, 1)
    env_data = example["source_env"].unsqueeze(0)
    target_env = example["target_env"].unsqueeze(0)
    source_spatial_dist = example["source_spatial_dist"].unsqueeze(0)
    source_temporal_dist = example["source_temporal_dist"].unsqueeze(0)
    source_doy_dist = example["source_doy_dist"].unsqueeze(0)
    source_time = example["source_time"].unsqueeze(0)
    target_time = example["target_time"].view(1)

    input_ids = torch.full((1, num_species, 1), 2, dtype=torch.long)
    labels = torch.full((1, num_species, 1), -100.0, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "source_ids": source_ids,
        "source_idx": source_idx,
        "target_site_idx": target_site_idx,
        "env_data": env_data,
        "target_env": target_env,
        "source_spatial_dist": source_spatial_dist,
        "source_temporal_dist": source_temporal_dist,
        "source_doy_dist": source_doy_dist,
        "source_time": source_time,
        "target_time": target_time,
        "labels": labels,
    }


def run_forward(model, batch, device, output_attentions=False):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return model(
        input_ids            = batch["input_ids"],
        source_ids           = batch["source_ids"],
        source_idx           = batch["source_idx"],
        target_site_idx      = batch["target_site_idx"],
        env_data             = batch["env_data"],
        target_env           = batch["target_env"],
        source_spatial_dist  = batch["source_spatial_dist"],
        source_temporal_dist = batch["source_temporal_dist"],
        source_doy_dist      = batch["source_doy_dist"],
        source_time          = batch["source_time"],
        target_time          = batch["target_time"],
        labels               = batch["labels"],
        output_attentions    = output_attentions,
    )


@torch.no_grad()
def predict_masked(model, loader, device, species_names):
    all_preds = []
    for batch in loader:
        output = run_forward(model, batch, device)
        probs  = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()
        all_preds.append(probs)
    return pd.DataFrame(np.concatenate(all_preds, axis=0), columns=species_names)


@torch.no_grad()
def extract_interactions(model, loader, device, species_names, num_batches=50):
    per_layer = {i: [] for i in range(model.config.num_hidden_layers)}
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        output = run_forward(model, batch, device, output_attentions=True)
        for layer_idx in range(model.config.num_hidden_layers):
            per_layer[layer_idx].append(extract_interaction_matrix(output, layer_idx).cpu())

    return {
        layer_idx: pd.DataFrame(
            torch.cat(mats, dim=0).mean(dim=0).numpy(),
            index=species_names, columns=species_names,
        )
        for layer_idx, mats in per_layer.items()
    }


@torch.no_grad()
def forecast_rollout(model, dataset, future_indices, device, species_names):
    preds = []
    for idx in future_indices:
        example = dataset[idx]
        batch = _build_forecast_batch(example, len(species_names))
        output = run_forward(model, batch, device)
        probs = torch.sigmoid(output.logits).squeeze(0).squeeze(-1)
        preds.append(probs.cpu().numpy())
        dataset.species_data[idx] = probs.cpu().numpy().astype(np.float32)

    return np.stack(preds, axis=0)


def main():
    parser = argparse.ArgumentParser(description="ST-JSDM Inference")
    parser.add_argument("command",     choices=["predict", "interactions", "forecast"])
    parser.add_argument("model_dir",   type=str)
    parser.add_argument("csv_path",    type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--future_csv", type=str, default=None)
    parser.add_argument("--batch_size",          type=int,   default=32)
    parser.add_argument("--num_workers",         type=int,   default=0)
    parser.add_argument("--num_batches",         type=int,   default=50)
    parser.add_argument("--spatial_scale_km",    type=float, default=None)
    parser.add_argument("--temporal_scale_days", type=float, default=None)
    parser.add_argument("--time_window_days",    type=float, default=None)
    parser.add_argument("--sampling_strategy",   type=str,   default="nearest",
                        choices=["nearest", "weighted"])
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = load_model(args.model_dir, device)
    config     = model.config

    with open(os.path.join(args.model_dir, "species_names.json")) as f:
        species_names = json.load(f)

    spatial_scale_km = args.spatial_scale_km if args.spatial_scale_km is not None else config.spatial_scale_km
    temporal_scale_days = args.temporal_scale_days if args.temporal_scale_days is not None else config.temporal_scale_days

    if args.command == "predict":
        dataset = JSDMDataset(
            csv_path=args.csv_path,
            num_source_sites=config.num_source_sites,
            spatial_scale_km=spatial_scale_km,
            temporal_scale_days=temporal_scale_days,
            time_window_days=args.time_window_days,
            sampling_strategy=args.sampling_strategy,
        )
        collator = JSDMDataCollator(
            mlm_probability=config.mlm_probability,
            mask_value=config.mask_value_init,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=args.num_workers,
        )
        result = predict_masked(model, loader, device, species_names)
        result.to_parquet(args.output_path, index=False)
        print(f"Predictions saved to {args.output_path}")

    elif args.command == "interactions":
        dataset = JSDMDataset(
            csv_path=args.csv_path,
            num_source_sites=config.num_source_sites,
            spatial_scale_km=spatial_scale_km,
            temporal_scale_days=temporal_scale_days,
            time_window_days=args.time_window_days,
            sampling_strategy=args.sampling_strategy,
        )
        collator = JSDMDataCollator(
            mlm_probability=config.mlm_probability,
            mask_value=config.mask_value_init,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=args.num_workers,
        )
        interaction_dfs = extract_interactions(
            model, loader, device, species_names, args.num_batches
        )
        last = max(interaction_dfs.keys())
        interaction_dfs[last].to_parquet(args.output_path)
        npy_dir = os.path.splitext(args.output_path)[0] + "_all_layers"
        os.makedirs(npy_dir, exist_ok=True)
        for layer_idx, df in interaction_dfs.items():
            np.save(os.path.join(npy_dir, f"layer_{layer_idx}.npy"), df.values)
        print(f"Interaction matrices saved to {args.output_path}")

    elif args.command == "forecast":
        if args.future_csv is None:
            raise ValueError("--future_csv is required for forecast")

        obs_df = pd.read_csv(args.csv_path)
        fut_df = pd.read_csv(args.future_csv)
        coord_cols = ["time", "latitude", "longitude"]
        if "doy" in obs_df.columns and "doy" in fut_df.columns:
            coord_cols.append("doy")
        env_cols = [c for c in obs_df.columns if c not in coord_cols and c not in species_names]
        missing_env = [c for c in env_cols if c not in fut_df.columns]
        if missing_env:
            raise ValueError("Future CSV missing env columns: " + ", ".join(missing_env))

        missing_species = [c for c in species_names if c not in obs_df.columns]
        if missing_species:
            raise ValueError("Observed CSV missing species columns: " + ", ".join(missing_species))

        for c in species_names:
            if c not in fut_df.columns:
                fut_df[c] = 0.0

        obs_df["_is_future"] = False
        fut_df["_is_future"] = True
        combined_df = pd.concat([obs_df, fut_df], ignore_index=True)
        future_indices = np.flatnonzero(combined_df["_is_future"].values)
        future_indices = future_indices[np.argsort(combined_df.loc[future_indices, "time"].values)]

        dataset = JSDMDataset(
            data=combined_df.drop(columns=["_is_future"]),
            num_source_sites=config.num_source_sites,
            env_cols=env_cols,
            spatial_scale_km=spatial_scale_km,
            temporal_scale_days=temporal_scale_days,
            time_window_days=args.time_window_days,
            sampling_strategy=args.sampling_strategy,
        )
        preds = forecast_rollout(model, dataset, future_indices, device, species_names)

        meta = combined_df.loc[future_indices, coord_cols].reset_index(drop=True)
        out_df = pd.concat([meta, pd.DataFrame(preds, columns=species_names)], axis=1)
        out_df.to_parquet(args.output_path, index=False)
        print(f"Forecast saved to {args.output_path}")


if __name__ == "__main__":
    main()
