"""
Predict all species presence/absence at validation sites using a fully masked approach.
For models trained with jsdm_ablation.py.

Usage:
    python jsdm_predict_val.py \
        --model_dir ./ablation/50masked/full \
        --data_csv data/ebutterfly_na_static_jsdm.csv \
        --output predictions_val.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from jsdm_ablation import AblationConfig, AblationForMaskedSpeciesPrediction
from jsdm_data import JSDMDataset, JSDMDataCollator, compute_dist_info, h3_block_split


def ensure_config_and_species(model_dir, data_csv):
    """Generate config.json and species_names.json if missing."""

    df = pd.read_csv(data_csv, nrows=0)
    coord_cols = ["time", "latitude", "longitude"]
    env_cols = [c for c in df.columns if c.startswith("env_")]
    species_cols = [c for c in df.columns if c not in coord_cols and c not in env_cols]

    # species_names.json
    species_path = os.path.join(model_dir, "species_names.json")
    if not os.path.exists(species_path):
        with open(species_path, "w") as f:
            json.dump(species_cols, f)
        print(f"Created {species_path} ({len(species_cols)} species)")

    # config.json
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print("config.json not found — building from training args + data...")
        dataset = JSDMDataset(data_csv, num_source_sites=64, no_time=True)
        config_dict = {
            "num_species": len(species_cols),
            "num_source_sites": 64,
            "num_target_sites": 1,
            "max_spatial_dist": float(dataset.spatial_dists.max()) * 1.1,
            "max_temporal_dist": 0.0,
            "use_temporal": False,
            "num_env_vars": len(env_cols),
            "num_env_groups": 5,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 3,
            "intermediate_size": 1024,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-6,
            "fire_hidden_size": 32,
            "mlm_probability": 0.5,
            "ablation": "full",
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Created {config_path}")
        del dataset

    return species_cols


def main():
    parser = argparse.ArgumentParser(description="Fully masked prediction on val set")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions_val.csv")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_st", "no_eco", "no_st_eco"],
                        help="Ablation mode the model was trained with")
    parser.add_argument("--eval_split", type=str, default="val",
                        choices=["val", "test"],
                        help="Which split to predict on")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure config and species names exist
    species_cols = ensure_config_and_species(args.model_dir, args.data_csv)

    # Load config — override ablation mode if specified
    with open(os.path.join(args.model_dir, "config.json")) as f:
        config_dict = json.load(f)
    config_dict["ablation"] = args.ablation
    config = AblationConfig(**config_dict)

    # Load model
    print(f"Loading model (ablation={args.ablation})...")
    model = AblationForMaskedSpeciesPrediction(config)
    model.load_state_dict(
        torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device)
    )
    model.to(device).eval()

    with open(os.path.join(args.model_dir, "species_names.json")) as f:
        species_names = json.load(f)

    # Load dataset
    print("Loading dataset...")
    dataset = JSDMDataset(args.data_csv, num_source_sites=64, no_time=True)
    dist_info = compute_dist_info(
        dataset.spatial_dists, dataset.temporal_dists,
        dataset.spatial_scale_km, dataset.temporal_scale_days,
    )

    # Recreate the same H3 split
    train_idx, val_idx, test_idx = h3_block_split(
        dataset.lats, dataset.lons, resolution=2,
        train_frac=0.8, test_frac=0.1, seed=args.seed,
    )
    dataset.source_pool = train_idx  # sources only from training sites

    eval_idx = val_idx if args.eval_split == "val" else test_idx
    print(f"Evaluating on {args.eval_split} set: {len(eval_idx)} sites")

    # Fully masked collator
    collator = JSDMDataCollator(
        mlm_probability=1.0,
        combined_dist=dist_info["combined_dist"],
        blind_threshold=dist_info["blind_threshold"],
    )

    eval_subset = Subset(dataset, eval_idx)
    loader = DataLoader(eval_subset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collator)

    # Predict
    print("Running fully masked prediction...")
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = model(
                input_ids=batch["input_ids"],
                source_ids=batch["source_ids"],
                source_idx=batch["source_idx"],
                target_site_idx=batch["target_site_idx"],
                env_data=batch["env_data"],
                labels=batch["labels"],
                spatial_dist_pairwise=dist_info["spatial_dist_pairwise"],
                temporal_dist_pairwise=dist_info["temporal_dist_pairwise"],
            )
            probs = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()
            labels = batch["labels"].squeeze(-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{len(loader)}")

    probs = np.concatenate(all_probs, axis=0)   # (N_eval, S)
    labels = np.concatenate(all_labels, axis=0)  # (N_eval, S)

    # Per-species AUC
    S = len(species_names)
    per_species_aucs = {}
    aucs = []
    for s in range(S):
        mask = labels[:, s] != -100
        if mask.any() and len(set(labels[mask, s])) == 2:
            auc = roc_auc_score(labels[mask, s], probs[mask, s])
            aucs.append(auc)
            per_species_aucs[species_names[s]] = auc

    mean_auc = np.mean(aucs) if aucs else float("nan")
    total_correct = 0
    total_count = 0
    for s in range(S):
        mask = labels[:, s] != -100
        if mask.any():
            total_correct += ((probs[mask, s] > 0.5) == labels[mask, s]).sum()
            total_count += mask.sum()
    acc = total_correct / max(total_count, 1)

    print(f"\n{'='*64}")
    print(f"Fully masked prediction on {args.eval_split} set")
    print(f"{'='*64}")
    print(f"  Mean AUC:  {mean_auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Species with AUC: {len(aucs)}/{S}")

    # Save predictions
    results = pd.DataFrame(probs, columns=species_names)
    results.insert(0, "latitude", dataset.lats[eval_idx])
    results.insert(1, "longitude", dataset.lons[eval_idx])
    results.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")

    # Save per-species AUC
    auc_path = os.path.join(args.model_dir, f"per_species_auc_fully_masked_{args.eval_split}.csv")
    auc_df = pd.DataFrame(
        sorted(per_species_aucs.items(), key=lambda x: -x[1]),
        columns=["species", "auc"]
    )
    auc_df.to_csv(auc_path, index=False)
    print(f"Saved per-species AUC to {auc_path}")

    # Top/bottom species
    print(f"\nTop 10 species (highest AUC):")
    for _, row in auc_df.head(10).iterrows():
        print(f"  {row['species']:<50} {row['auc']:.4f}")
    print(f"\nBottom 10 species (lowest AUC):")
    for _, row in auc_df.tail(10).iterrows():
        print(f"  {row['species']:<50} {row['auc']:.4f}")


if __name__ == "__main__":
    main()
