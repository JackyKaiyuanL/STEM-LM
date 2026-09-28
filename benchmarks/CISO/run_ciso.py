#!/usr/bin/env python3
import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_species_arg, load_dataset, resolve_output_dir, timed_phase, write_metrics  # noqa: E402

KNOWN_RATIOS = (0.0, 0.25, 0.5, 0.75)


def prepare_data(df, env_cols, species_cols, splits, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.DataFrame({"PlotObservationID": np.arange(len(df))})
    env = df[env_cols].fillna(df[env_cols].iloc[splits["train"]].mean())
    pd.concat([ids, env], axis=1).to_csv(data_dir / "worldclim_data.csv", index=False)
    ids.to_csv(data_dir / "soilgrid_data.csv", index=False)
    pd.DataFrame({"species": species_cols}).to_csv(data_dir / "species_list.csv", index=False)
    np.save(data_dir / "targets.npy", df[species_cols].to_numpy(dtype=np.float32))
    for name, idx in splits.items():
        np.save(data_dir / f"{name}_indices.npy", np.asarray(idx, dtype=np.int64))


def infer(repo, config_path):
    from src.config import Config
    from src.dataloaders.splot_dataloader import sPlotDataModule
    from src.trainers.splot_trainer import sPlotTrainer

    config = Config(**yaml.safe_load(config_path.read_text()))
    checkpoint = (Path(config.logger.checkpoint_path) / config.logger.experiment_name
                  / str(config.training.seed) / config.logger.checkpoint_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = sPlotDataModule(config.data)
    data_module.setup()
    task = sPlotTrainer(config).to(device)
    task.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    task.eval()
    logits, targets, masks = [], [], []
    with torch.no_grad():
        for batch in data_module.test_dataloader(num_workers=0, persistent_workers=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits.append(task(batch).float().cpu().numpy())
            targets.append(batch["targets"].cpu().numpy())
            masks.append(batch["mask"].cpu().numpy())
    return (np.concatenate(logits).astype(np.float64), np.concatenate(targets).astype(np.int64),
            np.concatenate(masks).astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate CISO on one dataset and split.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--splits_path", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--ciso_repo", required=True, type=Path)
    parser.add_argument("--num_epochs", type=int, default=100)
    add_species_arg(parser)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args, __file__)
    data_dir, config_dir = output_dir / "data", output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    df, env_cols, species_cols, splits = load_dataset(args.csv_path.resolve(), args.splits_path.resolve(),
                                                      args.min_train_presences)
    prepare_data(df, env_cols, species_cols, splits, data_dir)

    repo = args.ciso_repo.resolve()
    sys.path.insert(0, str(repo))
    config = {
        "mode": "train",
        "dataset_name": "sPlot",
        "model": {"name": "CISOModel", "input_dim": len(env_cols), "hidden_dim": 256,
                  "num_classes": len(species_cols), "backbone": "SimpleMLPBackbone"},
        "training": {"seed": args.seed, "learning_rate": 0.001, "max_epochs": args.num_epochs,
                     "accelerator": "gpu" if torch.cuda.is_available() else "cpu", "devices": 1},
        "logger": {"project_name": "stemlm_benchmark", "experiment_name": "ciso", "experiment_key": "",
                   "checkpoint_path": str(output_dir / "checkpoints"), "checkpoint_name": "",
                   "save_preds_path": ""},
        "data": {
            "dataloader_to_use": "sPlotMaskedDataset", "base": str(data_dir),
            "train": "train_indices.npy", "validation": "val_indices.npy", "test": "test_indices.npy",
            "targets": "targets.npy", "worldclim_data_path": "worldclim_data.csv",
            "soilgrid_data_path": "soilgrid_data.csv", "species_list": "species_list.csv",
            "species_occurrences_threshold": 0, "batch_size": 64, "env_columns": env_cols,
            "partial_labels": {"use": True, "quantized_mask_bins": 1, "train_known_ratio": 0.75,
                               "eval_known_ratio": 0, "predict_family_of_species": -1},
        },
    }
    train_config = config_dir / "train.yaml"
    train_config.write_text(yaml.safe_dump(config, sort_keys=False))
    environment = {**os.environ, "COMET_API_KEY": "offline-dummy", "COMET_MODE": "OFFLINE",
                   "COMET_OFFLINE_DIRECTORY": str(output_dir / "comet_logs"),
                   "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTHONPATH": str(repo)}

    def run(config_path, *extra):
        command = [sys.executable, str(repo / "main.py"), "--config", str(config_path), *map(str, extra)]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=repo, check=True, env=environment)

    with timed_phase(output_dir, "training"):
        run(train_config)
    checkpoints = sorted(p for p in (output_dir / "checkpoints" / "ciso" / str(args.seed)).glob("*.ckpt")
                         if "last" not in p.name)
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one best checkpoint, found {checkpoints}")
    print(f"best checkpoint: {checkpoints[0]}")

    results = []
    with timed_phase(output_dir, "evaluation"):
        for known in KNOWN_RATIOS:
            test_config = copy.deepcopy(config)
            test_config["mode"] = "test"
            test_config["logger"]["checkpoint_name"] = checkpoints[0].name
            test_config["data"]["partial_labels"]["eval_known_ratio"] = known
            path = config_dir / f"test_known_{known}.yaml"
            path.write_text(yaml.safe_dump(test_config, sort_keys=False))
            logits, targets, masks = infer(repo, path)
            labels = np.where(masks == -1, targets, -100)
            np.savez_compressed(output_dir / f"test_predictions_known_{known}.npz", logits=logits,
                                targets=targets, masks=masks, row_indices=splits["test"],
                                species=np.asarray(species_cols))
            results.append(({"cov_set": "env", "masking_p": 1.0 - known}, logits, labels))
    write_metrics(output_dir, "CISO", results, species_cols,
                  df[species_cols].to_numpy()[splits["train"]].sum(0))


if __name__ == "__main__":
    main()
