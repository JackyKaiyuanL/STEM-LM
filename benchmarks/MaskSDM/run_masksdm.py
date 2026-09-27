#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_dataset, resolve_output_dir, timed_phase, write_metrics  # noqa: E402


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MaskSDM on one dataset and split.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--splits_path", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--masksdm_repo", required=True, type=Path)
    parser.add_argument("--num_epochs", type=int, default=1000)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args, __file__)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    df, env_cols, species_cols, splits = load_dataset(args.csv_path.resolve(), args.splits_path.resolve())

    repo = args.masksdm_repo.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from data_helpers import get_torch_dataset
    from modules import get_model
    from training_helpers import seed_everything, train

    targets = df[species_cols].to_numpy(dtype=np.float32)
    tabular_x = df[env_cols].to_numpy(dtype=np.float32)
    satclip = np.zeros((len(df), 256), dtype=np.float32)
    data = {"tabular_x": tabular_x, "y": targets, "satclip_embeddings": satclip}
    for name, idx in splits.items():
        data[f"x_{name}"] = tabular_x[idx]
        data[f"y_{name}"] = targets[idx]
        data[f"satclip_embeddings_{name}"] = satclip[idx]
    mean = np.nanmean(data["x_train"], axis=0)
    std = np.nanstd(data["x_train"], axis=0)
    for name in splits:
        data[f"x_{name}"] = (data[f"x_{name}"] - mean) / (std + 1e-4)

    evaluated = np.intersect1d(np.intersect1d(data["y_train"].sum(0).nonzero()[0],
                                              data["y_val"].sum(0).nonzero()[0]),
                               data["y_test"].sum(0).nonzero()[0]).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    torch.set_default_device(device)
    config = {
        "device": device, "seed": args.seed, "dataset": "splot",
        "n_features": len(env_cols), "n_species": len(species_cols),
        "n_samples_train": len(splits["train"]), "n_samples_val": len(splits["val"]),
        "n_samples_test": len(splits["test"]),
        "indices_evaluated_species": evaluated, "n_evaluated_species": len(evaluated),
        "satclip": False, "model": "FTTransformer", "d_hidden": 192, "n_heads": 8, "n_blocks": 7,
        "dropout": 0.1, "d_out": len(species_cols), "epochs": args.num_epochs, "batch_size": 256,
        "batch_size_eval": 4096, "loss": "weighted",
        "species_weights": torch.tensor(len(splits["train"]) / (data["y_train"].sum(0) + 1e-5),
                                        dtype=torch.float32).to(device),
        "optimizer": "AdamW", "scheduler_free": True, "lr": 0.001, "weight_decay": 0.01,
        "warmup_steps": 1000, "masking": True, "extra_masking": True,
        "save_dir": str(checkpoint_dir), "use_wandb": False, "wandb_init": {},
    }
    (output_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, (torch.device, torch.Tensor)) else v for k, v in config.items()},
        indent=2) + "\n")

    log_path = output_dir / "training_console.log"
    stdout = sys.stdout
    with timed_phase(output_dir, "training"), log_path.open("w", buffering=1) as log:
        sys.stdout = Tee(stdout, log)
        try:
            train(config, data)
        finally:
            sys.stdout = stdout

    epochs = [(int(m.group(1)), float(m.group(2)))
              for m in re.finditer(r"Epoch (\d+), val AUC: ([\d.]+)", log_path.read_text())]
    best_epoch, best_val_auc = max(epochs, key=lambda e: e[1])
    print(f"best_epoch: {best_epoch}  val AUC {best_val_auc:.6f}")

    model = get_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / f"epoch_{best_epoch}.pt",
                                     map_location=device, weights_only=False))
    model.eval()
    loader = DataLoader(get_torch_dataset(config, data["x_test"], data["y_test"],
                                          data["satclip_embeddings_test"]),
                        batch_size=config["batch_size_eval"], shuffle=False)
    logits = []
    with torch.no_grad():
        for x, _, satclip_batch in loader:
            x, satclip_batch = x.to(device), satclip_batch.to(device)
            satclip_mask = torch.zeros(len(satclip_batch), dtype=torch.bool, device=device)
            logits.append(model(x, satclip_batch, ~torch.isnan(x), satclip_mask).float().cpu().numpy())
    logits = np.concatenate(logits).astype(np.float64)
    labels = data["y_test"].astype(np.int64)
    np.savez_compressed(output_dir / "test_predictions.npz", logits=logits, labels=labels,
                        row_indices=splits["test"], species=np.asarray(species_cols))
    write_metrics(output_dir, "MaskSDM", [({"cov_set": "env", "masking_p": 1.0}, logits, labels)],
                  species_cols, data["y_train"].sum(0))


if __name__ == "__main__":
    main()
