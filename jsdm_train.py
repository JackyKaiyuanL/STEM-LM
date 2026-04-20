"""
Training script for STEM-LM.
"""

import argparse
import csv
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from jsdm_model import JSDMConfig, JSDMForMaskedSpeciesPrediction, extract_interaction_matrix
from jsdm_data import create_dataloaders, save_splits, load_splits

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_rate(s):
    """argparse type: float in [0, 1] or 'rand[:lo,hi]' string (validated downstream)."""
    if isinstance(s, str) and (s == "rand" or s.startswith("rand:")):
        return s
    return float(s)


# =============================================================================
# Shared training utilities (imported by jsdm_ablation.py to avoid drift)
# =============================================================================

def move_dist_info_to_device(dist_info, device):
    """Return a shallow copy of dist_info with pairwise tensors on `device`.

    Call once before the training loop so forward passes don't retransfer
    the full (N, N) pairwise matrices every batch.
    """
    out = dict(dist_info)
    out["spatial_dist_pairwise"]  = out["spatial_dist_pairwise"].to(device)
    out["temporal_dist_pairwise"] = out["temporal_dist_pairwise"].to(device)
    return out


def compute_class_weights(species_data, train_indices, beta=0.999):
    """Effective-number-of-samples class weighting (Cui et al., 2019)."""
    if not (0.0 < beta < 1.0):
        raise ValueError(f"class_weighting_beta must be in (0, 1), got {beta}")
    n_s = species_data[train_indices].sum(axis=0).clip(min=1)
    effective_n = (1 - beta ** n_s) / (1 - beta)
    w = 1.0 / effective_n
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _forward(model, batch, dist_info, loss_weight=None):
    return model(
        input_ids=batch["input_ids"],
        source_ids=batch["source_ids"],
        source_idx=batch["source_idx"],
        target_site_idx=batch["target_site_idx"],
        env_data=batch["env_data"],
        target_env=batch["target_env"],
        labels=batch["labels"],
        loss_weight=loss_weight,
        spatial_dist_pairwise=dist_info["spatial_dist_pairwise"],
        temporal_dist_pairwise=dist_info["temporal_dist_pairwise"],
    )


def train_epoch(model, loader, optimizer, scheduler, device, dist_info, epoch,
                loss_weight=None, log_interval=50, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total_masked, num_batches = 0, 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        B = batch["input_ids"].shape[0]
        w = loss_weight[None, :].expand(B, -1).to(device) if loss_weight is not None else None

        output = _forward(model, batch, dist_info, loss_weight=w)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        mask = batch["labels"] != -100
        if mask.any():
            preds = (output.logits[mask] > 0).long()
            targets = batch["labels"][mask].long()
            total_correct += (preds == targets).sum().item()
            total_masked += mask.sum().item()

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {total_loss/num_batches:.4f} | "
                f"Acc: {total_correct/max(total_masked,1):.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / max(num_batches, 1), total_correct / max(total_masked, 1)


@torch.no_grad()
def evaluate(model, loader, device, dist_info):
    model.eval()
    total_loss, num_batches = 0, 0
    S = model.config.num_species
    species_preds  = [[] for _ in range(S)]
    species_labels = [[] for _ in range(S)]

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = _forward(model, batch, dist_info)

        total_loss += output.loss.item()
        num_batches += 1

        probs  = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()   # (B, S)
        labels = batch["labels"].squeeze(-1).cpu().numpy()                # (B, S)
        mask   = labels != -100
        for s in range(S):
            b_mask = mask[:, s]
            if b_mask.any():
                species_preds[s].extend(probs[b_mask, s].tolist())
                species_labels[s].extend(labels[b_mask, s].tolist())

    per_species_aucs = {}
    total_correct, total_masked = 0, 0
    aucs = []
    for s in range(S):
        if len(species_labels[s]) > 0:
            arr_p = np.array(species_preds[s])
            arr_l = np.array(species_labels[s])
            total_correct += ((arr_p > 0.5) == arr_l).sum()
            total_masked  += len(arr_l)
            if len(set(species_labels[s])) == 2 and not np.isnan(arr_p).any():
                auc_s = roc_auc_score(arr_l, arr_p)
                aucs.append(auc_s)
                per_species_aucs[s] = auc_s
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    acc = total_correct / max(total_masked, 1)

    return total_loss / max(num_batches, 1), acc, mean_auc, per_species_aucs


def main():
    parser = argparse.ArgumentParser(description="Train STEM-LM")
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--num_source_sites", type=int, default=64)
    parser.add_argument("--blind_percentile", type=float, default=2.0,
                        help="Percentile of pairwise distances used as proximity blind threshold")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--num_env_groups", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--p", type=_parse_rate, default=None,
                        help="Per-row mask rate applied to both presences and absences. "
                             "Float in [0,1] or 'rand[:lo,hi]' (Uniform[lo, hi] per row; "
                             "'rand' = 'rand:0.0,1.0'). Overrides --p_pres/--p_abs if set.")
    parser.add_argument("--p_pres", type=_parse_rate, default=0.15,
                        help=argparse.SUPPRESS)
    parser.add_argument("--p_abs",  type=_parse_rate, default=0.15,
                        help=argparse.SUPPRESS)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of data held out as test set for final AUC. "
                             "Val = 1 - train_frac - test_frac. Set to 0 to disable "
                             "(final AUC reported on val, same as early-stopping set).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./jsdm_output")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--spatial_scale_km", type=float, default=None)
    parser.add_argument("--temporal_scale_days", type=float, default=None)
    parser.add_argument("--no_time", action="store_true",
                        help="Disable temporal FIRE bias. Set automatically when all "
                             "time values in the CSV are identical (static datasets).")
    parser.add_argument("--euclidean_coords", action="store_true",
                        help="Use Euclidean distance instead of haversine. "
                             "For simulated or arbitrary 2D coordinates (not geographic degrees).")
    parser.add_argument("--class_weighting", action="store_true",
                        help="Up-weight rare species using effective number weighting")
    parser.add_argument("--class_weighting_beta", type=float, default=0.999,
                        help="Beta for effective number class weighting (default 0.999). "
                             "Lower values (e.g. 0.99) reduce the weight on rare species.")
    parser.add_argument("--env_cols", nargs="+", default=None,
                        help="Explicit list of env column names. If not set, columns with 'env_' "
                             "prefix are used. Useful for datasets with non-prefixed env columns "
                             "(e.g. annualtemp, annualprec).")
    parser.add_argument("--fold", choices=["random", "h3", "grid"], default="random",
                        help="Train/val/test split strategy. 'random': shuffled rows (default). "
                             "'h3': spatial blocks via H3 hexagonal grid (real lat/lon only). "
                             "'grid': spatial blocks via regular 2D grid (euclidean_coords only).")
    parser.add_argument("--h3_resolution", type=int, default=2,
                        help="H3 resolution for --fold h3 (default 2, ~183km edge). "
                             "Larger number = finer cells (res 1=483km, 2=183km, 3=69km, 4=26km).")
    parser.add_argument("--grid_cells", type=int, default=20,
                        help="Grid side length for --fold grid (default 20 → 20×20 cells).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping max norm (default 1.0).")
    parser.add_argument("--interaction_extract_batches", type=int, default=20,
                        help="Number of val batches used to estimate the species "
                             "interaction matrix at the end of training (default 20).")
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Path to a splits.json (written by a previous training run). "
                             "When set, overrides --fold / --h3_resolution / --grid_cells / "
                             "--seed for the split, so train/val/test are exactly reproduced.")
    parser.add_argument("--no_save_splits", action="store_true",
                        help="Skip writing splits.json to output_dir.")
    args = parser.parse_args()

    # --p is a convenience that sets both p_pres and p_abs to the same rate.
    if args.p is not None:
        args.p_pres = args.p
        args.p_abs = args.p

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional: preload saved splits (CSV row count will be sanity-checked later)
    saved_splits = None
    if args.splits_path is not None:
        logger.info(f"Loading splits from {args.splits_path}")
        # Row-count check deferred to after dataset load via assert.
        saved_splits = load_splits(args.splits_path)

    # Data
    train_loader, val_loader, test_loader, dataset, dist_info, splits = create_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_source_sites=args.num_source_sites,
        p_pres=args.p_pres,
        p_abs=args.p_abs,
        blind_percentile=args.blind_percentile,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        num_workers=args.num_workers,
        seed=args.seed,
        env_cols=args.env_cols,
        spatial_scale_km=args.spatial_scale_km,
        temporal_scale_days=args.temporal_scale_days,
        euclidean_coords=args.euclidean_coords,
        no_time=args.no_time,
        fold_method=args.fold,
        h3_resolution=args.h3_resolution,
        grid_cells=args.grid_cells,
        saved_splits=saved_splits,
    )

    # Persist splits for downstream scripts (inference, ablation comparisons)
    if not args.no_save_splits:
        splits_out = os.path.join(args.output_dir, "splits.json")
        save_splits(
            splits_out, splits["train"], splits["val"], splits["test"],
            num_rows=len(dataset),
            meta={
                "fold":          args.fold if saved_splits is None else "loaded",
                "h3_resolution": args.h3_resolution,
                "grid_cells":    args.grid_cells,
                "train_frac":    args.train_frac,
                "test_frac":     args.test_frac,
                "seed":          args.seed,
                "source":        args.splits_path if saved_splits is not None else None,
            },
        )
        logger.info(f"Splits saved to {splits_out}")

    # Per-species loss weighting (effective number)
    loss_weight = None
    if args.class_weighting:
        loss_weight = compute_class_weights(
            dataset.species_data, train_loader.dataset.indices, beta=args.class_weighting_beta
        )
        logger.info(
            f"Class weighting: weight range "
            f"[{loss_weight.min().item():.3f}, {loss_weight.max().item():.3f}]"
        )

    # use_temporal mirrors what the dataset actually did with time
    use_temporal = dist_info["max_temporal_dist"] > 0
    if not use_temporal:
        logger.info("Temporal FIRE bias disabled (no temporal variation in data)")

    # Config
    config = JSDMConfig(
        num_species=dataset.num_species,
        num_source_sites=args.num_source_sites,
        max_spatial_dist=dist_info["max_spatial_dist"] * 1.1,
        max_temporal_dist=dist_info["max_temporal_dist"] * 1.1,
        use_temporal=use_temporal,
        num_env_vars=dataset.num_env_vars,
        num_env_groups=args.num_env_groups,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        p_pres=args.p_pres,
        p_abs=args.p_abs,
    )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    # Model
    model = JSDMForMaskedSpeciesPrediction(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params:,} parameters, {config.num_species} species")

    if args.gradient_checkpointing:
        model.model.encoder.gradient_checkpointing = True

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Pre-move pairwise distance matrices to device ONCE — otherwise the (N, N)
    # tensors would be retransferred on every forward pass.
    dist_info = move_dist_info_to_device(dist_info, device)

    log_csv = os.path.join(args.output_dir, "training_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc", "lr", "elapsed_s"])

    best_val_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, dist_info, epoch,
            loss_weight=loss_weight, max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_acc, val_auc, _ = evaluate(model, val_loader, device, dist_info)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | {elapsed:.1f}s"
        )
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                     f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_auc:.6f}",
                                     f"{current_lr:.2e}", f"{elapsed:.1f}"])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"  → Best model saved")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))

    # Per-species AUC on best model — use held-out test set if available,
    # otherwise fall back to val (note: val was used for early stopping,
    # so AUC on val is optimistic).
    logger.info("Evaluating best model for per-species AUC...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
    eval_loader = test_loader if test_loader is not None else val_loader
    eval_split  = "test" if test_loader is not None else "val (no test set)"
    _, _, best_mean_auc, per_species_aucs = evaluate(model, eval_loader, device, dist_info)
    logger.info(f"Best model {eval_split} AUC (mean): {best_mean_auc:.4f}")
    auc_rows = [(dataset.species_cols[s], per_species_aucs[s]) for s in sorted(per_species_aucs)]
    import pandas as pd
    pd.DataFrame(auc_rows, columns=["species", "auc"]).to_csv(
        os.path.join(args.output_dir, "per_species_auc_jsdm.csv"), index=False)
    logger.info(f"Per-species AUC saved to {args.output_dir}/per_species_auc_jsdm.csv")

    # Extract interaction matrix
    logger.info("Extracting species interaction matrix...")
    model.eval()
    interactions = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.interaction_extract_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(
                input_ids=batch["input_ids"], source_ids=batch["source_ids"],
                source_idx=batch["source_idx"],
                target_site_idx=batch["target_site_idx"], env_data=batch["env_data"],
                target_env=batch["target_env"],
                labels=batch["labels"],
                spatial_dist_pairwise=dist_info["spatial_dist_pairwise"],
                temporal_dist_pairwise=dist_info["temporal_dist_pairwise"],
                output_attentions=True,
            )
            interactions.append(extract_interaction_matrix(output).cpu())

    interaction_matrix = torch.cat(interactions, dim=0).mean(dim=0).numpy()
    np.save(os.path.join(args.output_dir, "interaction_matrix.npy"), interaction_matrix)

    with open(os.path.join(args.output_dir, "species_names.json"), "w") as f:
        json.dump(dataset.species_cols, f)

    logger.info(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
