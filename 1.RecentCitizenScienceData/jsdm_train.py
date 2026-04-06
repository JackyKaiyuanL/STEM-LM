"""Training script for ST-JSDM (citizen-science variant)."""

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from jsdm_model import JSDMConfig, JSDMForMaskedSpeciesPrediction, extract_interaction_matrix
from jsdm_data import create_dataloaders

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _model_kwargs(batch):
    return dict(
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
    )


def train_epoch(model, loader, optimizer, scheduler, device, epoch, log_interval=50):
    model.train()
    total_loss, total_correct, total_masked, num_batches = 0, 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(**_model_kwargs(batch))

        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        mask = batch["labels"] != -100
        if mask.any():
            preds   = (output.logits[mask] > 0).long()
            targets = batch["labels"][mask].long()
            total_correct += (preds == targets).sum().item()
            total_masked  += mask.sum().item()

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {total_loss/num_batches:.4f} | "
                f"Acc: {total_correct/max(total_masked,1):.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return total_loss / max(num_batches, 1), total_correct / max(total_masked, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct, total_masked, num_batches = 0, 0, 0, 0

    for batch in loader:
        batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(**_model_kwargs(batch))

        total_loss += output.loss.item()
        num_batches += 1
        mask = batch["labels"] != -100
        if mask.any():
            preds   = (output.logits[mask] > 0).long()
            targets = batch["labels"][mask].long()
            total_correct += (preds == targets).sum().item()
            total_masked  += mask.sum().item()

    return total_loss / max(num_batches, 1), total_correct / max(total_masked, 1)


def main():
    parser = argparse.ArgumentParser(description="Train ST-JSDM")
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--num_source_sites",    type=int,   default=64)
    parser.add_argument("--hidden_size",         type=int,   default=256)
    parser.add_argument("--num_attention_heads", type=int,   default=8)
    parser.add_argument("--num_hidden_layers",   type=int,   default=4)
    parser.add_argument("--intermediate_size",   type=int,   default=512)
    parser.add_argument("--dropout",             type=float, default=0.1)
    parser.add_argument("--batch_size",          type=int,   default=32)
    parser.add_argument("--num_epochs",          type=int,   default=50)
    parser.add_argument("--learning_rate",       type=float, default=1e-4)
    parser.add_argument("--weight_decay",        type=float, default=0.01)
    parser.add_argument("--mlm_probability",     type=float, default=0.15)
    parser.add_argument("--mask_value",          type=float, default=-1.0)
    parser.add_argument("--train_frac",          type=float, default=0.8)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--num_workers",         type=int,   default=0)
    parser.add_argument("--output_dir",          type=str,   default="./jsdm_output")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--spatial_scale_km",    type=float, default=None)
    parser.add_argument("--temporal_scale_days", type=float, default=None)
    parser.add_argument("--time_window_days",    type=float, default=None)
    parser.add_argument("--sampling_strategy",   type=str,   default="nearest",
                        choices=["nearest", "weighted"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, dataset = create_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_source_sites=args.num_source_sites,
        mlm_probability=args.mlm_probability,
        mask_value=args.mask_value,
        train_frac=args.train_frac,
        num_workers=args.num_workers,
        seed=args.seed,
        spatial_scale_km=args.spatial_scale_km,
        temporal_scale_days=args.temporal_scale_days,
        time_window_days=args.time_window_days,
        sampling_strategy=args.sampling_strategy,
    )

    config = JSDMConfig(
        num_species             = dataset.num_species,
        mask_value_init         = args.mask_value,
        num_source_sites        = args.num_source_sites,
        max_spatial_dist        = dataset.max_spatial_dist  * 1.1,
        max_temporal_dist       = dataset.max_temporal_dist * 1.1,
        max_doy_dist            = max(dataset.max_doy_dist, 1.0),
        num_env_vars            = dataset.num_env_vars,
        hidden_size             = args.hidden_size,
        num_attention_heads     = args.num_attention_heads,
        num_hidden_layers       = args.num_hidden_layers,
        intermediate_size       = args.intermediate_size,
        hidden_dropout_prob     = args.dropout,
        attention_probs_dropout_prob = args.dropout,
        mlm_probability         = args.mlm_probability,
    )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    model = JSDMForMaskedSpeciesPrediction(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params:,} parameters, {config.num_species} species")

    if args.gradient_checkpointing:
        model.model.encoder.gradient_checkpointing = True

    optimizer    = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.num_epochs
    scheduler    = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    best_val_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss,   val_acc   = evaluate(model, val_loader, device)
        logger.info(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | {time.time()-t0:.1f}s"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info("  → Best model saved")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))

    logger.info("Extracting species interaction matrix...")
    model.eval()
    interactions = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 20:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(**{**_model_kwargs(batch), "output_attentions": True})
            interactions.append(extract_interaction_matrix(output).cpu())

    interaction_matrix = torch.cat(interactions, dim=0).mean(dim=0).numpy()
    np.save(os.path.join(args.output_dir, "interaction_matrix.npy"), interaction_matrix)

    with open(os.path.join(args.output_dir, "species_names.json"), "w") as f:
        json.dump(dataset.species_cols, f)

    logger.info(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
