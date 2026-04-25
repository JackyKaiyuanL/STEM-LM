import argparse
import csv
import json
import logging
import os
import time

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from jsdm_model import JSDMConfig, JSDMForMaskedSpeciesPrediction, extract_interaction_matrix
from jsdm_data import create_dataloaders, save_splits, load_splits, build_val_loaders_fixed_p

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_rate(s):
    if isinstance(s, str) and (s == "rand" or s.startswith("rand:")):
        return s
    return float(s)


def _parse_blind_pct(s):
    if isinstance(s, str) and s.lower() == "auto":
        return "auto"
    return float(s)

def move_dist_info_to_device(dist_info, device):
    out = dict(dist_info)
    for k in ("site_lats", "site_lons", "site_times"):
        out[k] = out[k].to(device)
    return out


def compute_class_weights(species_data, train_indices, beta=0.999):
    # Effective-number-of-samples class weighting
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
        site_lats=dist_info["site_lats"],
        site_lons=dist_info["site_lons"],
        site_times=dist_info["site_times"],
        euclidean=dist_info.get("euclidean", False),
    )


def _gate_l1_penalty(output) -> Optional[torch.Tensor]:

    gl = getattr(output, "gate_logits", None)
    if not gl:
        return None
    per_layer = [F.relu(g).mean() for g in gl if g is not None]
    if not per_layer:
        return None
    return torch.stack(per_layer).mean()


def train_epoch(model, loader, optimizer, scheduler, device, dist_info, epoch,
                loss_weight=None, log_interval=50, max_grad_norm=1.0,
                gate_l1: float = 0.0):
    model.train()
    total_loss, total_correct, total_masked, num_batches = 0, 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        B = batch["input_ids"].shape[0]
        w = loss_weight[None, :].expand(B, -1).to(device) if loss_weight is not None else None

        output = _forward(model, batch, dist_info, loss_weight=w)
        loss = output.loss
        if gate_l1 > 0.0:
            pen = _gate_l1_penalty(output)
            if pen is not None:
                loss = loss + gate_l1 * pen

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
    per_species_auprcs = {}
    total_correct, total_masked = 0, 0
    aucs, auprcs = [], []
    for s in range(S):
        if len(species_labels[s]) > 0:
            arr_p = np.array(species_preds[s])
            arr_l = np.array(species_labels[s])
            total_correct += ((arr_p > 0.5) == arr_l).sum()
            total_masked  += len(arr_l)
            if len(set(species_labels[s])) == 2 and not np.isnan(arr_p).any():
                auc_s   = roc_auc_score(arr_l, arr_p)
                auprc_s = average_precision_score(arr_l, arr_p)
                aucs.append(auc_s); auprcs.append(auprc_s)
                per_species_aucs[s]   = auc_s
                per_species_auprcs[s] = auprc_s
    mean_auc   = float(np.mean(aucs))   if aucs   else float("nan")
    mean_auprc = float(np.mean(auprcs)) if auprcs else float("nan")
    acc = total_correct / max(total_masked, 1)

    return (total_loss / max(num_batches, 1), acc,
            mean_auc, mean_auprc,
            per_species_aucs, per_species_auprcs)


def main():
    parser = argparse.ArgumentParser(description="Train STEM-LM")
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--num_source_sites", type=int, default=64)
    parser.add_argument("--blind_percentile", type=_parse_blind_pct, default="auto",
                        help="Percentile of normalized spatial pairwise distance used as the "
                             "blind radius. Default 'auto': picked via the half-decay rule on "
                             "the dataset's Jaccard(d) curve. Pass a float to override.")
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
    parser.add_argument("--p", type=_parse_rate, default=0.15,
                        help="Per-row mask rate. Float in [0,1] or 'rand[:lo,hi]' "
                             "(Uniform[lo,hi] per row; bare 'rand' = 'rand:0.0,1.0').")
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
    parser.add_argument(
        "--class_weighting",
        type=float,
        nargs="?",
        const=0.999,
        default=0.999,
        help="Effective-number class weighting beta in (0,1). "
             "Default 0.999 (enabled even if not provided). "
             "Pass e.g. '--class_weighting 0.99' to reduce weighting. "
             "Use --no_class_weighting to disable.",
    )
    parser.add_argument(
        "--no_class_weighting",
        action="store_true",
        help="Disable per-species loss weighting (overrides --class_weighting).",
    )
    parser.add_argument("--env_cols", nargs="+", default=None,
                        help="Explicit list of env column names. If not set, columns with 'env_' "
                             "prefix are used. Useful for datasets with non-prefixed env columns "
                             "(e.g. annualtemp, annualprec).")
    parser.add_argument("--fold", choices=["random", "h3", "grid"], default="h3",
                        help="Train/val/test split strategy. 'h3' (default): spatial blocks via "
                             "H3 hexagonal grid (real lat/lon only). 'grid': spatial blocks via "
                             "regular 2D grid (euclidean_coords only). 'random': shuffled rows.")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Block resolution for spatial splits. For --fold h3, this is the H3 "
                             "resolution in [0, 15] (default 2 ≈ 183 km edge). For --fold grid, "
                             "this is the grid side length (default 20 → 20×20 cells). Not valid "
                             "with --fold random.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping max norm (default 1.0).")
    parser.add_argument("--interaction_extract_batches", type=int, default=20,
                        help="Number of val batches used to estimate the species "
                             "interaction matrix at the end of training (default 20).")
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Path to a splits.json (written by a previous training run). "
                             "When set, overrides --fold / --resolution / "
                             "--seed for the split, so train/val/test are exactly reproduced.")
    parser.add_argument("--no_save_splits", action="store_true",
                        help="Skip writing splits.json to output_dir.")
    parser.add_argument("--val_p_list", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="Fixed mask rates for val AUC (deterministic per batch). "
                             "Mean AUC across these drives best_model.pt selection, giving "
                             "apples-to-apples comparison across runs (default 0.25 0.5 0.75 1.0).")
    parser.add_argument("--ablation", choices=["full", "no_st", "no_env", "no_st_env"],
                        default="full",
                        help="Ablation mode. 'full' uses both ST and Env cross-attention "
                             "with a gated combination; the others disable one or both.")
    parser.add_argument("--temporal_fire_init_periods", type=float, nargs="+", default=None,
                        help="Init periods (days) for learnable sin/cos channels in the "
                             "temporal FIRE bias. The length of this list is the number of "
                             "periodic channels K; each ω_k is learnable. Omit or pass empty "
                             "to disable periodicity (legacy monotone FIRE). "
                             "Recommended: '365 180 730 1825' (annual / semi / biennial / "
                             "5-yr) or '365 180 120 91 730 1825' (+ sub-annual for "
                             "multivoltine insects).")
    parser.add_argument("--fire_no_zero_init_periodic", action="store_true",
                        help="Disable zero-init of periodic channels in FIRE's MLP. "
                             "By default the periodic (sin/cos) weights start at zero so "
                             "the module is arithmetically equivalent to legacy FIRE at step 0; "
                             "the optimizer must earn the periodic contribution.")
    parser.add_argument("--per_species_scales", action="store_true",
                        help="Per-species spatial/temporal scales applied to the distance "
                             "input of FIRE: d_eff = d * exp(species_log_scale_s). Each "
                             "species learns its own decay range in space and time. "
                             "Init at 0 → multiplier 1, identical to legacy at step 0. "
                             "Adds 2*S parameters; registered no-decay so AdamW does not "
                             "drag the scalars to zero faster than per-species gradient.")
    parser.add_argument("--gate_hidden_size", type=int, default=None,
                        help="Bottleneck width of the ST/Env combine_gate MLP. "
                             "Default: hidden_size // 8.")
    parser.add_argument("--gate_init_bias", type=float, default=0.0,
                        help="Initial bias of combine_gate's final Linear. Negative values "
                             "(e.g. -2.0 → sigmoid≈0.12) start training with the gate "
                             "favoring Env, so ST must earn its weight.")
    parser.add_argument("--gate_l1", type=float, default=0.0,
                        help="L1 penalty weight on ReLU(gate_logit) (positive side only). "
                             "Encourages gate to remain near its Env-biased init unless "
                             "ST genuinely helps. 0 disables.")
    args = parser.parse_args()

    if args.splits_path is None:
        if args.fold == "random":
            if args.resolution is not None:
                raise ValueError("--resolution is not valid with --fold random.")
        elif args.fold == "h3":
            if args.resolution is None:
                args.resolution = 2
            if not (0 <= int(args.resolution) <= 15):
                raise ValueError("--resolution for --fold h3 must be an integer in [0, 15].")
        else:  # grid
            if args.resolution is None:
                args.resolution = 20
            if int(args.resolution) < 1:
                raise ValueError("--resolution for --fold grid must be a positive integer.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    saved_splits = None
    if args.splits_path is not None:
        logger.info(f"Loading splits from {args.splits_path}")
        saved_splits = load_splits(args.splits_path)

    train_loader, val_loader, test_loader, dataset, dist_info, splits = create_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_source_sites=args.num_source_sites,
        p=args.p,
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
        resolution=args.resolution,
        saved_splits=saved_splits,
    )

    if not args.no_save_splits:
        splits_out = os.path.join(args.output_dir, "splits.json")
        save_splits(
            splits_out, splits["train"], splits["val"], splits["test"],
            num_rows=len(dataset),
            meta={
                "fold":          args.fold if saved_splits is None else "loaded",
                "resolution":    args.resolution,
                "train_frac":    args.train_frac,
                "test_frac":     args.test_frac,
                "seed":          args.seed,
                "source":        args.splits_path if saved_splits is not None else None,
            },
        )
        logger.info(f"Splits saved to {splits_out}")

    loss_weight = None
    if not args.no_class_weighting:
        beta = args.class_weighting
        loss_weight = compute_class_weights(
            dataset.species_data, train_loader.dataset.indices, beta=beta
        )
        logger.info(
            f"Class weighting (beta={beta:.3f}): weight range "
            f"[{loss_weight.min().item():.3f}, {loss_weight.max().item():.3f}]"
        )

    use_temporal = dist_info["max_temporal_dist"] > 0
    if not use_temporal:
        logger.info("Temporal FIRE bias disabled (no temporal variation in data)")

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
        temporal_fire_init_periods=(
            tuple(args.temporal_fire_init_periods)
            if args.temporal_fire_init_periods else None
        ),
        fire_zero_init_periodic=(not args.fire_no_zero_init_periodic),
        per_species_scales=args.per_species_scales,
        gate_hidden_size=args.gate_hidden_size,
        gate_init_bias=args.gate_init_bias,
        ablation=args.ablation,
        p=args.p,
    )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    model = JSDMForMaskedSpeciesPrediction(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params:,} parameters, {config.num_species} species")

    if args.gradient_checkpointing:
        model.model.encoder.gradient_checkpointing = True

    no_decay_keys = ("norm", "bias", "combine_gate",
                     "species_log_spatial_scale", "species_log_temporal_scale")
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay_params if any(k in n for k in no_decay_keys) else decay_params).append(p)
    optimizer = AdamW(
        [
            {"params": decay_params,   "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    total_steps = len(train_loader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    dist_info = move_dist_info_to_device(dist_info, device)

    fixed_val_loaders = build_val_loaders_fixed_p(
        dataset, splits["val"], dist_info, args.val_p_list,
        batch_size=args.batch_size, num_workers=args.num_workers, base_seed=args.seed,
    )

    log_csv = os.path.join(args.output_dir, "training_log.csv")
    per_p_header = []
    for p, _ in fixed_val_loaders:
        per_p_header += [f"val_loss_p{p:.2f}", f"val_acc_p{p:.2f}",
                         f"val_auc_p{p:.2f}", f"val_auprc_p{p:.2f}"]
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc",
             *per_p_header,
             "val_loss_mean", "val_acc_mean", "val_auc_mean", "val_auprc_mean",
             "lr", "elapsed_s"]
        )

    best_val_auprc_mean = -float("inf")
    best_val_auc_mean = -float("inf")
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, dist_info, epoch,
            loss_weight=loss_weight, max_grad_norm=args.max_grad_norm,
            gate_l1=args.gate_l1,
        )
        per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_nauc = [], [], [], [], []
        for p, loader in fixed_val_loaders:
            l, a, u, ap, per_species, _ = evaluate(model, loader, device, dist_info)
            per_p_loss.append(l)
            per_p_acc.append(a)
            per_p_auc.append(u)
            per_p_auprc.append(ap)
            per_p_nauc.append(len(per_species))
        val_loss_mean  = float(np.mean(per_p_loss))  if per_p_loss  else float("nan")
        val_acc_mean   = float(np.mean(per_p_acc))   if per_p_acc   else float("nan")
        val_auc_mean   = float(np.mean(per_p_auc))   if per_p_auc   else float("nan")
        val_auprc_mean = float(np.mean(per_p_auprc)) if per_p_auprc else float("nan")
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        per_p_str = " ".join(
            f"p{p:.2f}(loss={l:.3f},acc={a:.3f},auc={u:.3f},auprc={ap:.3f},n={n})"
            for (p, _), l, a, u, ap, n in zip(
                fixed_val_loaders, per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_nauc
            )
        )
        logger.info(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"{per_p_str} | mean auprc={val_auprc_mean:.4f} auc={val_auc_mean:.4f} | {elapsed:.1f}s"
        )
        with open(log_csv, "a", newline="") as f:
            row = [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}"]
            for l, a, u, ap in zip(per_p_loss, per_p_acc, per_p_auc, per_p_auprc):
                row += [f"{l:.6f}", f"{a:.6f}", f"{u:.6f}", f"{ap:.6f}"]
            row += [f"{val_loss_mean:.6f}", f"{val_acc_mean:.6f}",
                    f"{val_auc_mean:.6f}", f"{val_auprc_mean:.6f}",
                    f"{current_lr:.2e}", f"{elapsed:.1f}"]
            csv.writer(f).writerow(row)
        if val_auprc_mean > best_val_auprc_mean:
            best_val_auprc_mean = val_auprc_mean
            best_val_auc_mean = val_auc_mean
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"  → Best model saved (val_auprc_mean={val_auprc_mean:.4f})")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss_mean": val_loss_mean,
                "val_auc_mean": val_auc_mean,
                "val_auprc_mean": val_auprc_mean,
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))

    logger.info("Evaluating best model on fixed-p test set...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device))
    eval_split   = "test" if len(splits["test"]) > 0 else "val (no test set)"
    eval_indices = splits["test"] if len(splits["test"]) > 0 else splits["val"]
    eval_loaders = build_val_loaders_fixed_p(
        dataset, eval_indices, dist_info, args.val_p_list,
        batch_size=args.batch_size, num_workers=args.num_workers,
        base_seed=args.seed + 10_000,
    )
    per_p_auc = {}
    per_p_auprc = {}
    per_p_per_species_auc   = {}
    per_p_per_species_auprc = {}
    for p, loader in eval_loaders:
        _, _, m_auc, m_auprc, per_sp_auc, per_sp_auprc = evaluate(model, loader, device, dist_info)
        per_p_auc[p]   = m_auc
        per_p_auprc[p] = m_auprc
        per_p_per_species_auc[p]   = per_sp_auc
        per_p_per_species_auprc[p] = per_sp_auprc
        logger.info(f"{eval_split} p={p:.2f}  mean AUPRC = {m_auprc:.4f}  AUC = {m_auc:.4f}  "
                    f"(species n={len(per_sp_auc)})")
    best_mean_auc   = float(np.mean(list(per_p_auc.values())))
    best_mean_auprc = float(np.mean(list(per_p_auprc.values())))
    logger.info(f"{eval_split} mean over p={list(per_p_auc.keys())}: "
                f"AUPRC={best_mean_auprc:.4f}  AUC={best_mean_auc:.4f}")

    import pandas as pd
    all_species = sorted({s for per_sp in per_p_per_species_auc.values() for s in per_sp})
    rows = []
    for s in all_species:
        row = {"species": dataset.species_cols[s]}
        for p in per_p_auc:
            row[f"auc_p{p:.2f}"]   = per_p_per_species_auc[p].get(s, float("nan"))
            row[f"auprc_p{p:.2f}"] = per_p_per_species_auprc[p].get(s, float("nan"))
        row["auc_mean"]   = float(np.nanmean([row[f"auc_p{p:.2f}"]   for p in per_p_auc]))
        row["auprc_mean"] = float(np.nanmean([row[f"auprc_p{p:.2f}"] for p in per_p_auc]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.output_dir, "per_species_auc_jsdm.csv"), index=False)
    logger.info(f"Per-species AUC/AUPRC saved to {args.output_dir}/per_species_auc_jsdm.csv")

    summary = {
        "ablation":           config.ablation,
        "num_params":         num_params,
        "best_val_auprc_mean": best_val_auprc_mean,
        "best_val_auc_mean":   best_val_auc_mean,
        "test_mean_auprc":    best_mean_auprc,
        "test_mean_auc":      best_mean_auc,
        "test_auprc_by_p":    {f"{p:.2f}": per_p_auprc[p] for p in per_p_auprc},
        "test_auc_by_p":      {f"{p:.2f}": per_p_auc[p]   for p in per_p_auc},
        "eval_split":         eval_split,
        "num_species":        config.num_species,
        "num_epochs":         args.num_epochs,
        "seed":               args.seed,
    }
    with open(os.path.join(args.output_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
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
                site_lats=dist_info["site_lats"],
                site_lons=dist_info["site_lons"],
                site_times=dist_info["site_times"],
                spatial_scale_km=dist_info["spatial_scale_km"],
                temporal_scale_days=dist_info["temporal_scale_days"],
                euclidean=dist_info.get("euclidean", False),
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
