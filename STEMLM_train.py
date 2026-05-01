import argparse
import csv
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from STEMLM_model import JSDMConfig, JSDMForMaskedSpeciesPrediction, extract_cooccurrence_matrix
from STEMLM_data import create_dataloaders, save_splits, load_splits, build_val_loaders_fixed_p
from STEMLM_metric import (
    compute_per_species_metrics,
    summarize_per_species_metrics,
    bagged_evaluate_at_p,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DDP utilities
# =============================================================================
class DistEnv:
    """
    Distributed environment helper. Auto-detects torchrun-style env vars.
    If torchrun isn't used, this object reports world_size=1, rank=0 and
    all the helpers become no-ops, so the same code runs single-GPU.
    """
    def __init__(self):
        self.is_distributed = "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ \
                              and int(os.environ.get("WORLD_SIZE", "1")) > 1
        if self.is_distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0

    def setup(self, backend: str = "nccl"):
        if self.is_distributed and not dist.is_initialized():
            dist.init_process_group(backend=backend)
            torch.cuda.set_device(self.local_rank)

    def cleanup(self):
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a tensor and divide by world_size."""
        if not self.is_distributed:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / self.world_size

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather_object(self, obj):
        if not self.is_distributed:
            return [obj]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered

    def broadcast_object(self, obj, src: int = 0):
        if not self.is_distributed:
            return obj
        container = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(container, src=src)
        return container[0]


def log_main(env: "DistEnv", msg: str, level: int = logging.INFO):
    if env.is_main:
        logger.log(level, msg)


def _parse_rate(s):
    if isinstance(s, str) and (s == "unif" or s.startswith("unif:") or s.startswith("beta:")):
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
    if not (0.0 < beta < 1.0):
        raise ValueError(f"class_weighting_beta must be in (0, 1), got {beta}")
    n_s = species_data[train_indices].sum(axis=0).clip(min=1)
    effective_n = (1 - beta ** n_s) / (1 - beta)
    w = 1.0 / effective_n
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _forward(model, batch, dist_info, loss_weight=None,
             loss_type: str = "bce",
             focal_alpha: float = 0.25, focal_gamma: float = 2.0):
    return model(
        input_ids=batch["input_ids"],
        source_ids=batch["source_ids"],
        source_idx=batch["source_idx"],
        target_site_idx=batch["target_site_idx"],
        env_data=batch["env_data"],
        target_env=batch["target_env"],
        labels=batch["labels"],
        loss_weight=loss_weight,
        loss_type=loss_type,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        site_lats=dist_info["site_lats"],
        site_lons=dist_info["site_lons"],
        site_times=dist_info["site_times"],
        euclidean=dist_info.get("euclidean", False),
    )


def train_epoch(model, loader, optimizer, scheduler, device, dist_info, epoch,
                loss_weight=None, log_interval=50, max_grad_norm=1.0,
                amp_dtype=None, grad_scaler=None,
                grad_accum_steps: int = 1, env: Optional[DistEnv] = None,
                loss_type: str = "bce",
                focal_alpha: float = 0.25, focal_gamma: float = 2.0):
    """
    amp_dtype: None | torch.bfloat16 | torch.float16
    grad_scaler: torch.amp.GradScaler instance (only used for fp16)
    grad_accum_steps: gradient accumulation steps (>=1)
    env: DistEnv. If None, treated as single-process.
    """
    if env is None:
        env = DistEnv()

    if env.is_distributed and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    model.train()
    total_loss, total_correct, total_masked, num_batches = 0.0, 0, 0, 0
    use_amp = amp_dtype is not None and device.type == "cuda"

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        B = batch["input_ids"].shape[0]
        w = loss_weight[None, :].expand(B, -1).to(device) if loss_weight is not None else None

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = _forward(model, batch, dist_info, loss_weight=w,
                                  loss_type=loss_type,
                                  focal_alpha=focal_alpha, focal_gamma=focal_gamma)
                loss = output.loss
        else:
            output = _forward(model, batch, dist_info, loss_weight=w,
                              loss_type=loss_type,
                              focal_alpha=focal_alpha, focal_gamma=focal_gamma)
            loss = output.loss

        loss_to_back = loss / grad_accum_steps

        is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(loader))
        if env.is_distributed and not is_accum_step and isinstance(model, DDP):
            with model.no_sync():
                if grad_scaler is not None:
                    grad_scaler.scale(loss_to_back).backward()
                else:
                    loss_to_back.backward()
        else:
            if grad_scaler is not None:
                grad_scaler.scale(loss_to_back).backward()
            else:
                loss_to_back.backward()

        if is_accum_step:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        mask = batch["labels"] != -100
        if mask.any():
            preds = (output.logits[mask] > 0).long()
            targets = batch["labels"][mask].long()
            total_correct += (preds == targets).sum().item()
            total_masked += mask.sum().item()

        if (batch_idx + 1) % log_interval == 0 and env.is_main:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {total_loss/num_batches:.4f} | "
                f"Acc: {total_correct/max(total_masked,1):.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    if env.is_distributed:
        agg = torch.tensor(
            [total_loss, num_batches, total_correct, total_masked],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_loss, num_batches, total_correct, total_masked = agg.tolist()

    return total_loss / max(num_batches, 1), total_correct / max(total_masked, 1)


@torch.no_grad()
def evaluate(model, loader, device, dist_info, amp_dtype=None,
             env: Optional[DistEnv] = None,
             loss_type: str = "bce",
             focal_alpha: float = 0.25, focal_gamma: float = 2.0):

    if env is None:
        env = DistEnv()

    model.eval()
    total_loss, num_batches = 0.0, 0
    S = (model.module if isinstance(model, DDP) else model).config.num_species
    species_preds  = [[] for _ in range(S)]
    species_labels = [[] for _ in range(S)]
    use_amp = amp_dtype is not None and device.type == "cuda"

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                output = _forward(model, batch, dist_info,
                                  loss_type=loss_type,
                                  focal_alpha=focal_alpha, focal_gamma=focal_gamma)
        else:
            output = _forward(model, batch, dist_info,
                              loss_type=loss_type,
                              focal_alpha=focal_alpha, focal_gamma=focal_gamma)

        total_loss += output.loss.item()
        num_batches += 1

        probs  = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()
        labels = batch["labels"].squeeze(-1).cpu().numpy()
        mask   = labels != -100
        for s in range(S):
            b_mask = mask[:, s]
            if b_mask.any():
                species_preds[s].extend(probs[b_mask, s].tolist())
                species_labels[s].extend(labels[b_mask, s].tolist())

    if env.is_distributed:
        agg = torch.tensor([total_loss, num_batches], dtype=torch.float64, device=device)
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_loss, num_batches = agg.tolist()

        gathered_preds  = env.all_gather_object(species_preds)
        gathered_labels = env.all_gather_object(species_labels)
        species_preds  = [sum((g[s] for g in gathered_preds), []) for s in range(S)]
        species_labels = [sum((g[s] for g in gathered_labels), []) for s in range(S)]

    max_n = max((len(species_labels[s]) for s in range(S)), default=0)
    if max_n == 0:
        empty = {"mean_auc_roc": float("nan"), "mean_auc_pr": float("nan"),
                 "mean_cbi": float("nan"),
                 "mean_brier": float("nan"), "mean_ece": float("nan"),
                 "auc_roc_q25": float("nan"), "auc_roc_q50": float("nan"),
                 "auc_roc_q75": float("nan"),
                 "n_species": 0}
        return total_loss / max(num_batches, 1), 0.0, empty, {}
    probs_arr  = np.full((max_n, S), 0.0, dtype=np.float64)
    labels_arr = np.full((max_n, S), -100, dtype=np.int64)
    total_correct, total_masked = 0, 0
    for s in range(S):
        n_s = len(species_labels[s])
        if n_s == 0:
            continue
        arr_p = np.asarray(species_preds[s],  dtype=np.float64)
        arr_l = np.asarray(species_labels[s], dtype=np.int64)
        probs_arr[:n_s, s]  = arr_p
        labels_arr[:n_s, s] = arr_l
        total_correct += int(((arr_p > 0.5) == arr_l).sum())
        total_masked  += n_s

    per_sp = compute_per_species_metrics(probs_arr, labels_arr)
    summary = summarize_per_species_metrics(per_sp)
    acc = total_correct / max(total_masked, 1)

    return total_loss / max(num_batches, 1), acc, summary, per_sp


def main():
    parser = argparse.ArgumentParser(description="Train STEM-LM")
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--num_source_sites", type=int, default=64)
    parser.add_argument("--blind_percentile", type=_parse_blind_pct, default="auto",
                        help="Percentile of normalized spatial pairwise distance used as the "
                             "blind radius. Default 'auto': picked via the half-decay rule on "
                             "the dataset's Jaccard(d) curve. Pass a float to override.")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--num_env_groups", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--p", type=_parse_rate, default=0.15,
                        help="Per-row mask rate. Float in [0,1], 'unif[:lo,hi]' "
                             "(Uniform[lo,hi] per row; bare 'unif' = 'unif:0.0,1.0'), or "
                             "'beta:alpha,beta' (Beta(alpha, beta) per row; e.g. "
                             "'beta:2,1' biases toward p=1, 'beta:0.5,0.5' is U-shaped).")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of data held out as test set for final AUC. "
                             "Val = 1 - train_frac - test_frac. Set to 0 to disable "
                             "(final AUC reported on val, same as early-stopping set).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./STEMLM_output")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="none",
        choices=["none", "bf16", "fp16"],
        help="Mixed-precision training mode. bf16 recommended on A40/L40/A100/H100; "
             "halves activation memory with no loss-scaling needed. fp16 is older and "
             "needs GradScaler. 'none' = full fp32 (highest VRAM)."
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum_steps. "
             "Use to keep large effective batch when reducing physical batch_size for VRAM."
    )
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
        default=None,
        help="Effective-number class weighting beta in (0,1). Off by default; "
             "pass '--class_weighting' (alone, uses beta=0.999) or "
             "'--class_weighting 0.99' to enable. Recommended for BCE only; "
             "with focal it's redundant and hurts AUC.",
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
    parser.add_argument("--cooccurrence_extract_batches", type=int, default=20,
                        help="Number of val batches used to estimate the species "
                             "cooccurrence matrix at the end of training (default 20).")
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Path to a splits.json (written by a previous training run). "
                             "When set, overrides --fold / --resolution / "
                             "--seed for the split, so train/val/test are exactly reproduced.")
    parser.add_argument("--no_save_splits", action="store_true",
                        help="Skip writing splits.json to output_dir.")
    parser.add_argument("--val_p_list", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="Fixed mask rates for val AUC (deterministic per batch). "
                             "Mean AUC across these drives best_model.pt selection. "
                             "AUPRC is reported but not used for selection. "
                             "Default 0.25 0.5 0.75 1.0.")
    parser.add_argument("--ablation", choices=["full", "no_st", "no_env", "no_st_env"],
                        default="full",
                        help="Ablation mode. 'full' uses both ST and Env cross-attention "
                             "(summed residual); the others disable one or both.")
    parser.add_argument("--temporal_fire_init_periods", type=float, nargs="+", default=None,
                        help="Init periods (days) for learnable sin/cos channels added to "
                             "FIRE temporal distance bias on ST attention scores. "
                             "Periodic on Δt directly; per-species scales (if enabled) "
                             "rescale Δt before the cos/sin. Omit to disable.")
    parser.add_argument("--fire_no_zero_init_periodic", action="store_true",
                        help="Disable zero-init of the periodic columns of FIRE's input "
                             "linear. By default the periodic contribution starts at zero "
                             "so the monotone bias is unaffected at step 0.")
    parser.add_argument("--per_species_env_rank", type=int, default=8,
                        help="Rank of the parallel per-species env head bolted in "
                             "alongside the shared env encoder. Reads raw target_env "
                             "via low-rank A∈(E,r)·B∈(r,S) (A zero-init, monotone safe) "
                             "+ per-species bias. Active in full / no_st ablations; "
                             "silent in no_env / no_st_env. Default 0 (disabled).")
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="bce",
                        help="Loss function. 'bce' = sigmoid BCE (default, "
                             "backward-compatible). 'focal' = sigmoid focal loss "
                             "(Lin et al. 2017). Focal trades AUC for CBI: on "
                             "eButterfly, BCE p13 vs focal: AUC 0.882->0.857, "
                             "CBI 0.298->0.638.")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha (positive-class weight). Set <0 "
                             "to disable alpha-balancing. Ignored when "
                             "--loss_type=bce. Default 0.25 (RetinaNet).")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss focusing parameter. 0 reduces to "
                             "weighted BCE. Ignored when --loss_type=bce. "
                             "Default 2.0 (RetinaNet).")
    parser.add_argument("--absence_mask_eval", action="store_true",
                        help="(Default ON; this flag is a no-op kept for "
                             "backward compat.) Run absence-mask test block "
                             "(mask all absences + p presences; presence-only "
                             "scenario). Use --no_absence_mask_eval to skip.")
    parser.add_argument("--no_absence_mask_eval", action="store_true",
                        help="Skip the absence-mask test block.")
    parser.add_argument("--absence_mask_p_list", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="Presence-mask rates for absence-mask eval.")
    parser.add_argument("--test_bag_K", type=int, default=10,
                        help="K-pass test-time eval. Each pass uses the SAME mask pattern (the "
                             "FixedPValCollator base_seed is held fixed) and re-seeds only "
                             "source-pool sampling, so bagging averages source-resample variance "
                             "without inflating numbers via mask-pattern ensembling. K=1 disables.")
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

    env = DistEnv()
    env.setup(backend="nccl")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = env.device()
    if env.is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    env.barrier()

    if env.is_distributed:
        log_main(env,
            f"Distributed training: world_size={env.world_size}, "
            f"backend=nccl, device={device}"
        )
    else:
        log_main(env, f"Single-process training, device={device}")

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
        euclidean_coords=args.euclidean_coords,
        no_time=args.no_time,
        fold_method=args.fold,
        resolution=args.resolution,
        saved_splits=saved_splits,
    )

    if env.is_distributed:
        from torch.utils.data import DataLoader
        train_dataset_obj = train_loader.dataset
        train_collator    = train_loader.collate_fn
        train_sampler = DistributedSampler(
            train_dataset_obj,
            num_replicas=env.world_size,
            rank=env.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=train_collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        log_main(env,
            f"Train loader sharded: per-rank batches={len(train_loader)}, "
            f"global batches/epoch={len(train_loader) * env.world_size}"
        )

    if saved_splits is None and not args.no_save_splits and env.is_main:
        splits_out = os.path.join(args.output_dir, "splits.json")
        save_splits(
            splits_out, splits["train"], splits["val"], splits["test"],
            num_rows=len(dataset),
            meta={
                "fold":          args.fold,
                "resolution":    args.resolution,
                "train_frac":    args.train_frac,
                "test_frac":     args.test_frac,
                "seed":          args.seed,
                "source":        None,
            },
        )
        logger.info(f"Splits saved to {splits_out}")
    env.barrier()

    loss_weight = None
    if args.class_weighting is not None:
        beta = args.class_weighting
        loss_weight = compute_class_weights(
            dataset.species_data, train_loader.dataset.indices, beta=beta
        )
        log_main(env,
            f"Class weighting (beta={beta:.3f}): weight range "
            f"[{loss_weight.min().item():.3f}, {loss_weight.max().item():.3f}]"
        )

    if args.loss_type == "focal":
        log_main(env,
            f"Loss: focal (alpha={args.focal_alpha}, gamma={args.focal_gamma})"
        )
    else:
        log_main(env, "Loss: bce")

    use_temporal = dist_info["max_temporal_dist"] > 0
    if not use_temporal:
        log_main(env, "Temporal FIRE bias disabled (no temporal variation in data)")

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
        ablation=args.ablation,
        p=args.p,
        per_species_env_rank=args.per_species_env_rank,
    )

    if env.is_main:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(config), f, indent=2)
    env.barrier()

    model = JSDMForMaskedSpeciesPrediction(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_main(env, f"Model: {num_params:,} parameters, {config.num_species} species")

    if args.gradient_checkpointing:
        model.model.encoder.gradient_checkpointing = True

    if env.is_distributed:
        model = DDP(
            model,
            device_ids=[env.local_rank],
            output_device=env.local_rank,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        log_main(env, "Model wrapped with DistributedDataParallel")

    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    amp_dtype = None
    grad_scaler = None
    if args.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
        log_main(env, "Mixed precision: bfloat16 (no GradScaler needed)")
    elif args.mixed_precision == "fp16":
        amp_dtype = torch.float16
        grad_scaler = torch.amp.GradScaler("cuda")
        log_main(env, "Mixed precision: float16 (using GradScaler)")
    else:
        log_main(env, "Mixed precision: disabled (full fp32)")

    if args.grad_accum_steps > 1:
        log_main(env,
            f"Gradient accumulation: {args.grad_accum_steps} steps "
            f"(effective batch = {args.batch_size * args.grad_accum_steps * env.world_size})"
        )

    no_decay_keys = ("norm", "bias",
                     "species_spatial_log_scale", "species_temporal_log_scale")
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
    
    total_steps = (len(train_loader) // max(args.grad_accum_steps, 1)) * args.num_epochs
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
                         f"val_auc_p{p:.2f}", f"val_auprc_p{p:.2f}",
                         f"val_cbi_p{p:.2f}"]
    if env.is_main:
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "train_acc",
                 *per_p_header,
                 "val_loss_mean", "val_acc_mean", "val_auc_mean", "val_auprc_mean",
                 "val_cbi_mean", "lr", "elapsed_s"]
            )

    best_val_auc_mean = -float("inf")
    best_val_auprc_mean = -float("inf")
    best_val_cbi_mean = -float("inf")
    start_epoch = 1

    resume_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
    if os.path.exists(resume_path):
        log_main(env, f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        unwrap(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if grad_scaler is not None and ckpt.get("grad_scaler_state_dict") is not None:
            grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc_mean   = ckpt.get("best_val_auc_mean",   -float("inf"))
        best_val_auprc_mean = ckpt.get("best_val_auprc_mean", -float("inf"))
        best_val_cbi_mean   = ckpt.get("best_val_cbi_mean",   -float("inf"))
        log_main(env, f"  → resumed at epoch {start_epoch}, "
                      f"best val_auc_mean so far={best_val_auc_mean:.4f}")
    env.barrier()

    for epoch in range(start_epoch, args.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, dist_info, epoch,
            loss_weight=loss_weight, max_grad_norm=args.max_grad_norm,
            amp_dtype=amp_dtype, grad_scaler=grad_scaler,
            grad_accum_steps=args.grad_accum_steps,
            env=env,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        )
        per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_cbi, per_p_nauc = [], [], [], [], [], []
        for p, loader in fixed_val_loaders:
            l, a, summary, per_sp = evaluate(
                model, loader, device, dist_info, amp_dtype=amp_dtype, env=env,
                loss_type=args.loss_type,
                focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
            )
            per_p_loss.append(l)
            per_p_acc.append(a)
            per_p_auc.append(summary["mean_auc_roc"])
            per_p_auprc.append(summary["mean_auc_pr"])
            per_p_cbi.append(summary.get("mean_cbi", float("nan")))
            per_p_nauc.append(summary["n_species"])
        val_loss_mean  = float(np.mean(per_p_loss))  if per_p_loss  else float("nan")
        val_acc_mean   = float(np.mean(per_p_acc))   if per_p_acc   else float("nan")
        val_auc_mean   = float(np.mean(per_p_auc))   if per_p_auc   else float("nan")
        val_auprc_mean = float(np.mean(per_p_auprc)) if per_p_auprc else float("nan")
        val_cbi_mean   = float(np.nanmean(per_p_cbi)) if per_p_cbi  else float("nan")
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        if env.is_main:
            per_p_str = " ".join(
                f"p{p:.2f}(loss={l:.3f},acc={a:.3f},auc={u:.3f},auprc={ap:.3f},cbi={c:.3f},n={n})"
                for (p, _), l, a, u, ap, c, n in zip(
                    fixed_val_loaders, per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_cbi, per_p_nauc
                )
            )
            logger.info(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"{per_p_str} | mean auc={val_auc_mean:.4f} auprc={val_auprc_mean:.4f} cbi={val_cbi_mean:.4f} | {elapsed:.1f}s"
            )
            with open(log_csv, "a", newline="") as f:
                row = [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}"]
                for l, a, u, ap, c in zip(per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_cbi):
                    row += [f"{l:.6f}", f"{a:.6f}", f"{u:.6f}", f"{ap:.6f}", f"{c:.6f}"]
                row += [f"{val_loss_mean:.6f}", f"{val_acc_mean:.6f}",
                        f"{val_auc_mean:.6f}", f"{val_auprc_mean:.6f}",
                        f"{val_cbi_mean:.6f}",
                        f"{current_lr:.2e}", f"{elapsed:.1f}"]
                csv.writer(f).writerow(row)

            if val_auc_mean > best_val_auc_mean:
                best_val_auc_mean = val_auc_mean
                best_val_auprc_mean = val_auprc_mean
                torch.save(
                    unwrap(model).state_dict(),
                    os.path.join(args.output_dir, "best_model.pt"),
                )
                logger.info(f"  → Best model saved (val_auc_mean={val_auc_mean:.4f})")

            if np.isfinite(val_cbi_mean) and val_cbi_mean > best_val_cbi_mean:
                best_val_cbi_mean = val_cbi_mean
                torch.save(
                    unwrap(model).state_dict(),
                    os.path.join(args.output_dir, "best_model_by_cbi.pt"),
                )
                logger.info(f"  → Best-by-CBI model saved (val_cbi_mean={val_cbi_mean:.4f})")

            torch.save({
                "epoch": epoch,
                "model_state_dict": unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "grad_scaler_state_dict": grad_scaler.state_dict() if grad_scaler is not None else None,
                "best_val_auc_mean": best_val_auc_mean,
                "best_val_auprc_mean": best_val_auprc_mean,
                "best_val_cbi_mean": best_val_cbi_mean,
                "val_loss_mean": val_loss_mean,
                "val_auc_mean": val_auc_mean,
                "val_auprc_mean": val_auprc_mean,
                "val_cbi_mean": val_cbi_mean,
            }, resume_path)

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": unwrap(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss_mean": val_loss_mean,
                    "val_auc_mean": val_auc_mean,
                    "val_auprc_mean": val_auprc_mean,
                }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))

        if env.is_distributed:
            best_state = torch.tensor([best_val_auc_mean, best_val_auprc_mean],
                                      dtype=torch.float64, device=device)
            dist.broadcast(best_state, src=0)
            best_val_auc_mean, best_val_auprc_mean = best_state.tolist()
        env.barrier()

    eval_split   = "test" if len(splits["test"]) > 0 else "val (no test set)"
    eval_indices = splits["test"] if len(splits["test"]) > 0 else splits["val"]
    log_main(env,
        f"Evaluating best model on fixed-p {eval_split} set "
        f"(K={args.test_bag_K} bagging passes per p)..."
    )

    best_state = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    unwrap(model).load_state_dict(best_state)

    per_p_auc = {}
    per_p_auprc = {}
    per_p_cbi = {}
    per_p_brier = {}
    per_p_ece = {}
    per_p_q25 = {}
    per_p_q50 = {}
    per_p_q75 = {}
    per_p_per_species: dict = {}
    for p in args.val_p_list:
        result = bagged_evaluate_at_p(
            unwrap(model), dataset, eval_indices, dist_info,
            p_value=p, bag_K=args.test_bag_K,
            batch_size=args.batch_size, device=device,
            num_workers=args.num_workers,
            base_seed=args.seed + 10_000,
            amp_dtype=amp_dtype,
            distributed_sampler=env.is_distributed,
        )
        s = result["summary"]
        per_p_auc[p]    = s["mean_auc_roc"]
        per_p_auprc[p]  = s["mean_auc_pr"]
        per_p_cbi[p]    = s["mean_cbi"]
        per_p_brier[p]  = s.get("mean_brier", float("nan"))
        per_p_ece[p]    = s.get("mean_ece", float("nan"))
        per_p_q25[p]    = s.get("auc_roc_q25", float("nan"))
        per_p_q50[p]    = s.get("auc_roc_q50", float("nan"))
        per_p_q75[p]    = s.get("auc_roc_q75", float("nan"))
        per_p_per_species[p] = result["per_species"]
        log_main(env,
            f"{eval_split} p={p:.2f}  bag(K={args.test_bag_K}) "
            f"AUC={s['mean_auc_roc']:.4f}  "
            f"AUCq25/50/75={per_p_q25[p]:.3f}/{per_p_q50[p]:.3f}/{per_p_q75[p]:.3f}  "
            f"AUPRC={s['mean_auc_pr']:.4f}  "
            f"CBI={s['mean_cbi']:.3f}  "
            f"Brier={per_p_brier[p]:.4f}  ECE={per_p_ece[p]:.4f}  "
            f"(n={s['n_species']})"
        )
    best_mean_auc   = float(np.mean(list(per_p_auc.values())))
    best_mean_auprc = float(np.mean(list(per_p_auprc.values())))
    best_mean_cbi   = float(np.nanmean(list(per_p_cbi.values()))) if per_p_cbi else float("nan")
    log_main(env,
        f"{eval_split} mean over p={list(per_p_auc.keys())}: "
        f"AUC={best_mean_auc:.4f}  AUPRC={best_mean_auprc:.4f}  CBI={best_mean_cbi:.3f}"
    )

    cbi_sel_per_p_auc = {}
    cbi_sel_per_p_auprc = {}
    cbi_sel_per_p_cbi = {}
    cbi_sel_per_p_ece = {}
    cbi_sel_mean_auc = float("nan")
    cbi_sel_mean_auprc = float("nan")
    cbi_sel_mean_cbi = float("nan")
    cbi_ckpt_path = os.path.join(args.output_dir, "best_model_by_cbi.pt")
    if os.path.exists(cbi_ckpt_path):
        log_main(env, f"Evaluating CBI-selected model on {eval_split} (K={args.test_bag_K})...")
        unwrap(model).load_state_dict(torch.load(cbi_ckpt_path, map_location=device))
        for p in args.val_p_list:
            result = bagged_evaluate_at_p(
                unwrap(model), dataset, eval_indices, dist_info,
                p_value=p, bag_K=args.test_bag_K,
                batch_size=args.batch_size, device=device,
                num_workers=args.num_workers,
                base_seed=args.seed + 10_000,
                amp_dtype=amp_dtype,
                distributed_sampler=env.is_distributed,
            )
            s = result["summary"]
            cbi_sel_per_p_auc[p]   = s["mean_auc_roc"]
            cbi_sel_per_p_auprc[p] = s["mean_auc_pr"]
            cbi_sel_per_p_cbi[p]   = s["mean_cbi"]
            cbi_sel_per_p_ece[p]   = s.get("mean_ece", float("nan"))
            log_main(env,
                f"[cbi-sel] p={p:.2f}  AUC={s['mean_auc_roc']:.4f}  "
                f"AUPRC={s['mean_auc_pr']:.4f}  CBI={s['mean_cbi']:.3f}  "
                f"ECE={cbi_sel_per_p_ece[p]:.4f}"
            )
        cbi_sel_mean_auc   = float(np.mean(list(cbi_sel_per_p_auc.values())))
        cbi_sel_mean_auprc = float(np.mean(list(cbi_sel_per_p_auprc.values())))
        cbi_sel_mean_cbi   = float(np.nanmean(list(cbi_sel_per_p_cbi.values())))
        log_main(env,
            f"[cbi-sel] mean: AUC={cbi_sel_mean_auc:.4f}  AUPRC={cbi_sel_mean_auprc:.4f}  "
            f"CBI={cbi_sel_mean_cbi:.3f}"
        )
        unwrap(model).load_state_dict(torch.load(
            os.path.join(args.output_dir, "best_model.pt"), map_location=device
        ))

    absmask_per_p_auc = {}
    absmask_per_p_auprc = {}
    absmask_per_p_cbi = {}
    absmask_per_p_brier = {}
    absmask_per_p_ece = {}
    absmask_per_p_q25 = {}
    absmask_per_p_q50 = {}
    absmask_per_p_q75 = {}
    absmask_mean_auc = float("nan")
    absmask_mean_auprc = float("nan")
    absmask_mean_cbi = float("nan")
    if not args.no_absence_mask_eval:
        from STEMLM_data import AbsenceMaskCollator
        log_main(env,
            f"Absence-mask eval on {eval_split} (mask all absences + p "
            f"presences, K={args.test_bag_K})..."
        )
        for p in args.absence_mask_p_list:
            if p == 1.0 and 1.0 in per_p_auc:
                absmask_per_p_auc[p]   = per_p_auc[p]
                absmask_per_p_auprc[p] = per_p_auprc[p]
                absmask_per_p_cbi[p]   = per_p_cbi[p]
                absmask_per_p_brier[p] = per_p_brier[p]
                absmask_per_p_ece[p]   = per_p_ece[p]
                absmask_per_p_q25[p]   = per_p_q25[p]
                absmask_per_p_q50[p]   = per_p_q50[p]
                absmask_per_p_q75[p]   = per_p_q75[p]
                log_main(env, f"absmask p={p:.2f}  (= uniform p=1.00, reused)")
                continue
            result = bagged_evaluate_at_p(
                unwrap(model), dataset, eval_indices, dist_info,
                p_value=p, bag_K=args.test_bag_K,
                batch_size=args.batch_size, device=device,
                num_workers=args.num_workers,
                base_seed=args.seed + 20_000,
                amp_dtype=amp_dtype,
                distributed_sampler=env.is_distributed,
                collator_cls=AbsenceMaskCollator,
            )
            s = result["summary"]
            absmask_per_p_auc[p]   = s["mean_auc_roc"]
            absmask_per_p_auprc[p] = s["mean_auc_pr"]
            absmask_per_p_cbi[p]   = s["mean_cbi"]
            absmask_per_p_brier[p] = s.get("mean_brier", float("nan"))
            absmask_per_p_ece[p]   = s.get("mean_ece", float("nan"))
            absmask_per_p_q25[p]   = s.get("auc_roc_q25", float("nan"))
            absmask_per_p_q50[p]   = s.get("auc_roc_q50", float("nan"))
            absmask_per_p_q75[p]   = s.get("auc_roc_q75", float("nan"))
            log_main(env,
                f"absmask p={p:.2f}  bag(K={args.test_bag_K}) "
                f"AUC={s['mean_auc_roc']:.4f}  "
                f"AUCq25/50/75={absmask_per_p_q25[p]:.3f}/{absmask_per_p_q50[p]:.3f}/{absmask_per_p_q75[p]:.3f}  "
                f"AUPRC={s['mean_auc_pr']:.4f}  "
                f"CBI={s['mean_cbi']:.3f}  "
                f"Brier={absmask_per_p_brier[p]:.4f}  ECE={absmask_per_p_ece[p]:.4f}  "
                f"(n={s['n_species']})"
            )
        if absmask_per_p_auc:
            absmask_mean_auc   = float(np.mean(list(absmask_per_p_auc.values())))
            absmask_mean_auprc = float(np.mean(list(absmask_per_p_auprc.values())))
            absmask_mean_cbi   = float(np.nanmean(list(absmask_per_p_cbi.values())))
            log_main(env,
                f"absmask mean over p={list(absmask_per_p_auc.keys())}: "
                f"AUC={absmask_mean_auc:.4f}  AUPRC={absmask_mean_auprc:.4f}  "
                f"CBI={absmask_mean_cbi:.3f}"
            )

    if env.is_main:
        import pandas as pd
        all_species = sorted({
            s for ps in per_p_per_species.values() for s in ps.get("auc_roc", {})
        })
        rows = []
        for sp in all_species:
            row = {"species": dataset.species_cols[sp]}
            for p in args.val_p_list:
                ps = per_p_per_species[p]
                row[f"auc_p{p:.2f}"]      = ps.get("auc_roc",     {}).get(sp, float("nan"))
                row[f"auprc_p{p:.2f}"]    = ps.get("auc_pr",      {}).get(sp, float("nan"))
                row[f"cbi_p{p:.2f}"]      = ps.get("cbi",         {}).get(sp, float("nan"))
            row["auc_mean"]   = float(np.nanmean([row[f"auc_p{p:.2f}"]   for p in args.val_p_list]))
            row["auprc_mean"] = float(np.nanmean([row[f"auprc_p{p:.2f}"] for p in args.val_p_list]))
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(args.output_dir, "per_species_auc.csv"), index=False)
        logger.info(f"Per-species metrics saved to {args.output_dir}/per_species_auc.csv")

        test_rows = []
        for p in args.val_p_list:
            test_rows.append({
                "mask_scheme": "uniform", "p": p,
                "auc": per_p_auc.get(p, float("nan")),
                "auc_q25": per_p_q25.get(p, float("nan")),
                "auc_q50": per_p_q50.get(p, float("nan")),
                "auc_q75": per_p_q75.get(p, float("nan")),
                "auprc": per_p_auprc.get(p, float("nan")),
                "cbi": per_p_cbi.get(p, float("nan")),
                "brier": per_p_brier.get(p, float("nan")),
                "ece": per_p_ece.get(p, float("nan")),
            })
        if not args.no_absence_mask_eval:
            for p in args.absence_mask_p_list:
                test_rows.append({
                    "mask_scheme": "absence_mask", "p": p,
                    "auc": absmask_per_p_auc.get(p, float("nan")),
                    "auc_q25": absmask_per_p_q25.get(p, float("nan")),
                    "auc_q50": absmask_per_p_q50.get(p, float("nan")),
                    "auc_q75": absmask_per_p_q75.get(p, float("nan")),
                    "auprc": absmask_per_p_auprc.get(p, float("nan")),
                    "cbi": absmask_per_p_cbi.get(p, float("nan")),
                    "brier": absmask_per_p_brier.get(p, float("nan")),
                    "ece": absmask_per_p_ece.get(p, float("nan")),
                })
        pd.DataFrame(test_rows).to_csv(
            os.path.join(args.output_dir, "test_results.csv"), index=False)
        logger.info(f"Test results saved to {args.output_dir}/test_results.csv")

        summary = {
            "ablation":           config.ablation,
            "num_params":         num_params,
            "best_val_auprc_mean": best_val_auprc_mean,
            "best_val_auc_mean":   best_val_auc_mean,
            "test_bag_K":         int(args.test_bag_K),
            "test_mean_auprc":    best_mean_auprc,
            "test_mean_auc":      best_mean_auc,
            "test_mean_cbi":      best_mean_cbi,
            "test_auprc_by_p":    {f"{p:.2f}": per_p_auprc[p] for p in per_p_auprc},
            "test_auc_by_p":      {f"{p:.2f}": per_p_auc[p]   for p in per_p_auc},
            "test_cbi_by_p":      {f"{p:.2f}": per_p_cbi[p]   for p in per_p_cbi},
            "test_brier_by_p":    {f"{p:.2f}": per_p_brier[p] for p in per_p_brier},
            "test_ece_by_p":      {f"{p:.2f}": per_p_ece[p]   for p in per_p_ece},
            "test_auc_q25_by_p":  {f"{p:.2f}": per_p_q25[p]   for p in per_p_q25},
            "test_auc_q50_by_p":  {f"{p:.2f}": per_p_q50[p]   for p in per_p_q50},
            "test_auc_q75_by_p":  {f"{p:.2f}": per_p_q75[p]   for p in per_p_q75},
            "eval_split":         eval_split,
            "num_species":        config.num_species,
            "num_epochs":         args.num_epochs,
            "seed":               args.seed,
            "world_size":         env.world_size,
            "mixed_precision":    args.mixed_precision,
            "loss_type":          args.loss_type,
            "focal_alpha":        args.focal_alpha if args.loss_type == "focal" else None,
            "focal_gamma":        args.focal_gamma if args.loss_type == "focal" else None,
            "best_val_cbi_mean":  best_val_cbi_mean if best_val_cbi_mean > -float("inf") else None,
            "test_by_selection": {
                "auc": {
                    "ckpt": "best_model.pt",
                    "test_mean_auc":  best_mean_auc,
                    "test_mean_cbi":  best_mean_cbi,
                },
                **({
                    "cbi": {
                        "ckpt": "best_model_by_cbi.pt",
                        "test_mean_auc":  cbi_sel_mean_auc,
                        "test_mean_auprc": cbi_sel_mean_auprc,
                        "test_mean_cbi":  cbi_sel_mean_cbi,
                        "test_auc_by_p":  {f"{p:.2f}": cbi_sel_per_p_auc[p] for p in cbi_sel_per_p_auc},
                        "test_cbi_by_p":  {f"{p:.2f}": cbi_sel_per_p_cbi[p] for p in cbi_sel_per_p_cbi},
                        "test_ece_by_p":  {f"{p:.2f}": cbi_sel_per_p_ece[p] for p in cbi_sel_per_p_ece},
                    }
                } if cbi_sel_per_p_auc else {})
            },
            "absmask_mean_auc":     absmask_mean_auc,
            "absmask_mean_auprc":   absmask_mean_auprc,
            "absmask_mean_cbi":     absmask_mean_cbi,
            "absmask_auc_by_p":     {f"{p:.2f}": absmask_per_p_auc[p]   for p in absmask_per_p_auc},
            "absmask_auprc_by_p":   {f"{p:.2f}": absmask_per_p_auprc[p] for p in absmask_per_p_auprc},
            "absmask_cbi_by_p":     {f"{p:.2f}": absmask_per_p_cbi[p]   for p in absmask_per_p_cbi},
            "absmask_brier_by_p":   {f"{p:.2f}": absmask_per_p_brier[p] for p in absmask_per_p_brier},
            "absmask_ece_by_p":     {f"{p:.2f}": absmask_per_p_ece[p]   for p in absmask_per_p_ece},
            "absmask_auc_q25_by_p": {f"{p:.2f}": absmask_per_p_q25[p]   for p in absmask_per_p_q25},
            "absmask_auc_q50_by_p": {f"{p:.2f}": absmask_per_p_q50[p]   for p in absmask_per_p_q50},
            "absmask_auc_q75_by_p": {f"{p:.2f}": absmask_per_p_q75[p]   for p in absmask_per_p_q75},
        }
        with open(os.path.join(args.output_dir, "ablation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
    if env.is_main:
        logger.info("Extracting species cooccurrence matrix...")
        unwrap(model).eval()
        cooccurrences = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= args.cooccurrence_extract_batches:
                    break
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = unwrap(model)(
                    input_ids=batch["input_ids"], source_ids=batch["source_ids"],
                    source_idx=batch["source_idx"],
                    target_site_idx=batch["target_site_idx"], env_data=batch["env_data"],
                    target_env=batch["target_env"],
                    labels=batch["labels"],
                    site_lats=dist_info["site_lats"],
                    site_lons=dist_info["site_lons"],
                    site_times=dist_info["site_times"],
                    euclidean=dist_info.get("euclidean", False),
                    output_attentions=True,
                )
                cooccurrences.append(extract_cooccurrence_matrix(output).cpu())

        cooccurrence_matrix = torch.cat(cooccurrences, dim=0).mean(dim=0).numpy()
        np.save(os.path.join(args.output_dir, "cooccurrence_matrix.npy"), cooccurrence_matrix)

        with open(os.path.join(args.output_dir, "species_names.json"), "w") as f:
            json.dump(dataset.species_cols, f)

        logger.info(f"Done. Output: {args.output_dir}")

    env.barrier()
    env.cleanup()


if __name__ == "__main__":
    main()
