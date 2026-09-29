import argparse
import csv
import itertools
import json
import logging
import os
import platform
import shlex
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from stemlm.data import (
    MIN_TRAIN_PRESENCES,
    AbsenceMaskCollator,
    build_val_loader_fixed_p,
    create_dataloaders,
    save_splits,
    seed_worker,
)
from stemlm.metric import (
    compute_per_species_metrics,
    evaluate_at_p,
    fit_temperature,
    gather_logits_at_p,
    move_dist_info_to_device,
    run_forward,
    summarize_per_species_metrics,
)
from stemlm.model import JSDMConfig, JSDMForMaskedSpeciesPrediction

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

    def all_gather_object(self, obj):
        if not self.is_distributed:
            return [obj]
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return gathered


def log_main(env: "DistEnv", msg: str, level: int = logging.INFO):
    if env.is_main:
        logger.log(level, msg)


def run_info(args, env, device):
    cuda = device.type == "cuda"
    if args.splits_path:
        with open(args.splits_path) as f:
            meta = json.load(f).get("meta", {})
        split = {"file": os.path.abspath(args.splits_path),
                 "resolution": meta.get("resolution"), "seed": meta.get("seed")}
    else:
        split = {"file": None, "resolution": args.resolution, "seed": args.seed,
                 "train_frac": args.train_frac, "test_frac": args.test_frac}
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "num_gpus": env.world_size if cuda else 0,
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 1) if cuda else None,
        "seed": args.seed,
        "split": split,
        "command": shlex.join(sys.argv),
        "args": {k: v for k, v in vars(args).items() if k != "func"},
    }


def _parse_rate(s):
    if isinstance(s, str) and (s == "unif" or s.startswith("unif:")):
        return s
    return float(s)


def train_epoch(model, loader, optimizer, scheduler, device, dist_info, epoch, env,
                log_interval=50, max_grad_norm=1.0, amp_dtype=None, grad_scaler=None,
                grad_accum_steps: int = 1, **loss_kw):
    if env.is_distributed:
        loader.sampler.set_epoch(epoch)

    model.train()
    # Accumulate on-GPU so the loop stays sync-free: .item() forces a CPU/GPU
    # sync that stalls the CPU from queuing the next step, idling the GPU. We
    # read these back only at the logging cadence and at epoch end.
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct_sum = torch.zeros((), device=device, dtype=torch.long)
    masked_sum = torch.zeros((), device=device, dtype=torch.long)
    num_batches = 0
    use_amp = amp_dtype is not None and device.type == "cuda"

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = run_forward(model, batch, dist_info, labels=batch["labels"], **loss_kw)
        loss = output.loss

        is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(loader))
        with model.no_sync() if env.is_distributed and not is_accum_step else nullcontext():
            loss_to_back = loss / grad_accum_steps
            (grad_scaler.scale(loss_to_back) if grad_scaler is not None else loss_to_back).backward()

        if is_accum_step:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_sum += loss.detach().double()
        num_batches += 1

        # Fixed-shape accuracy: no boolean-index gather (which would nonzero()
        # and sync on a data-dependent size) and no mask.any() guard. Compare
        # everywhere, then count only masked positions.
        mask = batch["labels"] != -100
        preds = output.logits > 0
        correct_sum += (preds & (batch["labels"] == 1) & mask).sum() \
            + (~preds & (batch["labels"] == 0) & mask).sum()
        masked_sum += mask.sum()

        if (batch_idx + 1) % log_interval == 0 and env.is_main:
            avg_loss = (loss_sum / max(num_batches, 1)).item()
            avg_acc = (correct_sum.double() / masked_sum.clamp(min=1)).item()
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # Single sync per epoch to materialise the accumulators as Python scalars.
    total_loss = loss_sum.item()
    total_correct = int(correct_sum.item())
    total_masked = int(masked_sum.item())

    if env.is_distributed:
        agg = torch.tensor(
            [total_loss, num_batches, total_correct, total_masked],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(agg, op=dist.ReduceOp.SUM)
        total_loss, num_batches, total_correct, total_masked = agg.tolist()

    return total_loss / max(num_batches, 1), total_correct / max(total_masked, 1)


@torch.no_grad()
def evaluate(model, loader, device, dist_info, env, amp_dtype=None, **loss_kw):
    model.eval()
    K = len(loader.collate_fn.collators)
    loss_sums = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(K)]
    batch_logits, batch_labels = [[] for _ in range(K)], [[] for _ in range(K)]
    num_batches = 0
    use_amp = amp_dtype is not None and device.type == "cuda"

    for batches in loader:
        for k, batch in enumerate(batches):
            batch = {key: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for key, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = run_forward(model, batch, dist_info, labels=batch["labels"], **loss_kw)
            loss_sums[k] += output.loss.detach().double()
            batch_logits[k].append(output.logits.float().squeeze(-1).double().cpu().numpy())
            batch_labels[k].append(batch["labels"].squeeze(-1).cpu().numpy())
        num_batches += 1

    results = []
    for loss_sum, z_parts, y_parts in zip(loss_sums, batch_logits, batch_labels, strict=True):
        total_loss, n = loss_sum.item(), num_batches
        logits, labels = np.concatenate(z_parts, axis=0), np.concatenate(y_parts, axis=0)
        if env.is_distributed:
            agg = torch.tensor([total_loss, n], dtype=torch.float64, device=device)
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            total_loss, n = agg.tolist()
            logits = np.concatenate(env.all_gather_object(logits), axis=0)
            labels = np.concatenate(env.all_gather_object(labels), axis=0)
        mask = labels != -100
        per_sp = compute_per_species_metrics(logits, labels)
        results.append((total_loss / n, float(((logits > 0) == labels)[mask].mean()),
                        summarize_per_species_metrics(per_sp), per_sp))
    return results


def add_train_args(parser):
    parser.add_argument("csv_path", type=str,
                        help="Wide table as .csv or .parquet (one column per species), or "
                             "(with --vocab_path) a sparse-parquet file or directory of parquet shards.")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="species_vocab.json; if set, csv_path is read as sparse parquet "
                             "(species_idx) via JSDMSparseDataset.")
    parser.add_argument("--num_source_sites", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--num_env_groups", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--p", type=_parse_rate, default="unif:0.0,1.0",
                        help="Per-row mask rate. Float in [0,1], or 'unif[:lo,hi]' "
                             "(Uniform[lo,hi] per row; bare 'unif' = 'unif:0.0,1.0').")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of data held out as test set for final AUC. "
                             "Val = 1 - train_frac - test_frac. Set to 0 to disable "
                             "(final AUC reported on val, same as early-stopping set).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=min(32, os.cpu_count() or 1),
                        help="DataLoader worker processes. Default min(32, cpu_count). "
                             "The per-sample source-pool KNN sampling is CPU-bound and "
                             "serial per worker, so 0 starves the GPU (measured ~7x slower "
                             "on an H100). Throughput plateaus at 32 on an H100 node "
                             "(16->32 is +17%%, 48 and 64 are flat). Set 0 only for debugging.")
    parser.add_argument("--output_dir", type=str, default="./STEMLM_output")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model in torch.compile to fuse kernels and "
                             "cut per-step launch overhead. First steps pay a "
                             "one-time compilation cost.")
    parser.add_argument("--compile_mode", default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode. 'reduce-overhead' uses CUDA graphs "
                             "to cut launch gaps; 'max-autotune' also autotunes kernels "
                             "(long warmup). Only used with --compile.")
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
                        help="Ignore the time column (purely spatial model). The temporal FIRE "
                             "bias is also disabled automatically when every time value is equal.")
    parser.add_argument("--train_exclusion", action=argparse.BooleanOptionalAction, default=True,
                        help="Per training target, remove candidate sources within a radius and, "
                             "independently, within a time window; each is 0 with probability 1/2 "
                             "and otherwise drawn log-uniformly between the nearest candidate and "
                             "the candidate pool's extent.")
    parser.add_argument("--eval_exclusion_km", type=float, nargs="+", default=None,
                        help="Also score the selected checkpoint on the test set with the training "
                             "sources within each of these radii (km) of the target removed, crossed "
                             "with every --eval_exclusion_days value; writes test_sweep.csv. "
                             "Validation, checkpoint selection and test_results.csv use no exclusion.")
    parser.add_argument("--eval_exclusion_days", type=float, nargs="+", default=[0.0, 1.0],
                        help="Time windows (days) crossed with --eval_exclusion_km for the test sweep "
                             "(default 0 1, so same-day sources are removed in one sweep row).")
    parser.add_argument("--min_train_presences", type=int, default=MIN_TRAIN_PRESENCES,
                        help="Keep the species with at least this many presences in the training "
                             "rows of the split. Set it at or above the --min_presences the table "
                             "was built with, so that test rows do not decide which species are kept.")
    parser.add_argument("--causal_context", action="store_true",
                        help="Sources must precede the target in time; time windows then count "
                             "days before the target only.")
    parser.add_argument("--source_cell_resolution", type=int, default=7,
                        help="Validation and test targets draw sources from the training rows and from "
                             "the other rows of their own split outside the target's H3 cell at this "
                             "resolution (default 7, about 1.4 km edge). The uniform test scheme is also "
                             "scored with training rows as the only sources.")
    parser.add_argument("--env_cols", nargs="+", default=None,
                        help="Explicit list of env column names. If not set, columns with 'env_' "
                             "prefix are used. Useful for datasets with non-prefixed env columns "
                             "(e.g. annualtemp, annualprec).")
    parser.add_argument("--resolution", type=int, default=2,
                        help="H3 resolution in [0, 15] of the spatial-block train/val/test split "
                             "(default 2 ≈ 183 km edge).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping max norm (default 1.0).")
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Path to a splits.json (written by a previous training run). "
                             "When set, overrides --resolution / "
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
    parser.add_argument("--per_species_env_rank", type=int, default=8,
                        help="Rank of the parallel per-species env head bolted in "
                             "alongside the shared env encoder. Reads raw target_env "
                             "via low-rank A∈(E,r)·B∈(r,S) (A zero-init, monotone safe) "
                             "+ per-species bias. Active in full / no_st ablations; "
                             "silent in no_env / no_st_env. 0 disables it.")
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="focal",
                        help="Loss function. 'focal' = sigmoid focal loss "
                             "(Lin et al. 2017; default, alpha=0.25, gamma=2.0 "
                             "RetinaNet defaults). 'bce' = sigmoid BCE.")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha (positive-class weight). Set <0 "
                             "to disable alpha-balancing. Ignored when "
                             "--loss_type=bce. Default 0.25 (RetinaNet).")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss focusing parameter. 0 reduces to "
                             "weighted BCE. Ignored when --loss_type=bce. "
                             "Default 2.0 (RetinaNet).")
    parser.add_argument("--no_absence_mask_eval", action="store_true",
                        help="Skip the absence-mask test block.")
    parser.add_argument("--absence_mask_p_list", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="Presence-mask rates for absence-mask eval.")
    parser.add_argument("--temperature_scaling", action=argparse.BooleanOptionalAction, default=True,
                        help="Fit a temperature T* on validation logits at p=1 for each "
                             "evaluated checkpoint; every test metric is computed on "
                             "logits / T*. T* is saved to temperature.json.")
    parser.set_defaults(func=run_train)


def run_train(args):
    started = time.monotonic()
    env = DistEnv()
    env.setup(backend="nccl")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = env.device()
    if env.is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    env.barrier()

    final_artifacts = [
        os.path.join(args.output_dir, "test_results.csv"),
        os.path.join(args.output_dir, "per_species_auc.csv"),
        os.path.join(args.output_dir, "species_names.json"),
    ]
    if all(os.path.exists(p) for p in final_artifacts):
        log_main(env, f"All final artifacts already exist in {args.output_dir}; skipping.")
        for p in final_artifacts:
            log_main(env, f"  {p}")
        return

    info = run_info(args, env, device)
    log_main(env, f"Run: {info['num_gpus']}x {info['gpu']} ({info['gpu_memory_gib']} GiB), "
                  f"seed {args.seed}, torch {info['torch']}, CUDA {info['cuda']}, {info['platform']}")
    split = info["split"]
    log_main(env, f"Split: file {split['file']} (H3 resolution {split['resolution']}, split seed {split['seed']})"
                  if split["file"] else
                  f"Split: generated with H3 resolution {split['resolution']}, split seed {split['seed']}, "
                  f"train fraction {split['train_frac']}, test fraction {split['test_frac']}")
    log_main(env, f"Command: {info['command']}")
    if env.is_main:
        with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    train_loader, dataset, dist_info, splits = create_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_source_sites=args.num_source_sites,
        p=args.p,
        train_frac=args.train_frac,
        test_frac=args.test_frac,
        num_workers=args.num_workers,
        seed=args.seed,
        env_cols=args.env_cols,
        no_time=args.no_time,
        train_exclusion=args.train_exclusion,
        causal_context=args.causal_context,
        resolution=args.resolution,
        splits_path=args.splits_path,
        vocab_path=args.vocab_path,
        min_train_presences=args.min_train_presences,
        source_cell_resolution=args.source_cell_resolution,
    )

    if env.is_distributed:
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
            worker_init_fn=seed_worker,
            persistent_workers=args.num_workers > 0,
        )
        log_main(env,
            f"Train loader sharded: per-rank batches={len(train_loader)}, "
            f"global batches/epoch={len(train_loader) * env.world_size}"
        )

    if args.splits_path is None and not args.no_save_splits and env.is_main:
        splits_out = os.path.join(args.output_dir, "splits.json")
        save_splits(
            splits_out, splits["train"], splits["val"], splits["test"],
            num_rows=len(dataset),
            meta={
                "fold":          "h3",
                "resolution":    args.resolution,
                "train_frac":    args.train_frac,
                "test_frac":     args.test_frac,
                "seed":          args.seed,
                "source":        None,
            },
        )
        logger.info(f"Splits saved to {splits_out}")
    if env.is_main:
        with open(os.path.join(args.output_dir, "env_stats.json"), "w") as f:
            json.dump({"env_cols": dataset.env_cols, "mean": dataset.env_mean.tolist()}, f, indent=2)
    env.barrier()

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

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        log_main(env, f"Model compiled with torch.compile (mode={args.compile_mode})")

    if env.is_distributed:
        model = DDP(
            model,
            device_ids=[env.local_rank],
            output_device=env.local_rank,
            find_unused_parameters=args.ablation != "full",
            gradient_as_bucket_view=True,
        )
        log_main(env, "Model wrapped with DistributedDataParallel")

    def unwrap(m):
        m = m.module if isinstance(m, DDP) else m
        return getattr(m, "_orig_mod", m)

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
        (nodecay_params if any(k in n for k in no_decay_keys) else decay_params).append(p)
    optimizer = AdamW(
        [
            {"params": decay_params,   "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        fused=device.type == "cuda",
    )

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    dist_info = move_dist_info_to_device(dist_info, device)

    val_loader = build_val_loader_fixed_p(
        dataset, splits["val"], args.val_p_list,
        batch_size=args.batch_size, num_workers=args.num_workers, base_seed=args.seed,
    )

    log_csv = os.path.join(args.output_dir, "training_log.csv")
    per_p_header = []
    for p in args.val_p_list:
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
            max_grad_norm=args.max_grad_norm,
            amp_dtype=amp_dtype, grad_scaler=grad_scaler,
            grad_accum_steps=args.grad_accum_steps,
            env=env,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        )
        per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_cbi, per_p_nauc = [], [], [], [], [], []
        for loss_v, acc_v, summary, _per_sp in evaluate(
            model, val_loader, device, dist_info, amp_dtype=amp_dtype, env=env,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        ):
            per_p_loss.append(loss_v)
            per_p_acc.append(acc_v)
            per_p_auc.append(summary["mean_auc_roc"])
            per_p_auprc.append(summary["mean_auc_pr"])
            per_p_cbi.append(summary["mean_cbi"])
            per_p_nauc.append(summary["n_species"])
        val_loss_mean  = float(np.mean(per_p_loss))
        val_acc_mean   = float(np.mean(per_p_acc))
        val_auc_mean   = float(np.mean(per_p_auc))
        val_auprc_mean = float(np.mean(per_p_auprc))
        val_cbi_mean   = float(np.nanmean(per_p_cbi))
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        if env.is_main:
            per_p_str = " ".join(
                f"p{p:.2f}(loss={lo:.3f},acc={a:.3f},auc={u:.3f},auprc={ap:.3f},cbi={c:.3f},n={n})"
                for p, lo, a, u, ap, c, n in zip(
                    args.val_p_list, per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_cbi, per_p_nauc,
                    strict=True,
                )
            )
            logger.info(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"{per_p_str} | mean auc={val_auc_mean:.4f} auprc={val_auprc_mean:.4f} cbi={val_cbi_mean:.4f} | {elapsed:.1f}s"
            )
            with open(log_csv, "a", newline="") as f:
                row = [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}"]
                for lo, a, u, ap, c in zip(per_p_loss, per_p_acc, per_p_auc, per_p_auprc, per_p_cbi, strict=False):
                    row += [f"{lo:.6f}", f"{a:.6f}", f"{u:.6f}", f"{ap:.6f}", f"{c:.6f}"]
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
    log_main(env, f"Evaluating best model on fixed-p {eval_split} set...")

    best_state = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    unwrap(model).load_state_dict(best_state)

    def fit_checkpoint_temperature():
        if not args.temperature_scaling:
            return 1.0
        val_logits, val_labels = gather_logits_at_p(
            unwrap(model), dataset, np.array(splits["val"]), dist_info,
            p_value=1.0, batch_size=args.batch_size, device=device,
            num_workers=args.num_workers, base_seed=args.seed + 30_000,
            amp_dtype=amp_dtype, distributed_sampler=env.is_distributed,
        )
        T = fit_temperature(val_logits, val_labels)
        log_main(env, f"[temperature_scaling] T* = {T:.4f}; test metrics use logits / T*")
        return T

    T_star = fit_checkpoint_temperature()

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
        result = evaluate_at_p(
            unwrap(model), dataset, eval_indices, dist_info,
            p_value=p,
            batch_size=args.batch_size, device=device,
            num_workers=args.num_workers,
            base_seed=args.seed + 10_000,
            amp_dtype=amp_dtype,
            distributed_sampler=env.is_distributed,
            temperature=T_star,
        )
        s = result["summary"]
        per_p_auc[p]    = s["mean_auc_roc"]
        per_p_auprc[p]  = s["mean_auc_pr"]
        per_p_cbi[p]    = s["mean_cbi"]
        per_p_brier[p]  = s["mean_brier"]
        per_p_ece[p]    = s["mean_ece"]
        per_p_q25[p]    = s["auc_roc_q25"]
        per_p_q50[p]    = s["auc_roc_q50"]
        per_p_q75[p]    = s["auc_roc_q75"]
        per_p_per_species[p] = result["per_species"]
        log_main(env,
            f"{eval_split} p={p:.2f}  "
            f"AUC={s['mean_auc_roc']:.4f}  "
            f"AUCq25/50/75={per_p_q25[p]:.3f}/{per_p_q50[p]:.3f}/{per_p_q75[p]:.3f}  "
            f"AUPRC={s['mean_auc_pr']:.4f}  "
            f"CBI={s['mean_cbi']:.3f}  "
            f"Brier={per_p_brier[p]:.4f}  ECE={per_p_ece[p]:.4f}  "
            f"(n={s['n_species']})"
        )
    best_mean_auc   = float(np.mean(list(per_p_auc.values())))
    best_mean_auprc = float(np.mean(list(per_p_auprc.values())))
    best_mean_cbi   = float(np.nanmean(list(per_p_cbi.values())))
    log_main(env,
        f"{eval_split} mean over p={list(per_p_auc.keys())}: "
        f"AUC={best_mean_auc:.4f}  AUPRC={best_mean_auprc:.4f}  CBI={best_mean_cbi:.3f}"
    )

    heldout_ids, dataset.heldout_split = dataset.heldout_split, None
    other_per_p = {}
    for p in args.val_p_list:
        s = evaluate_at_p(
            unwrap(model), dataset, eval_indices, dist_info,
            p_value=p,
            batch_size=args.batch_size, device=device,
            num_workers=args.num_workers,
            base_seed=args.seed + 10_000,
            amp_dtype=amp_dtype,
            distributed_sampler=env.is_distributed,
            temperature=T_star,
        )["summary"]
        other_per_p[p] = s
        log_main(env, f"{eval_split} sources=train p={p:.2f}  AUC={s['mean_auc_roc']:.4f}  "
                      f"AUPRC={s['mean_auc_pr']:.4f}  CBI={s['mean_cbi']:.3f}  (n={s['n_species']})")
    dataset.heldout_split = heldout_ids

    cbi_sel_per_p_auc = {}
    cbi_sel_per_p_auprc = {}
    cbi_sel_per_p_cbi = {}
    cbi_sel_per_p_ece = {}
    cbi_sel_mean_auc = float("nan")
    cbi_sel_mean_auprc = float("nan")
    cbi_sel_mean_cbi = float("nan")
    cbi_ckpt_path = os.path.join(args.output_dir, "best_model_by_cbi.pt")
    if os.path.exists(cbi_ckpt_path):
        log_main(env, f"Evaluating CBI-selected model on {eval_split}...")
        unwrap(model).load_state_dict(torch.load(cbi_ckpt_path, map_location=device))
        T_cbi = fit_checkpoint_temperature()
        for p in args.val_p_list:
            result = evaluate_at_p(
                unwrap(model), dataset, eval_indices, dist_info,
                p_value=p,
                batch_size=args.batch_size, device=device,
                num_workers=args.num_workers,
                base_seed=args.seed + 10_000,
                amp_dtype=amp_dtype,
                distributed_sampler=env.is_distributed,
                temperature=T_cbi,
            )
            s = result["summary"]
            cbi_sel_per_p_auc[p]   = s["mean_auc_roc"]
            cbi_sel_per_p_auprc[p] = s["mean_auc_pr"]
            cbi_sel_per_p_cbi[p]   = s["mean_cbi"]
            cbi_sel_per_p_ece[p]   = s["mean_ece"]
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
        log_main(env,
            f"Absence-mask eval on {eval_split} (mask all absences + p presences)..."
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
            result = evaluate_at_p(
                unwrap(model), dataset, eval_indices, dist_info,
                p_value=p,
                batch_size=args.batch_size, device=device,
                num_workers=args.num_workers,
                base_seed=args.seed + 20_000,
                amp_dtype=amp_dtype,
                distributed_sampler=env.is_distributed,
                collator_cls=AbsenceMaskCollator,
                temperature=T_star,
            )
            s = result["summary"]
            absmask_per_p_auc[p]   = s["mean_auc_roc"]
            absmask_per_p_auprc[p] = s["mean_auc_pr"]
            absmask_per_p_cbi[p]   = s["mean_cbi"]
            absmask_per_p_brier[p] = s["mean_brier"]
            absmask_per_p_ece[p]   = s["mean_ece"]
            absmask_per_p_q25[p]   = s["auc_roc_q25"]
            absmask_per_p_q50[p]   = s["auc_roc_q50"]
            absmask_per_p_q75[p]   = s["auc_roc_q75"]
            log_main(env,
                f"absmask p={p:.2f}  "
                f"AUC={s['mean_auc_roc']:.4f}  "
                f"AUCq25/50/75={absmask_per_p_q25[p]:.3f}/{absmask_per_p_q50[p]:.3f}/{absmask_per_p_q75[p]:.3f}  "
                f"AUPRC={s['mean_auc_pr']:.4f}  "
                f"CBI={s['mean_cbi']:.3f}  "
                f"Brier={absmask_per_p_brier[p]:.4f}  ECE={absmask_per_p_ece[p]:.4f}  "
                f"(n={s['n_species']})"
            )
        absmask_mean_auc   = float(np.mean(list(absmask_per_p_auc.values())))
        absmask_mean_auprc = float(np.mean(list(absmask_per_p_auprc.values())))
        absmask_mean_cbi   = float(np.nanmean(list(absmask_per_p_cbi.values())))
        log_main(env,
            f"absmask mean over p={list(absmask_per_p_auc.keys())}: "
            f"AUC={absmask_mean_auc:.4f}  AUPRC={absmask_mean_auprc:.4f}  "
            f"CBI={absmask_mean_cbi:.3f}"
        )

    sweep_rows = []
    if args.eval_exclusion_km or args.eval_exclusion_days:
        for r, tau in itertools.product(args.eval_exclusion_km or [0.0], args.eval_exclusion_days or [0.0]):
            dataset.eval_exclusion_km, dataset.eval_exclusion_days = r, tau
            for p in args.val_p_list:
                s = evaluate_at_p(
                    unwrap(model), dataset, eval_indices, dist_info,
                    p_value=p,
                    batch_size=args.batch_size, device=device,
                    num_workers=args.num_workers,
                    base_seed=args.seed + 10_000,
                    amp_dtype=amp_dtype,
                    distributed_sampler=env.is_distributed,
                    temperature=T_star,
                )["summary"]
                log_main(env, f"sweep r={r:g} km tau={tau:g} d p={p:.2f}  AUC={s['mean_auc_roc']:.4f}  "
                              f"AUPRC={s['mean_auc_pr']:.4f}  CBI={s['mean_cbi']:.3f}  (n={s['n_species']})")
                sweep_rows.append({
                    "eval_exclusion_km": r, "eval_exclusion_days": tau, "p": p,
                    "auc": s["mean_auc_roc"], "auc_q25": s["auc_roc_q25"], "auc_q50": s["auc_roc_q50"],
                    "auc_q75": s["auc_roc_q75"], "auprc": s["mean_auc_pr"], "cbi": s["mean_cbi"],
                    "brier": s["mean_brier"], "ece": s["mean_ece"], "n_species": s["n_species"],
                })

    if args.temperature_scaling and env.is_main:
        with open(os.path.join(args.output_dir, "temperature.json"), "w") as f:
            json.dump({"T_star": float(T_star), "fitted_at_p": 1.0, "fitted_on_split": "val"},
                      f, indent=2)
        log_main(env, f"[temperature_scaling] saved -> {args.output_dir}/temperature.json")

    if env.is_main:
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
        for p, s in other_per_p.items():
            test_rows.append({
                "mask_scheme": "uniform", "sources": "train", "p": p,
                "auc": s["mean_auc_roc"], "auc_q25": s["auc_roc_q25"], "auc_q50": s["auc_roc_q50"],
                "auc_q75": s["auc_roc_q75"], "auprc": s["mean_auc_pr"], "cbi": s["mean_cbi"],
                "brier": s["mean_brier"], "ece": s["mean_ece"],
            })
        for p in args.val_p_list:
            test_rows.append({
                "mask_scheme": "uniform", "sources": "heldout", "p": p,
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
                    "mask_scheme": "absence_mask", "sources": "heldout", "p": p,
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
        if sweep_rows:
            pd.DataFrame(sweep_rows).to_csv(os.path.join(args.output_dir, "test_sweep.csv"), index=False)
            logger.info(f"Exclusion sweep saved to {args.output_dir}/test_sweep.csv")

        summary = {
            "ablation":           config.ablation,
            "num_params":         num_params,
            "best_val_auprc_mean": best_val_auprc_mean,
            "best_val_auc_mean":   best_val_auc_mean,
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
            "temperature":        T_star,
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
        with open(os.path.join(args.output_dir, "species_names.json"), "w") as f:
            json.dump(dataset.species_cols, f)
        wall = time.monotonic() - started
        info.update(ended_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    wall_seconds=round(wall, 1),
                    wall_hm=f"{int(wall // 3600)}:{int(wall % 3600 // 60):02d}")
        with open(os.path.join(args.output_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"Done. Wall time {info['wall_hm']} (h:mm) on {info['num_gpus']}x {info['gpu']}. "
                    f"Output: {args.output_dir}")

    env.barrier()
    env.cleanup()
