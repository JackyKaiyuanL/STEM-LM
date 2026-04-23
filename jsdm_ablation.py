"""
Ablation study for STEM-LM cross-attention components.

Three ablation modes (plus the full model baseline):
    full        — both ST cross-attention and ecological cross-attention (default)
    no_st       — remove spatio-temporal cross-attention, keep ecological
    no_eco      — remove ecological cross-attention, keep spatio-temporal
    no_st_eco   — remove both cross-attentions (species self-attention + FFN only)

Architecture summary (full model):
    SpeciesSelfAttention → (STColAttention ⊕ EcoColAttention) → gate → FFN

When a cross-attention branch is ablated, its output is replaced with zeros so
that the gated residual has no contribution from that branch. The species
self-attention and FFN remain untouched in all ablation modes.

Usage:
    python jsdm_ablation.py data.csv --ablation no_st --output_dir ./ablation_no_st
    python jsdm_ablation.py data.csv --ablation no_eco --output_dir ./ablation_no_eco
    python jsdm_ablation.py data.csv --ablation no_st_eco --output_dir ./ablation_no_st_eco
    python jsdm_ablation.py data.csv --ablation full --output_dir ./ablation_full

All other flags from jsdm_train.py are supported.
"""

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.modeling_outputs import ModelOutput

from jsdm_data import create_dataloaders, save_splits, load_splits, build_val_loaders_fixed_p
from jsdm_train import (
    train_epoch,
    evaluate,
    compute_class_weights,
    move_dist_info_to_device,
    _parse_rate,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# Ablation modes
# =====================================================================

ABLATION_MODES = ("full", "no_st", "no_eco", "no_st_eco")


# =====================================================================
# Re-import unchanged components from jsdm_model
# =====================================================================
from jsdm_model import (
    JSDMConfig,
    FIREDistanceBias,
    _haversine_bt_n as _haversine_bt_n_abl,
    RMSNorm,
    TargetInput,
    EcoSourceModule,
    TargetEnvModule,
    SpeciesSelfAttention,
    SpeciesSelfOutput,
    SpeciesRowAttention,
    STCrossAttention,
    STCrossOutput,
    STColAttention,
    EcoCrossAttention,
    EcoCrossOutput,
    EcoColAttention,
    FeedForward,
    JSDMPredictionHead,
    JSDMEncoderOutput,
    JSDMOutput,
    extract_interaction_matrix,
)


# =====================================================================
# Ablation config — extends JSDMConfig with ablation mode
# =====================================================================

@dataclass
class AblationConfig(JSDMConfig):
    ablation: str = "full"  # one of ABLATION_MODES


# =====================================================================
# Ablation Attention — replaces JSDMAttention with switchable branches
# =====================================================================

class AblationAttention(nn.Module):
    """
    Drop-in replacement for JSDMAttention that can disable cross-attention
    branches based on the ablation mode.

    When a branch is disabled:
      - Its parameters are NOT instantiated (saves memory, prevents gradient flow)
      - Its contribution to the gated residual is zero

    When only one branch is active, the gate is not used — the active branch's
    output is scaled to use the full hidden_size projection. When both branches
    are disabled, the layer degenerates to species self-attention + FFN.
    """

    def __init__(self, config: AblationConfig):
        super().__init__()
        self.ablation = config.ablation
        self.use_st = config.ablation in ("full", "no_eco")
        self.use_eco = config.ablation in ("full", "no_st")

        # Species self-attention is always present
        self.row_attention = SpeciesRowAttention(config)

        # Pre-norm layers
        self.row_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Conditionally instantiate cross-attention branches
        if self.use_st:
            self.st_col_attention = STColAttention(config)

        if self.use_eco:
            self.eco_col_attention = EcoColAttention(config)

        # Gate only needed when both branches are active.
        # Content-conditional: shared H -> H/8 -> 1 MLP (see JSDMAttention).
        if self.use_st and self.use_eco:
            self.combine_gate = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 8),
                nn.SiLU(),
                nn.Linear(config.hidden_size // 8, 1),
            )

        # When only one branch is active, we scale by 2x to compensate for
        # the missing branch's contribution (since each branch projects from
        # hidden_size//2, the residual magnitude would be halved without this).
        self._single_branch_scale = 2.0 if (self.use_st != self.use_eco) else 1.0

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None,
        output_attentions=False,
    ):
        # 1. Pre-norm → species self-attention → residual
        row_output = self.row_attention(
            hidden_states=self.row_norm(hidden_states).transpose(-2, -3),
            output_attentions=output_attentions,
        )
        h = hidden_states + row_output[0].transpose(-2, -3)

        # 2. Cross-attention branches (parallel, gated)
        st_attn_out = None
        eco_attn_out = None

        if self.use_st or self.use_eco:
            h_normed = self.cross_norm(h)

        if self.use_st:
            st_output = self.st_col_attention(
                h_normed, st_source_embeddings, st_attention_mask, st_dist, output_attentions
            )
            st_attn_out = st_output[1] if (output_attentions and len(st_output) > 1) else None

        if self.use_eco:
            eco_output = self.eco_col_attention(
                h_normed, eco_embeddings, output_attentions
            )
            eco_attn_out = eco_output[1] if (output_attentions and len(eco_output) > 1) else None

        # 3. Combine via gate or single-branch scale
        if self.use_st and self.use_eco:
            # Full model: content-conditional gated combination
            gate = torch.sigmoid(self.combine_gate(h_normed))  # (B, S, T, 1)
            h = h + gate * st_output[0] + (1 - gate) * eco_output[0]
        elif self.use_st:
            # no_eco: only ST branch active
            h = h + self._single_branch_scale * st_output[0]
        elif self.use_eco:
            # no_st: only eco branch active
            h = h + self._single_branch_scale * eco_output[0]
        # else: no_st_eco — skip cross-attention entirely, h stays as-is

        out = (h,)
        if output_attentions:
            out = out + (
                row_output[1] if len(row_output) > 1 else None,
                st_attn_out,
                eco_attn_out,
            )
        return out


# =====================================================================
# Ablation Layer, Encoder, Model, ForMaskedPrediction
# =====================================================================

class AblationLayer(nn.Module):
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.attention = AblationAttention(config)
        self.ffn = FeedForward(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None,
        output_attentions=False,
    ):
        attn_outputs = self.attention(
            hidden_states, st_source_embeddings, eco_embeddings,
            st_attention_mask, st_dist, output_attentions,
        )
        h = attn_outputs[0]
        h = h + self.ffn(self.ffn_norm(h))
        return (h,) + attn_outputs[1:]


class AblationEncoder(nn.Module):
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [AblationLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None,
        output_attentions=False, output_hidden_states=False,
    ):
        all_hidden = () if output_hidden_states else None
        all_sp_attn = () if output_attentions else None
        all_st_attn = () if output_attentions else None
        all_eco_attn = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_sp_attn = all_sp_attn + (layer_outputs[1],)
                all_st_attn = all_st_attn + (layer_outputs[2],)
                all_eco_attn = all_eco_attn + (layer_outputs[3],)

        if output_hidden_states:
            all_hidden = all_hidden + (hidden_states,)

        return JSDMEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden,
            species_attentions=all_sp_attn,
            st_attentions=all_st_attn,
            eco_attentions=all_eco_attn,
        )


class AblationModel(nn.Module):
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        self.target_input = TargetInput(config)

        # Eco source module only needed if eco cross-attention is active
        self.use_eco = config.ablation in ("full", "no_st")
        if self.use_eco:
            self.eco_source_module = EcoSourceModule(config)
            self.target_env_module = TargetEnvModule(config)

        self.encoder = AblationEncoder(config)

    def forward(
        self,
        input_ids, source_ids, source_idx, target_site_idx,
        env_data, target_env,
        site_lats, site_lons, site_times,
        spatial_scale_km, temporal_scale_days, euclidean=False,
        output_attentions=False, output_hidden_states=False,
    ):
        # 1. Target input
        hidden_states = self.target_input(input_ids)

        # 2. Source site embedding (always needed for ST; still built for
        #    consistency even if ST is ablated — encoder will ignore it)
        species_idx = torch.arange(source_ids.size(1), device=source_ids.device)
        species_emb = self.target_input.species_embedding(species_idx)
        source_emb = self.target_input.embedding(source_ids) + species_emb[None, :, None, :]

        # 3. Ecological context
        if self.use_eco:
            eco_emb = self.eco_source_module(env_data)
            target_env_token = self.target_env_module(target_env)
            eco_emb = torch.cat([eco_emb, target_env_token], dim=1)
        else:
            eco_emb = None

        # 4. Target-to-source distances computed on-the-fly from site coords.
        dev = hidden_states.device
        lats = site_lats.to(dev); lons = site_lons.to(dev); times = site_times.to(dev)
        lat_t = lats[target_site_idx][:, :, None]
        lon_t = lons[target_site_idx][:, :, None]
        ti_t  = times[target_site_idx][:, :, None]
        lat_s = lats[source_idx][:, None, :]
        lon_s = lons[source_idx][:, None, :]
        ti_s  = times[source_idx][:, None, :]
        if euclidean:
            target_to_source_sp = torch.sqrt((lat_t - lat_s) ** 2 + (lon_t - lon_s) ** 2)
        else:
            target_to_source_sp = _haversine_bt_n_abl(lat_t, lon_t, lat_s, lon_s)
        target_to_source_tp = (ti_t - ti_s).abs()
        st_dist = torch.stack([target_to_source_sp, target_to_source_tp], dim=-1)

        # 5. Encode
        encoder_out = self.encoder(
            hidden_states, source_emb, eco_emb,
            st_attention_mask=None, st_dist=st_dist,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return encoder_out


class AblationForMaskedSpeciesPrediction(nn.Module):
    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config
        self.model = AblationModel(config)
        self.cls = JSDMPredictionHead(config)

    def forward(self, labels=None, loss_weight=None, output_attentions=False, **kwargs):
        encoder_out = self.model(output_attentions=output_attentions, **kwargs)
        logits = self.cls(encoder_out.last_hidden_state)

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                if loss_weight is None:
                    loss = F.binary_cross_entropy_with_logits(
                        logits[mask].float(), labels[mask].float()
                    )
                else:
                    per_el = F.binary_cross_entropy_with_logits(
                        logits[mask].float(), labels[mask].float(), reduction="none"
                    )
                    w = loss_weight.unsqueeze(-1).expand_as(labels)[mask]
                    loss = (per_el * w).sum() / w.sum()

        return JSDMOutput(
            loss=loss, logits=logits,
            hidden_states=encoder_out.hidden_states,
            species_attentions=encoder_out.species_attentions,
            st_attentions=encoder_out.st_attentions,
            eco_attentions=encoder_out.eco_attentions,
        )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation study for STEM-LM")
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--ablation", type=str, default="full", choices=ABLATION_MODES,
                        help="Ablation mode: full (baseline), no_st (remove ST cross-attn), "
                             "no_eco (remove eco cross-attn), no_st_eco (remove both)")
    parser.add_argument("--num_source_sites", type=int, default=64)
    parser.add_argument("--blind_percentile", type=float, default=2.0)
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
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./ablation_output")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--spatial_scale_km", type=float, default=None)
    parser.add_argument("--temporal_scale_days", type=float, default=None)
    parser.add_argument("--no_time", action="store_true")
    parser.add_argument("--euclidean_coords", action="store_true")
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
    parser.add_argument("--env_cols", nargs="+", default=None)
    parser.add_argument("--fold", choices=["random", "h3", "grid"], default="random")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Block resolution for spatial splits. For --fold h3, this is the H3 "
                             "resolution in [0, 15] (default 3). For --fold grid, this is the grid "
                             "side length (default 20). Not valid with --fold random.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Path to a splits.json (e.g. from a baseline training run). "
                             "When set, all ablation modes use the SAME train/val/test "
                             "split — required for a clean ablation comparison.")
    parser.add_argument("--no_save_splits", action="store_true")
    parser.add_argument("--val_p_list", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="Fixed mask rates for val AUC (deterministic per batch). "
                             "Mean AUC across these drives best_model.pt selection.")
    args = parser.parse_args()

    # --p is a convenience that sets both p_pres and p_abs to the same rate.
    if args.p is not None:
        args.p_pres = args.p
        args.p_abs = args.p

    # Split arg validation/defaults (ignored when loading saved splits).
    if args.splits_path is None:
        if args.fold == "random":
            if args.resolution is not None:
                raise ValueError("--resolution is not valid with --fold random.")
        elif args.fold == "h3":
            if args.resolution is None:
                args.resolution = 3
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

    # Tag output dir with ablation mode
    output_dir = args.output_dir
    if not output_dir.endswith(args.ablation):
        output_dir = os.path.join(output_dir, args.ablation)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=" * 60)
    logger.info(f"ABLATION MODE: {args.ablation}")
    logger.info(f"  ST cross-attention:  {'ON' if args.ablation in ('full', 'no_eco') else 'OFF'}")
    logger.info(f"  Eco cross-attention: {'ON' if args.ablation in ('full', 'no_st') else 'OFF'}")
    logger.info(f"  Output directory:    {output_dir}")
    logger.info(f"=" * 60)

    saved_splits = None
    if args.splits_path is not None:
        logger.info(f"Loading splits from {args.splits_path}")
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
        resolution=args.resolution,
        saved_splits=saved_splits,
    )

    if not args.no_save_splits:
        splits_out = os.path.join(output_dir, "splits.json")
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

    # Per-species loss weighting
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

    # Config
    config = AblationConfig(
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
        ablation=args.ablation,
    )

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    # Model
    model = AblationForMaskedSpeciesPrediction(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params:,} parameters, {config.num_species} species, ablation={config.ablation}")

    if args.gradient_checkpointing:
        model.model.encoder.gradient_checkpointing = True

    # Exclude norms, biases, and gate MLP from weight decay.
    no_decay_keys = ("norm", "bias", "combine_gate")
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

    # Pre-move pairwise distance matrices to device ONCE.
    dist_info = move_dist_info_to_device(dist_info, device)

    fixed_val_loaders = build_val_loaders_fixed_p(
        dataset, splits["val"], dist_info, args.val_p_list,
        batch_size=args.batch_size, num_workers=args.num_workers, base_seed=args.seed,
    )

    log_csv = os.path.join(output_dir, "training_log.csv")
    per_p_header = []
    for p, _ in fixed_val_loaders:
        per_p_header += [f"val_loss_p{p:.2f}", f"val_acc_p{p:.2f}", f"val_auc_p{p:.2f}"]
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            *per_p_header, "val_loss_mean", "val_acc_mean", "val_auc_mean",
            "lr", "elapsed_s", "ablation"
        ])

    best_val_auc_mean = -float("inf")
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, dist_info, epoch,
            loss_weight=loss_weight, max_grad_norm=args.max_grad_norm,
        )
        per_p_loss, per_p_acc, per_p_auc, per_p_nauc = [], [], [], []
        for p, loader in fixed_val_loaders:
            l, a, u, per_species = evaluate(model, loader, device, dist_info)
            per_p_loss.append(l)
            per_p_acc.append(a)
            per_p_auc.append(u)
            per_p_nauc.append(len(per_species))
        val_loss_mean = float(np.mean(per_p_loss)) if per_p_loss else float("nan")
        val_acc_mean  = float(np.mean(per_p_acc))  if per_p_acc  else float("nan")
        val_auc_mean  = float(np.mean(per_p_auc))  if per_p_auc  else float("nan")
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        per_p_str = " ".join(
            f"p{p:.2f}(loss={l:.3f},acc={a:.3f},auc={u:.3f},n_auc={n})"
            for (p, _), l, a, u, n in zip(
                fixed_val_loaders, per_p_loss, per_p_acc, per_p_auc, per_p_nauc
            )
        )
        logger.info(
            f"[{args.ablation}] Epoch {epoch}/{args.num_epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"{per_p_str} | mean auc={val_auc_mean:.4f} | {elapsed:.1f}s"
        )
        with open(log_csv, "a", newline="") as f:
            row = [epoch, f"{train_loss:.6f}", f"{train_acc:.6f}"]
            for l, a, u in zip(per_p_loss, per_p_acc, per_p_auc):
                row += [f"{l:.6f}", f"{a:.6f}", f"{u:.6f}"]
            row += [f"{val_loss_mean:.6f}", f"{val_acc_mean:.6f}", f"{val_auc_mean:.6f}",
                    f"{current_lr:.2e}", f"{elapsed:.1f}", args.ablation]
            csv.writer(f).writerow(row)
        if val_auc_mean > best_val_auc_mean:
            best_val_auc_mean = val_auc_mean
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info(f"  → Best model saved (val_auc_mean={val_auc_mean:.4f})")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss_mean": val_loss_mean,
                "val_auc_mean": val_auc_mean,
                "ablation": args.ablation,
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt"))

    # Final evaluation on test (or val) — per-p fixed mask rates, not random
    logger.info("Evaluating best model on fixed-p test set...")
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    )
    eval_split   = "test" if len(splits["test"]) > 0 else "val (no test set)"
    eval_indices = splits["test"] if len(splits["test"]) > 0 else splits["val"]
    eval_loaders = build_val_loaders_fixed_p(
        dataset, eval_indices, dist_info, args.val_p_list,
        batch_size=args.batch_size, num_workers=args.num_workers,
        base_seed=args.seed + 10_000,
    )
    per_p_auc   = {}
    per_p_per_species_auc = {}
    for p, loader in eval_loaders:
        _, _, m_auc, per_sp = evaluate(model, loader, device, dist_info)
        per_p_auc[p] = m_auc
        per_p_per_species_auc[p] = per_sp
        logger.info(f"[{args.ablation}] {eval_split} p={p:.2f}  mean AUC = {m_auc:.4f}  (species n={len(per_sp)})")
    test_mean_auc = float(np.mean(list(per_p_auc.values())))
    logger.info(f"[{args.ablation}] {eval_split} mean over p={list(per_p_auc.keys())}: {test_mean_auc:.4f}")

    import pandas as pd
    auc_rows = []
    all_species = sorted({s for per_sp in per_p_per_species_auc.values() for s in per_sp})
    for s in all_species:
        row = {"species": dataset.species_cols[s]}
        for p in per_p_auc:
            row[f"auc_p{p:.2f}"] = per_p_per_species_auc[p].get(s, float("nan"))
        row["auc_mean"] = float(np.nanmean([row[f"auc_p{p:.2f}"] for p in per_p_auc]))
        auc_rows.append(row)
    pd.DataFrame(auc_rows).to_csv(
        os.path.join(output_dir, f"per_species_auc_{args.ablation}.csv"), index=False
    )

    summary = {
        "ablation": args.ablation,
        "num_params": num_params,
        "best_val_auc_mean": best_val_auc_mean,
        "test_mean_auc": test_mean_auc,
        "test_auc_by_p": {f"{p:.2f}": per_p_auc[p] for p in per_p_auc},
        "eval_split": eval_split,
        "num_species": config.num_species,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
    }
    logger.info(f"[{args.ablation}] Param count: {num_params:,}")
    with open(os.path.join(output_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[{args.ablation}] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
