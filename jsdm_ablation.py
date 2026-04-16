"""
Ablation study for ST-JSDM cross-attention components.

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

from jsdm_data import create_dataloaders

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
    RMSNorm,
    TargetInput,
    EcoSourceModule,
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

        # Gate only needed when both branches are active
        if self.use_st and self.use_eco:
            self.combine_gate = nn.Parameter(torch.zeros(config.num_species))

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
            # Full model: gated combination
            gate = torch.sigmoid(self.combine_gate[None, :, None, None])
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

        self.encoder = AblationEncoder(config)

    def forward(
        self,
        input_ids, source_ids, source_idx, target_site_idx,
        env_data, spatial_dist_pairwise, temporal_dist_pairwise,
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
        else:
            # Placeholder — encoder won't use it
            eco_emb = None

        # 4. Target-to-source distances
        sp_pw = spatial_dist_pairwise.to(hidden_states.device)
        tp_pw = temporal_dist_pairwise.to(hidden_states.device)
        B, T = target_site_idx.shape
        N = source_idx.shape[1]

        target_sp_all = sp_pw[target_site_idx]
        target_tp_all = tp_pw[target_site_idx]
        src_idx_exp = source_idx[:, None, :].expand(-1, T, -1)
        target_to_source_sp = target_sp_all.gather(2, src_idx_exp)
        target_to_source_tp = target_tp_all.gather(2, src_idx_exp)
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
# Training & evaluation (mirrors jsdm_train.py)
# =====================================================================

def train_epoch(model, loader, optimizer, scheduler, device, dist_info, epoch,
                loss_weight=None, log_interval=50):
    model.train()
    total_loss, total_correct, total_masked, num_batches = 0, 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        B = batch["input_ids"].shape[0]
        w = loss_weight[None, :].expand(B, -1).to(device) if loss_weight is not None else None

        output = model(
            input_ids=batch["input_ids"],
            source_ids=batch["source_ids"],
            source_idx=batch["source_idx"],
            target_site_idx=batch["target_site_idx"],
            env_data=batch["env_data"],
            labels=batch["labels"],
            loss_weight=w,
            spatial_dist_pairwise=dist_info["spatial_dist_pairwise"],
            temporal_dist_pairwise=dist_info["temporal_dist_pairwise"],
        )

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
    species_preds = [[] for _ in range(S)]
    species_labels = [[] for _ in range(S)]

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
        total_loss += output.loss.item()
        num_batches += 1

        probs = torch.sigmoid(output.logits).squeeze(-1).cpu().numpy()
        labels = batch["labels"].squeeze(-1).cpu().numpy()
        mask = labels != -100
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
            total_masked += len(arr_l)
            if len(set(species_labels[s])) == 2 and not np.isnan(arr_p).any():
                auc_s = roc_auc_score(arr_l, arr_p)
                aucs.append(auc_s)
                per_species_aucs[s] = auc_s
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    acc = total_correct / max(total_masked, 1)

    return total_loss / max(num_batches, 1), acc, mean_auc, per_species_aucs


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablation study for ST-JSDM")
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
    parser.add_argument("--mlm_probability", type=float, default=0.15)
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
    parser.add_argument("--class_weighting", action="store_true")
    parser.add_argument("--class_weighting_beta", type=float, default=0.999)
    parser.add_argument("--env_cols", nargs="+", default=None)
    parser.add_argument("--fold", choices=["random", "h3", "grid"], default="random")
    parser.add_argument("--h3_resolution", type=int, default=2)
    parser.add_argument("--grid_cells", type=int, default=20)
    args = parser.parse_args()

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

    # Data
    train_loader, val_loader, test_loader, dataset, dist_info = create_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_source_sites=args.num_source_sites,
        mlm_probability=args.mlm_probability,
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
    )

    # Per-species loss weighting
    loss_weight = None
    if args.class_weighting:
        train_indices = train_loader.dataset.indices
        n_s = dataset.species_data[train_indices].sum(axis=0).clip(min=1)
        beta = args.class_weighting_beta
        effective_n = (1 - beta ** n_s) / (1 - beta)
        w = 1.0 / effective_n
        w = w / w.mean()
        loss_weight = torch.tensor(w, dtype=torch.float32)
        logger.info(f"Class weighting: weight range [{w.min():.3f}, {w.max():.3f}]")

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
        mlm_probability=args.mlm_probability,
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

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    log_csv = os.path.join(output_dir, "training_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc",
            "lr", "elapsed_s", "ablation"
        ])

    best_val_loss = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, dist_info, epoch, loss_weight
        )
        val_loss, val_acc, val_auc, _ = evaluate(model, val_loader, device, dist_info)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"[{args.ablation}] Epoch {epoch}/{args.num_epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} | {elapsed:.1f}s"
        )
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_auc:.6f}",
                f"{current_lr:.2e}", f"{elapsed:.1f}", args.ablation,
            ])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info(f"  → Best model saved")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "ablation": args.ablation,
            }, os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt"))

    # Final evaluation on test (or val)
    logger.info("Evaluating best model for per-species AUC...")
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    )
    eval_loader = test_loader if test_loader is not None else val_loader
    eval_split = "test" if test_loader is not None else "val (no test set)"
    _, _, best_mean_auc, per_species_aucs = evaluate(model, eval_loader, device, dist_info)
    logger.info(f"[{args.ablation}] Best model {eval_split} AUC (mean): {best_mean_auc:.4f}")

    auc_rows = [(dataset.species_cols[s], per_species_aucs[s]) for s in sorted(per_species_aucs)]
    import pandas as pd
    pd.DataFrame(auc_rows, columns=["species", "auc"]).to_csv(
        os.path.join(output_dir, f"per_species_auc_{args.ablation}.csv"), index=False
    )

    # Summary JSON for easy comparison across runs
    summary = {
        "ablation": args.ablation,
        "num_params": num_params,
        "best_val_loss": best_val_loss,
        "test_mean_auc": best_mean_auc,
        "eval_split": eval_split,
        "num_species": config.num_species,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
    }
    with open(os.path.join(output_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"[{args.ablation}] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
