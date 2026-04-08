"""
Spatial-Temporal Joint Species Distribution Model (ST-JSDM)

Architecture directly mirrors GPN-star with corrected axis mapping:

    GPN-star                          →  ST-JSDM
    ──────────────────────────────────────────────────────────
    Nucleotide (A/C/G/T/-)            →  Species state (0=absent, 1=present, ?=mask)
    L genomic positions (sequence)     →  S species (the "sequence" to predict)
    N species in MSA (source)          →  N source site-time observations
    Target species (e.g. human)        →  Target site-time (the row to predict)
    Source species (other animals)     →  Source sites (other rows in CSV)
    Phylogenetic clades                →  Spatial-temporal clusters of sites
    Within-clade evolutionary dist     →  Within-cluster spatial-temporal dist
    Between-clade evolutionary dist    →  Between-cluster spatial-temporal dist
    Clade masking (don't attend own)   →  Cluster masking (don't attend own)

    Row self-attention (along L)       →  Self-attention along S species
      = positions attend to positions  =  species attend to species (INTERACTION TERM)

    Col cross-attention (along N)      →  Cross-attn 1: spatial-temporal context
      = target attends to clades       =  target site attends to site clusters

    (new, no GPN-star analog)          →  Cross-attn 2: ecological context
                                       =  target site attends to env variable groups

Data layout:
    CSV columns: time, latitude, longitude, env1, env2, ..., species1, species2, ...
    Each row: one site-time observation with environmental covariates + species 0/1

Training (mirrors GPN-star exactly):
    1. Sample a batch of target site-time rows
    2. For each target, sample source sites from the dataset (like sampling MSA species)
    3. Group source sites into spatial-temporal clusters (like clades)
    4. Pool within clusters using distance-weighted attention (like within-clade pooling)
    5. Mask 15% of species in the target row
    6. Self-attention along species axis to capture interactions
    7. Cross-attention to cluster-pooled source site embeddings (ST context)
    8. Cross-attention to ecological variable group embeddings (env context)
    9. Predict masked species (BCE loss, like CE loss on masked nucleotides)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JSDMConfig:
    """Configuration for ST-JSDM."""
    # Species
    num_species: int = 100          # S: number of species columns
    # No vocab_size needed: species states are raw floats (0.0, 1.0, mask_value)
    # mask_value is a learned parameter in TargetInput
    mask_value_init: float = -1.0   # initial value for learned mask token

    # Source sites (analogous to MSA species count)
    num_source_sites: int = 64      # N: number of source site-time obs per example
    num_target_sites: int = 1       # T: number of target sites (like target species)

    # Spatial-temporal clustering (analogous to phylogenetic clades)
    # cluster_dict, within/between distances are computed from data at init
    max_spatial_dist: float = 180.0
    max_temporal_dist: float = 365.0

    # Ecological variables
    num_env_vars: int = 10          # number of environmental covariates
    num_env_groups: int = 3         # grouping of env vars (e.g. climate, soil, topo)

    # Architecture
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_hidden_layers: int = 4
    intermediate_size: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6

    # Positional encoding
    fire_hidden_size: int = 32

    # Training
    mlm_probability: float = 0.15


# =============================================================================
# FIRE Distance Bias (same as GPN-star FIRETimeBias)
# =============================================================================

class FIREDistanceBias(nn.Module):
    """
    Learned distance bias. Identical to GPN-star's FIRETimeBias.
    In GPN-star: encodes evolutionary distance.
    Here: encodes spatial distance, temporal distance, or environmental distance.
    """

    def __init__(self, max_dist: float, fire_hidden_size: int = 32):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(100.0))
        self.mlp = nn.Sequential(
            nn.Linear(1, fire_hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(fire_hidden_size, 1, bias=False),
        )
        self.max_dist = max_dist

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: (...) raw distances
        Returns:
            bias: (...) learned bias values
        """
        dist = dist.unsqueeze(-1).float()
        c = self.c.clamp(min=0)
        dist = torch.log(c * dist + 1) / torch.log(c * self.max_dist + 1)
        return self.mlp(dist).squeeze(-1)


def _haversine_pairwise(lat_deg: np.ndarray, lon_deg: np.ndarray) -> torch.Tensor:
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    lat1 = lat[:, None]
    lat2 = lat[None, :]
    dlat = lat1 - lat2
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return torch.tensor(6371.0 * c, dtype=torch.float32)


def _resolve_scale(value: Optional[float], fallback: float, name: str) -> float:
    if value is None:
        if fallback <= 0:
            raise ValueError(f"Cannot auto-scale {name}: max pairwise distance is {fallback}")
        return float(fallback)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return float(value)


# =============================================================================
# RMSNorm — faster and simpler than LayerNorm; standard in modern transformers
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# =============================================================================
# Target Input
#
# GPN-star: nucleotides are categorical → nn.Embedding(6, H) → (B, L, T, H)
# Here: species states are categorical with 3 tokens → nn.Embedding(3, H) → (B, S, T, H)
#   0 = absent, 1 = present, 2 = mask
#
# Using a 3-token embedding instead of nn.Linear(1, H) gives each state
# an unrestricted H-dim vector, rather than 3 collinear points on a 1-D line.
# =============================================================================

class TargetInput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        # 3 discrete tokens: 0=absent, 1=present, 2=mask
        self.embedding = nn.Embedding(3, config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S, T) long tensor — 0=absent, 1=present, 2=mask
        Returns:
            (B, S, T, H)
        """
        return self.embedding(input_ids)  # (B, S, T, H)


# =============================================================================
# Source Module (analogous to GPNStarSourceModule)
#
# GPN-star: takes source_ids (B, L, N) = nucleotides at L positions for N species
#   → groups species into clades
#   → attention-pools within each clade, biased by within-clade evolutionary dist
#   → returns (B, L, C, H) = clade-pooled embeddings at each position
#
# Here: takes source_ids (B, S, N) = species states at S species for N source sites
#   → groups source sites into spatial-temporal clusters
#   → attention-pools within each cluster, biased by within-cluster ST distance
#   → returns (B, S, C_st, H) = cluster-pooled embeddings at each species
# =============================================================================

class STSourceModule(nn.Module):
    """
    Vectorized spatial-temporal source module.

    GPN-star: source_ids (B,L,N) → group by clade → pool → (B,L,C,H)
    Here: source_ids (B,S,N) → group by ST cluster → pool → (B,S,C_st,H)

    Fully vectorized: for each cluster, a masked softmax over the N sampled source sites
    ensures only member sites contribute. Empty clusters produce zero embeddings.
    Memory: O(B·S·N·A·C) for the masked logit tensor.
    """

    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads // 2
        self.head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_size

        # Source sites remain float {0, 1, mask_value} — projected from 1-dim input
        self.attention_weights = nn.Linear(1, self.num_heads)
        self.value = nn.Linear(1, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.ffn = nn.Linear(self.all_head_size, config.hidden_size)

        self.fire_spatial = FIREDistanceBias(config.max_spatial_dist, config.fire_hidden_size)
        self.fire_temporal = FIREDistanceBias(config.max_temporal_dist, config.fire_hidden_size)

    def forward(
        self,
        source_ids: torch.Tensor,                # (B, S, N)
        source_cluster_labels: torch.Tensor,      # (B, N) long
        in_cluster_spatial_dist: torch.Tensor,   # (B, N)
        in_cluster_temporal_dist: torch.Tensor,  # (B, N)
        num_clusters: int,
    ) -> torch.Tensor:
        B, S, N = source_ids.shape
        C = num_clusters

        # Within-cluster distance bias: (B, N)
        in_cluster_bias = (
            self.fire_spatial(in_cluster_spatial_dist)
            + self.fire_temporal(in_cluster_temporal_dist)
        )

        x = source_ids.unsqueeze(-1).float()                          # (B, S, N, 1)
        attn_logits = self.attention_weights(x)                        # (B, S, N, A)
        v = self.value(x).view(B, S, N, self.num_heads, self.head_size)  # (B, S, N, A, D)

        # Add distance bias, broadcast over S and A dims
        attn_logits = attn_logits + in_cluster_bias[:, None, :, None]  # (B, S, N, A)

        # Cluster membership mask: 1.0 if site n belongs to cluster c
        member = F.one_hot(source_cluster_labels, num_classes=C).float()  # (B, N, C)
        # Non-member sites get a large negative penalty: (B, 1, N, 1, C)
        non_member_mask = (1.0 - member[:, None, :, None, :]) * -1e9

        # Masked logits per cluster, then softmax over N within each cluster
        masked_logits = attn_logits.unsqueeze(-1) + non_member_mask   # (B, S, N, A, C)
        # nan_to_num: empty clusters (all -inf) produce nan from softmax → replace with 0
        attn_probs = torch.nan_to_num(
            F.softmax(masked_logits, dim=2), nan=0.0
        )  # (B, S, N, A, C)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum: (B, S, A, C, N) @ (B, S, A, N, D) → (B, S, A, C, D)
        pooled = torch.matmul(
            attn_probs.permute(0, 1, 3, 4, 2),   # (B, S, A, C, N)
            v.permute(0, 1, 3, 2, 4),             # (B, S, A, N, D)
        )  # (B, S, A, C, D)
        pooled = pooled.permute(0, 1, 3, 2, 4).reshape(B, S, C, self.all_head_size)

        result = self.ffn(pooled)  # (B, S, C, H)

        # Zero out embeddings for clusters with no sampled sources in this batch
        has_member = member.any(dim=1)  # (B, C)
        return result * has_member[:, None, :, None].float()


# =============================================================================
# Ecological Source Module (new, parallel to ST source module)
# =============================================================================

class EcoSourceModule(nn.Module):
    """
    Encodes environmental covariates from source sites, pooled into groups.
    Provides ecological context for cross-attention 2.
    """

    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_env_groups = config.num_env_groups
        self.proj = nn.Linear(config.num_env_vars, config.hidden_size)
        self.group_query = nn.Parameter(
            torch.randn(config.num_env_groups, config.hidden_size) * 0.02
        )
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, env_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            env_data: (B, N, E) environmental covariates at N source sites
        Returns:
            eco_embeddings: (B, C_eco, H)
        """
        B = env_data.size(0)
        site_emb = self.proj(env_data)  # (B, N, H)
        k = self.key_proj(site_emb)
        v = self.value_proj(site_emb)
        q = self.group_query.unsqueeze(0).expand(B, -1, -1)  # (B, C_eco, H)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, v)
        return self.layer_norm(self.out_proj(pooled))


# =============================================================================
# Row Self-Attention along species axis
# Mirrors GPNStarRowSelfAttention exactly.
# This is the CORE — species attend to each other = interaction term.
# =============================================================================

class SpeciesSelfAttention(nn.Module):
    """
    GPN-star detail: Q,K from target species only (index :1), V from all.
    Here: Q,K from target site only, V from all target sites.
    """

    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.rotary_value = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def forward(self, hidden_states, sinusoidal_pos=None, output_attentions=False):
        """
        Args:
            hidden_states: (B, T, S, H) transposed before calling
        """
        query_layer = self.transpose_for_scores(
            self.query(hidden_states[:, :1, ...])
        )  # (B, 1, A, S, D)
        key_layer = self.transpose_for_scores(
            self.key(hidden_states[:, :1, ...])
        )  # (B, 1, A, S, D)
        value_layer = self.transpose_for_scores(
            self.value(hidden_states)
        )  # (B, T, A, S, D)

        if sinusoidal_pos is not None:
            query_layer, key_layer = self._apply_rotary(
                sinusoidal_pos, query_layer, key_layer
            )

        if output_attentions:
            attn_scores = torch.matmul(
                query_layer, key_layer.transpose(-1, -2)
            ) / math.sqrt(self.attention_head_size)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            context = torch.matmul(attn_probs, value_layer)
            context = context.transpose(-2, -3).contiguous()
            new_shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(*new_shape)
            return context, attn_probs
        else:
            context = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
                scale=1.0 / math.sqrt(self.attention_head_size),
            )
            context = context.transpose(-2, -3).contiguous()
            new_shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(*new_shape)
            return (context,)

    @staticmethod
    def _apply_rotary(sinusoidal_pos, query_layer, key_layer):
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        rotate_half_q = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_q * sin_pos
        rotate_half_k = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_k * sin_pos
        return query_layer, key_layer


class SpeciesSelfOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class SpeciesRowAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.self_attn = SpeciesSelfAttention(config)
        self.output = SpeciesSelfOutput(config)

    def forward(self, hidden_states, sinusoidal_pos=None, output_attentions=False):
        self_outputs = self.self_attn(hidden_states, sinusoidal_pos, output_attentions)
        output = (self.output(self_outputs[0]),)  # project+dropout only; residual in JSDMAttention
        if output_attentions:
            output = output + (self_outputs[1],)
        return output


# =============================================================================
# ST Cross-Attention (analogous to GPNStarColCrossAttention)
# =============================================================================

class STCrossAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def forward(
        self, hidden_states, source_embeddings,
        attention_mask=None, st_dist_bias=None, output_attentions=False,
    ):
        """
        Args:
            hidden_states: (B, S, T, H)
            source_embeddings: (B, S, C_st, H)
            attention_mask: (B, 1, 1, T, C_st) with -inf for own cluster
            st_dist_bias: (B, S, 1, T, C_st) per-species FIRE distance bias
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(source_embeddings))
        value_layer = self.transpose_for_scores(self.value(source_embeddings))

        combined_mask = None
        if attention_mask is not None and st_dist_bias is not None:
            combined_mask = attention_mask + st_dist_bias.to(query_layer.dtype)
        elif attention_mask is not None:
            combined_mask = attention_mask
        elif st_dist_bias is not None:
            combined_mask = st_dist_bias.to(query_layer.dtype)

        if output_attentions:
            attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attn_scores = attn_scores / math.sqrt(self.attention_head_size)
            if combined_mask is not None:
                attn_scores = attn_scores + combined_mask
            attn_probs = F.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_probs, value_layer)
            context = context.transpose(-2, -3).contiguous()
            new_shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(*new_shape)
            return context, attn_probs
        else:
            context = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                attn_mask=combined_mask,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
                scale=1.0 / math.sqrt(self.attention_head_size),
            )
            context = context.transpose(-2, -3).contiguous()
            new_shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(*new_shape)
            return (context,)


class STCrossOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class STColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = STCrossAttention(config)
        self.output = STCrossOutput(config)
        self.fire_spatial = FIREDistanceBias(config.max_spatial_dist, config.fire_hidden_size)
        self.fire_temporal = FIREDistanceBias(config.max_temporal_dist, config.fire_hidden_size)
        # Per-species log-scale for spatial and temporal FIRE biases.
        # Both init to 0 → exp(0)=1, so initial behavior is identical to a shared bias.
        # After training, these reveal which species respond more to space vs time.
        self.species_spatial_log_scale = nn.Parameter(torch.zeros(config.num_species))
        self.species_temporal_log_scale = nn.Parameter(torch.zeros(config.num_species))

    def forward(
        self, hidden_states, source_embeddings,
        attention_mask=None, st_dist=None, output_attentions=False,
    ):
        # Compute per-species FIRE distance bias
        st_dist_bias = None
        if st_dist is not None:
            spatial_bias = self.fire_spatial(st_dist[..., 0])   # (B, T, C_st)
            temporal_bias = self.fire_temporal(st_dist[..., 1])  # (B, T, C_st)
            s_scale = self.species_spatial_log_scale.exp()   # (S,)
            t_scale = self.species_temporal_log_scale.exp()  # (S,)
            # (B, T, C_st) → (B, S, T, C_st) weighted per species
            st_dist_bias = (
                spatial_bias[:, None, :, :] * s_scale[None, :, None, None]
                + temporal_bias[:, None, :, :] * t_scale[None, :, None, None]
            )
            st_dist_bias = st_dist_bias[:, :, None, :, :]  # (B, S, 1, T, C_st)

        cross_outputs = self.cross_attn(
            hidden_states, source_embeddings, attention_mask, st_dist_bias, output_attentions,
        )
        output = (self.output(cross_outputs[0]),)  # project+dropout only; residual in JSDMAttention
        if output_attentions:
            output = output + (cross_outputs[1],)
        return output


# =============================================================================
# Ecological Cross-Attention
# =============================================================================

class EcoCrossAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def forward(self, hidden_states, eco_embeddings, output_attentions=False):
        """
        Args:
            hidden_states: (B, S, T, H)
            eco_embeddings: (B, C_eco, H)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        eco_exp = eco_embeddings.unsqueeze(1).expand(-1, hidden_states.size(1), -1, -1)
        key_layer = self.transpose_for_scores(self.key(eco_exp))
        value_layer = self.transpose_for_scores(self.value(eco_exp))

        if output_attentions:
            attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attn_scores = attn_scores / math.sqrt(self.attention_head_size)
            attn_probs = F.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_probs, value_layer)
            context = context.transpose(-2, -3).contiguous()
            new_shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(*new_shape)
            return context, attn_probs
        else:
            context = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
                scale=1.0 / math.sqrt(self.attention_head_size),
            )
            context = context.transpose(-2, -3).contiguous()
            new_shape = context.size()[:-2] + (self.all_head_size,)
            context = context.view(*new_shape)
            return (context,)


class EcoCrossOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class EcoColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = EcoCrossAttention(config)
        self.output = EcoCrossOutput(config)

    def forward(self, hidden_states, eco_embeddings, output_attentions=False):
        cross_outputs = self.cross_attn(hidden_states, eco_embeddings, output_attentions)
        output = (self.output(cross_outputs[0]),)  # project+dropout only; residual in JSDMAttention
        if output_attentions:
            output = output + (cross_outputs[1],)
        return output


# =============================================================================
# Combined Attention (analogous to GPNStarAttention)
#
# GPN-star: row_attention → col_attention (sequential)
# Here: row_attention → (st_col ⊕ eco_col) → gate combine
# =============================================================================

class JSDMAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.row_attention = SpeciesRowAttention(config)
        self.st_col_attention = STColAttention(config)
        self.eco_col_attention = EcoColAttention(config)
        # Per-species gate: each species learns its own ST vs ecological context balance.
        # init=0 → sigmoid(0)=0.5 for all species. Learns habitat-specialists vs range-generalists.
        self.combine_gate = nn.Parameter(torch.zeros(config.num_species))
        # Isolation scale: learned scalar that down-weights ST context when target site is far from
        # all source sites. init=0 → no adjustment at start; positive → isolated sites lean on eco.
        self.isolation_scale = nn.Parameter(torch.tensor(0.0))
        # Pre-norm layers: applied before each sub-block; residual added here, not inside sub-modules
        self.row_norm  = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None, sinusoidal_pos=None,
        isolation=None, output_attentions=False,
    ):
        # 1. Pre-norm → species self-attention → residual  (B,S,T,H) ↔ (B,T,S,H)
        row_output = self.row_attention(
            hidden_states=self.row_norm(hidden_states).transpose(-2, -3),
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
        )
        h = hidden_states + row_output[0].transpose(-2, -3)

        # 2 & 3. Shared pre-norm → ST and eco cross-attention in parallel → gated residual
        h_normed = self.cross_norm(h)
        st_output  = self.st_col_attention(h_normed, st_source_embeddings, st_attention_mask, st_dist, output_attentions)
        eco_output = self.eco_col_attention(h_normed, eco_embeddings, output_attentions)

        gate_logit = self.combine_gate[None, :, None, None]  # (1, S, 1, 1)
        if isolation is not None:
            # isolation: (B, T) → (B, 1, T, 1); broadcasts over species and hidden dims
            gate_logit = gate_logit - self.isolation_scale * isolation[:, None, :, None]
        gate = torch.sigmoid(gate_logit)
        h = h + gate * st_output[0] + (1 - gate) * eco_output[0]

        out = (h,)
        if output_attentions:
            out = out + (
                row_output[1] if len(row_output) > 1 else None,
                st_output[1]  if len(st_output)  > 1 else None,
                eco_output[1] if len(eco_output) > 1 else None,
            )
        return out


# =============================================================================
# FFN
# =============================================================================

class FeedForward(nn.Module):
    """
    SwiGLU FFN: down(SiLU(gate(x)) * up(x))
    Pre-norm and residual are applied externally in JSDMLayer.
    Note: gate + up doubles the first-layer params vs vanilla FFN.
    Set intermediate_size to ~2/3 of the original value to keep param count equal.
    """
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# =============================================================================
# JSDM Layer, Encoder, Model, ForMaskedPrediction
# =============================================================================

class JSDMLayer(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.attention = JSDMAttention(config)
        self.ffn = FeedForward(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None, sinusoidal_pos=None,
        isolation=None, output_attentions=False,
    ):
        attn_outputs = self.attention(
            hidden_states, st_source_embeddings, eco_embeddings,
            st_attention_mask, st_dist, sinusoidal_pos, isolation, output_attentions,
        )
        h = attn_outputs[0]
        h = h + self.ffn(self.ffn_norm(h))  # pre-norm + residual
        return (h,) + attn_outputs[1:]


class JSDMEncoder(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        # TargetInput's nn.Embedding already outputs (B, S, T, H) — no projection needed here

        self.layers = nn.ModuleList(
            [JSDMLayer(config) for _ in range(config.num_hidden_layers)]
        )
        from transformers.models.roformer.modeling_roformer import (
            RoFormerSinusoidalPositionalEmbedding,
        )
        head_size = config.hidden_size // config.num_attention_heads
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.num_species + 2, head_size
        )
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None, isolation=None,
        output_attentions=False, output_hidden_states=False,
    ):
        all_hidden = () if output_hidden_states else None
        all_sp_attn = () if output_attentions else None
        all_st_attn = () if output_attentions else None
        all_eco_attn = () if output_attentions else None

        sinusoidal_pos = self.embed_positions(
            hidden_states.shape[:-1], 0
        )[None, None, :, :]

        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, sinusoidal_pos, isolation, output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, sinusoidal_pos, isolation, output_attentions,
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


@dataclass
class JSDMEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    species_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    st_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    eco_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# =============================================================================
# ST Cluster Info (analogous to GPNStarPhyloInfo)
# =============================================================================

class STClusterInfo:
    def __init__(
        self,
        spatial_coords,
        temporal_coords,
        threshold=5.0,
        spatial_scale_km: Optional[float] = None,
        temporal_scale_days: Optional[float] = None,
    ):
        import networkx as nx
        from scipy.spatial.distance import cdist

        N = len(spatial_coords)
        self.spatial_dist_pairwise = _haversine_pairwise(
            spatial_coords[:, 0], spatial_coords[:, 1]
        )
        t2d = temporal_coords.reshape(-1, 1)
        self.temporal_dist_pairwise = torch.tensor(
            cdist(t2d, t2d), dtype=torch.float32
        )
        spatial_scale_km = _resolve_scale(
            spatial_scale_km, float(self.spatial_dist_pairwise.max()), "spatial_scale_km"
        )
        temporal_scale_days = _resolve_scale(
            temporal_scale_days, float(self.temporal_dist_pairwise.max()), "temporal_scale_days"
        )
        combined = torch.sqrt(
            (self.spatial_dist_pairwise / spatial_scale_km) ** 2
            + (self.temporal_dist_pairwise / temporal_scale_days) ** 2
        )

        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if combined[i, j] <= threshold:
                    G.add_edge(i, j)

        self.cluster_dict = {
            i: nodes for i, nodes in enumerate(list(nx.connected_components(G)))
        }
        self.cluster_labels = torch.zeros(N, dtype=torch.long)
        for cid, sites in self.cluster_dict.items():
            for s in sites:
                self.cluster_labels[s] = cid

        self.in_cluster_spatial_dist = torch.zeros(N)
        self.in_cluster_temporal_dist = torch.zeros(N)
        for cid, sites in self.cluster_dict.items():
            sites_list = list(sites)
            if len(sites_list) > 1:
                center_s = spatial_coords[sites_list].mean(axis=0)
                center_t = temporal_coords[sites_list].mean()
                for s in sites_list:
                    # haversine from site to cluster centroid (km), consistent with pairwise spatial_dist
                    self.in_cluster_spatial_dist[s] = float(_haversine_pairwise(
                        np.array([spatial_coords[s, 0], center_s[0]]),
                        np.array([spatial_coords[s, 1], center_s[1]])
                    )[0, 1].item())
                    self.in_cluster_temporal_dist[s] = float(
                        abs(temporal_coords[s] - center_t)
                    )

        self.max_spatial_dist = self.spatial_dist_pairwise.max().item()
        self.max_temporal_dist = self.temporal_dist_pairwise.max().item()


# =============================================================================
# Full Model (analogous to GPNStarModel)
# =============================================================================

class JSDMModel(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.config = config
        self.target_input = TargetInput(config)
        self.st_source_module = STSourceModule(config)
        self.eco_source_module = EcoSourceModule(config)
        self.encoder = JSDMEncoder(config)

    def forward(
        self,
        input_ids,              # (B, S, T)
        source_ids,             # (B, S, N)
        source_idx,             # (B, N)
        target_site_idx,        # (B, T)
        env_data,               # (B, N, E)
        cluster_dict,
        cluster_labels,
        in_cluster_spatial_dist,
        in_cluster_temporal_dist,
        spatial_dist_pairwise,
        temporal_dist_pairwise,
        output_attentions=False,
        output_hidden_states=False,
    ):
        # 1. Target input: raw 0/1/mask → (B, S, T, 1)
        hidden_states = self.target_input(input_ids)  # (B, S, T, 1)

        # 2. Pool source sites into ST clusters
        cl_labels = cluster_labels.to(hidden_states.device)
        in_spatial = in_cluster_spatial_dist.to(hidden_states.device)
        in_temporal = in_cluster_temporal_dist.to(hidden_states.device)
        source_cluster_labels = cl_labels[source_idx]  # (B, N)
        source_in_spatial = in_spatial[source_idx]     # (B, N)
        source_in_temporal = in_temporal[source_idx]   # (B, N)
        st_source = self.st_source_module(
            source_ids,
            source_cluster_labels,
            source_in_spatial,
            source_in_temporal,
            num_clusters=len(cluster_dict),
        )  # (B, S, C_st, H)

        # 3. Ecological context
        eco_emb = self.eco_source_module(env_data)  # (B, C_eco, H)

        # 4. Target-to-cluster distances (like target-to-clade phylo dist)
        num_clusters = len(cluster_dict)
        sp_pw = spatial_dist_pairwise.to(hidden_states.device)
        tp_pw = temporal_dist_pairwise.to(hidden_states.device)

        target_sp = sp_pw[target_site_idx]  # (B, T, N)
        target_tp = tp_pw[target_site_idx]  # (B, T, N)
        target_sp_cl = self._cluster_means(target_sp, cl_labels, num_clusters)
        target_tp_cl = self._cluster_means(target_tp, cl_labels, num_clusters)
        st_dist = torch.stack([target_sp_cl, target_tp_cl], dim=-1)  # (B, T, C_st, 2)

        # Isolation: mean normalized ST distance from target to its source sites (B, T)
        # High value → all sources are far → ST context is weak → gate will down-weight ST
        sp_norm = target_sp / self.config.max_spatial_dist
        tp_norm = target_tp / self.config.max_temporal_dist
        isolation = torch.sqrt(sp_norm ** 2 + tp_norm ** 2).mean(dim=-1)  # (B, T)

        # 5. Own-cluster masking
        target_clusters = cl_labels[target_site_idx]  # (B, T)
        attn_mask = F.one_hot(target_clusters, num_classes=num_clusters).to(hidden_states.dtype)
        attn_mask = attn_mask * torch.finfo(hidden_states.dtype).min
        attn_mask = attn_mask[:, None, None, :, :]  # (B, 1, 1, T, C_st)

        # Mask clusters with no sampled sources for this batch
        present = torch.zeros(
            source_cluster_labels.size(0), num_clusters,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        present.scatter_(1, source_cluster_labels, 1.0)
        empty_mask = (present == 0)[:, None, None, None, :]
        attn_mask = attn_mask + empty_mask * torch.finfo(hidden_states.dtype).min

        # 6. Encode
        return self.encoder(
            hidden_states, st_source, eco_emb,
            st_attention_mask=attn_mask, st_dist=st_dist, isolation=isolation,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    @staticmethod
    def _cluster_means(A, indices, C):
        B, T, N = A.shape
        A_flat = A.reshape(-1, N)
        idx_exp = indices.unsqueeze(0).expand(A_flat.shape[0], N)
        sums = torch.zeros(A_flat.shape[0], C, device=A.device, dtype=A.dtype)
        counts = torch.zeros(A_flat.shape[0], C, device=A.device, dtype=A.dtype)
        sums.scatter_add_(1, idx_exp, A_flat)
        counts.scatter_add_(1, idx_exp, torch.ones_like(A_flat))
        counts = counts.clamp(min=1)
        return (sums / counts).view(B, T, C)


# =============================================================================
# Masked prediction wrapper (analogous to GPNStarForMaskedLM)
# =============================================================================

@dataclass
class JSDMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    species_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    st_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    eco_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class JSDMPredictionHead(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.SiLU()
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        x = self.act(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x).squeeze(-1)


class JSDMForMaskedSpeciesPrediction(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.config = config
        self.model = JSDMModel(config)
        self.cls = JSDMPredictionHead(config)

    def forward(self, labels=None, loss_weight=None, output_attentions=False, **kwargs):
        encoder_out = self.model(output_attentions=output_attentions, **kwargs)
        logits = self.cls(encoder_out.last_hidden_state)  # (B, S, T)

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


# =============================================================================
# Extract interaction matrix
# =============================================================================

def extract_interaction_matrix(output: JSDMOutput, layer_idx: int = -1) -> torch.Tensor:
    """Extract S×S species interaction from self-attention. (B, S, S)"""
    if output.species_attentions is None:
        raise ValueError("Run with output_attentions=True")
    attn = output.species_attentions[layer_idx]  # (B, 1, A, S, S)
    return attn.squeeze(1).mean(dim=1)
