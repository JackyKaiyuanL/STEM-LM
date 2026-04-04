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
from typing import Optional, Tuple, Dict, List, Set

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


# =============================================================================
# Target Input (NO embedding needed)
#
# GPN-star: nucleotides are categorical → nn.Embedding(6, H) → (B, L, T, H)
# Here: species states are already numeric 0/1 → use directly as 1-dim embeddings
#   Input: (B, S, T) float values {0.0, 1.0, mask_value}
#   Output: (B, S, T, 1) — the "H=1" embedding, expanded by attention Q/K/V projections
#
# The mask token is a learned parameter distinct from 0 and 1.
# =============================================================================

class TargetInput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        # Learned mask value — distinct from 0.0 (absent) and 1.0 (present)
        # Initialized to -1.0 so it's clearly different from valid states
        self.mask_value = nn.Parameter(torch.tensor(config.mask_value_init))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S, T) float tensor
                       0.0 = absent, 1.0 = present, nan or special → replaced with mask_value
        Returns:
            (B, S, T, 1) — each species state as a 1-dim "embedding"
        """
        return input_ids.unsqueeze(-1)  # (B, S, T, 1)


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

class STAttentionPool(nn.Module):
    """
    Attention pool source sites within a spatial-temporal cluster.
    Mirrors GPNStarAttentionPool structure.

    GPN-star: uses nn.Embedding(vocab_size, ...) because nucleotides are discrete tokens.
    Here: uses nn.Linear(1, ...) because species states are raw 0/1 floats.
    The 0/1 value at each species position is a 1-dim input.
    """

    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections from 1-dim input (instead of Embedding lookups)
        self.attention_weights = nn.Linear(1, self.num_attention_heads)
        self.value = nn.Linear(1, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.ffn = nn.Linear(self.all_head_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def forward(self, source_ids, in_cluster_dist_bias):
        """
        Args:
            source_ids: (B, S, N_c) float species states at N_c sites in this cluster
            in_cluster_dist_bias: broadcastable bias for within-cluster distances

        Returns:
            pooled: (B, S, H)
        """
        x = source_ids.unsqueeze(-1).float()  # (B, S, N_c, 1)

        attention_scores = (
            self.attention_weights(x).transpose(-1, -2).unsqueeze(-2)
        )  # (B, S, A, 1, N_c)
        value_layer = self.transpose_for_scores(
            self.value(x)
        )  # (B, S, A, N_c, D)

        attention_scores = attention_scores + in_cluster_dist_bias
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        pooled = torch.matmul(attention_probs, value_layer)  # (B, S, A, 1, D)
        pooled = pooled.transpose(-2, -3).contiguous()       # (B, S, 1, A, D)
        new_shape = pooled.size()[:-3] + (self.all_head_size,)
        pooled = pooled.view(*new_shape)  # (B, S, H//2)
        pooled = self.ffn(pooled)         # (B, S, H)

        return pooled


class STSourceModule(nn.Module):
    """
    Spatial-temporal source module. Mirrors GPNStarSourceModule exactly.

    GPN-star: source_ids (B,L,N) → group by clade → pool → (B,L,C,H)
    Here: source_ids (B,S,N) → group by ST cluster → pool → (B,S,C_st,H)
    """

    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.attn_pool = STAttentionPool(config)
        # For single-site clusters: project 1-dim float → hidden_size
        self.single_site_proj = nn.Linear(1, config.hidden_size)

        self.fire_spatial = FIREDistanceBias(config.max_spatial_dist, config.fire_hidden_size)
        self.fire_temporal = FIREDistanceBias(config.max_temporal_dist, config.fire_hidden_size)

    def forward(
        self,
        source_ids: torch.Tensor,
        cluster_dict: Dict[int, Set[int]],
        in_cluster_spatial_dist: torch.Tensor,
        in_cluster_temporal_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source_ids: (B, S, N) species states at S species for N source sites
            cluster_dict: {cluster_id: set(site_indices)} like clade_dict
            in_cluster_spatial_dist: (N,) dist from each source to its cluster center
            in_cluster_temporal_dist: (N,) temporal dist to cluster center

        Returns:
            cluster_pooled: (B, S, C_st, H)
        """
        # Compute within-cluster distance bias (like in_clade_time_bias)
        spatial_bias = self.fire_spatial(in_cluster_spatial_dist[None, None, :])
        temporal_bias = self.fire_temporal(in_cluster_temporal_dist[None, None, :])
        in_cluster_bias = spatial_bias + temporal_bias  # (1, 1, N)

        cluster_pooled = []
        for cluster_id, site_indices in cluster_dict.items():
            site_indices = sorted(list(site_indices))
            if len(site_indices) > 1:
                bias = in_cluster_bias[..., site_indices]  # (1, 1, N_c)
                # Reshape for attention pool: (B, S, A, 1, N_c)
                bias = bias[:, :, None, None, :]  # (1, 1, 1, 1, N_c)

                cluster_pooled.append(
                    self.attn_pool(
                        source_ids[..., site_indices],  # (B, S, N_c)
                        bias,
                    )
                )
            else:
                # Single site cluster: project 1-dim float → H (like single-species clade)
                single = source_ids[..., site_indices[0]].unsqueeze(-1).float()  # (B, S, 1)
                cluster_pooled.append(self.single_site_proj(single))

        cluster_pooled = torch.stack(cluster_pooled, dim=-2)  # (B, S, C_st, H)
        return cluster_pooled


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
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class SpeciesRowAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.self_attn = SpeciesSelfAttention(config)
        self.output = SpeciesSelfOutput(config)

    def forward(self, hidden_states, sinusoidal_pos=None, output_attentions=False):
        self_outputs = self.self_attn(hidden_states, sinusoidal_pos, output_attentions)
        output = (self.output(self_outputs[0], hidden_states),)
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
            st_dist_bias: (B, 1, 1, T, C_st) FIRE distance bias
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
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class STColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = STCrossAttention(config)
        self.output = STCrossOutput(config)
        self.fire_spatial = FIREDistanceBias(config.max_spatial_dist, config.fire_hidden_size)
        self.fire_temporal = FIREDistanceBias(config.max_temporal_dist, config.fire_hidden_size)

    def forward(
        self, hidden_states, source_embeddings,
        attention_mask=None, st_dist=None, output_attentions=False,
    ):
        # Compute FIRE distance bias (like evol_time_bias in GPNStarColAttention)
        st_dist_bias = None
        if st_dist is not None:
            spatial_bias = self.fire_spatial(st_dist[..., 0])   # (B, T, C_st)
            temporal_bias = self.fire_temporal(st_dist[..., 1])  # (B, T, C_st)
            st_dist_bias = (spatial_bias + temporal_bias)[:, None, None, :, :]

        cross_outputs = self.cross_attn(
            hidden_states, source_embeddings, attention_mask, st_dist_bias, output_attentions,
        )
        output = (self.output(cross_outputs[0], hidden_states),)
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
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class EcoColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = EcoCrossAttention(config)
        self.output = EcoCrossOutput(config)

    def forward(self, hidden_states, eco_embeddings, output_attentions=False):
        cross_outputs = self.cross_attn(hidden_states, eco_embeddings, output_attentions)
        output = (self.output(cross_outputs[0], hidden_states),)
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
        self.combine_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None, sinusoidal_pos=None,
        output_attentions=False,
    ):
        # 1. Species self-attention (transpose (B,S,T,H) → (B,T,S,H))
        row_output = self.row_attention(
            hidden_states=hidden_states.transpose(-2, -3),
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
        )
        h = row_output[0].transpose(-2, -3)  # back to (B, S, T, H)

        # 2. ST cross-attention
        st_output = self.st_col_attention(
            h, st_source_embeddings, st_attention_mask, st_dist, output_attentions,
        )

        # 3. Ecological cross-attention
        eco_output = self.eco_col_attention(h, eco_embeddings, output_attentions)

        # 4. Combine
        gate = torch.sigmoid(self.combine_gate)
        combined = gate * st_output[0] + (1 - gate) * eco_output[0]

        out = (combined,)
        if output_attentions:
            out = out + (
                row_output[1] if len(row_output) > 1 else None,
                st_output[1] if len(st_output) > 1 else None,
                eco_output[1] if len(eco_output) > 1 else None,
            )
        return out


# =============================================================================
# FFN
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        residual = hidden_states
        x = self.dense1(hidden_states)
        x = self.act(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.LayerNorm(x + residual)


# =============================================================================
# JSDM Layer, Encoder, Model, ForMaskedPrediction
# =============================================================================

class JSDMLayer(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.attention = JSDMAttention(config)
        self.ffn = FeedForward(config)

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None, sinusoidal_pos=None,
        output_attentions=False,
    ):
        attn_outputs = self.attention(
            hidden_states, st_source_embeddings, eco_embeddings,
            st_attention_mask, st_dist, sinusoidal_pos, output_attentions,
        )
        layer_output = self.ffn(attn_outputs[0])
        return (layer_output,) + attn_outputs[1:]


class JSDMEncoder(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        # Project raw 1-dim species states to hidden_size
        # This is the only place where dim expansion happens
        self.input_proj = nn.Linear(1, config.hidden_size)

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
        st_attention_mask=None, st_dist=None,
        output_attentions=False, output_hidden_states=False,
    ):
        all_hidden = () if output_hidden_states else None
        all_sp_attn = () if output_attentions else None
        all_st_attn = () if output_attentions else None
        all_eco_attn = () if output_attentions else None

        # Project raw 1-dim input to hidden_size: (B, S, T, 1) → (B, S, T, H)
        hidden_states = self.input_proj(hidden_states)

        sinusoidal_pos = self.embed_positions(
            hidden_states.shape[:-1], 0
        )[None, None, :, :]

        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, sinusoidal_pos, output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, sinusoidal_pos, output_attentions,
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
    def __init__(self, spatial_coords, temporal_coords, threshold=5.0):
        import networkx as nx
        from scipy.spatial.distance import cdist

        N = len(spatial_coords)
        self.spatial_dist_pairwise = torch.tensor(
            cdist(spatial_coords, spatial_coords), dtype=torch.float32
        )
        t2d = temporal_coords.reshape(-1, 1)
        self.temporal_dist_pairwise = torch.tensor(
            cdist(t2d, t2d), dtype=torch.float32
        )
        combined = self.spatial_dist_pairwise + self.temporal_dist_pairwise

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
                    self.in_cluster_spatial_dist[s] = float(
                        np.sqrt(((spatial_coords[s] - center_s) ** 2).sum())
                    )
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
        st_source = self.st_source_module(
            source_ids, cluster_dict,
            in_cluster_spatial_dist.to(hidden_states.device),
            in_cluster_temporal_dist.to(hidden_states.device),
        )  # (B, S, C_st, H)

        # 3. Ecological context
        eco_emb = self.eco_source_module(env_data)  # (B, C_eco, H)

        # 4. Target-to-cluster distances (like target-to-clade phylo dist)
        num_clusters = len(cluster_dict)
        cl_labels = cluster_labels.to(hidden_states.device)
        sp_pw = spatial_dist_pairwise.to(hidden_states.device)
        tp_pw = temporal_dist_pairwise.to(hidden_states.device)

        target_sp = sp_pw[target_site_idx]  # (B, T, N)
        target_tp = tp_pw[target_site_idx]  # (B, T, N)
        target_sp_cl = self._cluster_means(target_sp, cl_labels, num_clusters)
        target_tp_cl = self._cluster_means(target_tp, cl_labels, num_clusters)
        st_dist = torch.stack([target_sp_cl, target_tp_cl], dim=-1)  # (B, T, C_st, 2)

        # 5. Own-cluster masking
        target_clusters = cl_labels[target_site_idx]  # (B, T)
        attn_mask = F.one_hot(target_clusters, num_classes=num_clusters).to(hidden_states.dtype)
        attn_mask = attn_mask * torch.finfo(hidden_states.dtype).min
        attn_mask = attn_mask[:, None, None, :, :]  # (B, 1, 1, T, C_st)

        # 6. Encode
        return self.encoder(
            hidden_states, st_source, eco_emb,
            st_attention_mask=attn_mask, st_dist=st_dist,
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
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
