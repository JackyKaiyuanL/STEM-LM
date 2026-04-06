"""
Spatial-Temporal Joint Species Distribution Model (ST-JSDM) — citizen-science variant.

Species presence/absence at a target site is predicted from:
  1. Pooled observations from nearby sites in space and time (ST cross-attention)
  2. Environmental covariate context (ecological cross-attention)
  3. Joint species interactions at the target site (species self-attention)

Temporal attention is causal: source sites from the future cannot inform the target.
Time is decomposed into:
  - Elapsed Δt: causal recency, signed (past sources only; future masked out)
  - Day-of-year: circular seasonal distance for phenological context
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
    num_species: int = 100
    mask_value_init: float = -1.0

    num_source_sites: int = 64
    num_target_sites: int = 1

    max_spatial_dist: float = 180.0
    max_temporal_dist: float = 365.0
    max_doy_dist: float = 182.0       # half-year; max circular DOY distance

    num_env_vars: int = 10

    hidden_size: int = 256
    num_attention_heads: int = 8
    num_hidden_layers: int = 4
    intermediate_size: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6

    fire_hidden_size: int = 32
    mlm_probability: float = 0.15


# =============================================================================
# Utilities
# =============================================================================

class FIREDistanceBias(nn.Module):
    """Log-compressed learned distance bias."""

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
        dist = dist.unsqueeze(-1).float()
        c = self.c.clamp(min=0)
        dist = torch.log(c * dist + 1) / torch.log(c * self.max_dist + 1)
        return self.mlp(dist).squeeze(-1)


def _haversine_pairwise(lat_deg: np.ndarray, lon_deg: np.ndarray) -> torch.Tensor:
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return torch.tensor(6371.0 * 2.0 * np.arcsin(np.sqrt(a)), dtype=torch.float32)


def _resolve_scale(value: Optional[float], fallback: float, name: str) -> float:
    if value is None:
        if fallback <= 0:
            raise ValueError(f"Cannot auto-scale {name}: max pairwise distance is {fallback}")
        return float(fallback)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return float(value)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# =============================================================================
# Target Input — 3 discrete tokens: 0=absent, 1=present, 2=mask
# =============================================================================

class TargetInput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, S, T) long → (B, S, T, H)"""
        return self.embedding(input_ids)


# =============================================================================
# ST Source Embedder
#
# Embed per-source observations directly as tokens (B, S, N, H). Distance and
# causal structure is handled in cross-attention, not in this module.
# =============================================================================

class STSourceEmbedder(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.value = nn.Linear(1, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, source_ids: torch.Tensor) -> torch.Tensor:
        x = self.value(source_ids.unsqueeze(-1).float())
        return self.layer_norm(self.dropout(x))


# =============================================================================
# Ecological Source Module
# =============================================================================

class EcoSourceModule(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.proj = nn.Linear(config.num_env_vars, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, env_data: torch.Tensor) -> torch.Tensor:
        """env_data: (B, N, E) → (B, C_eco, H)"""
        site_emb = self.proj(env_data)
        return self.layer_norm(self.dropout(site_emb))


# =============================================================================
# Species Self-Attention (along species axis — interaction term)
# =============================================================================

class SpeciesSelfAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query   = nn.Linear(config.hidden_size, self.all_head_size)
        self.key     = nn.Linear(config.hidden_size, self.all_head_size)
        self.value   = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):
        x = x.view(x.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        return x.transpose(-2, -3)

    def forward(self, hidden_states, sinusoidal_pos=None, output_attentions=False):
        """hidden_states: (B, T, S, H)"""
        query_layer = self.transpose_for_scores(self.query(hidden_states[:, :1]))
        key_layer   = self.transpose_for_scores(self.key(hidden_states[:, :1]))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if sinusoidal_pos is not None:
            query_layer, key_layer = self._apply_rotary(sinusoidal_pos, query_layer, key_layer)

        if output_attentions:
            attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attn_scores = attn_scores / math.sqrt(self.attention_head_size)
            attn_probs  = self.dropout(F.softmax(attn_scores, dim=-1))
            context     = torch.matmul(attn_probs, value_layer)
            context     = context.transpose(-2, -3).contiguous()
            context     = context.view(*context.size()[:-2] + (self.all_head_size,))
            return context, attn_probs
        else:
            context = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
                scale=1.0 / math.sqrt(self.attention_head_size),
            )
            context = context.transpose(-2, -3).contiguous()
            context = context.view(*context.size()[:-2] + (self.all_head_size,))
            return (context,)

    @staticmethod
    def _apply_rotary(sinusoidal_pos, query_layer, key_layer):
        sin, cos    = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos     = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos     = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        rotate_half_q = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(query_layer)
        rotate_half_k = torch.stack([-key_layer[..., 1::2],   key_layer[..., ::2]],   dim=-1).reshape_as(key_layer)
        return query_layer * cos_pos + rotate_half_q * sin_pos, key_layer * cos_pos + rotate_half_k * sin_pos


class SpeciesSelfOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense   = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class SpeciesRowAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.self_attn = SpeciesSelfAttention(config)
        self.output    = SpeciesSelfOutput(config)

    def forward(self, hidden_states, sinusoidal_pos=None, output_attentions=False):
        self_outputs = self.self_attn(hidden_states, sinusoidal_pos, output_attentions)
        output = (self.output(self_outputs[0]),)
        if output_attentions:
            output = output + (self_outputs[1],)
        return output


# =============================================================================
# ST Cross-Attention
#
# FIRE bias combines:
#   spatial:  learned bias on km distance to each cluster (per-species scale)
#   elapsed:  learned bias on |Δt| to each cluster (per-species scale)
#   seasonal: learned bias on circular DOY distance (shared across species)
# =============================================================================

class STCrossAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads // 2
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query   = nn.Linear(config.hidden_size, self.all_head_size)
        self.key     = nn.Linear(config.hidden_size, self.all_head_size)
        self.value   = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):
        x = x.view(x.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        return x.transpose(-2, -3)

    def forward(self, hidden_states, source_embeddings,
                attention_mask=None, st_dist_bias=None, output_attentions=False):
        """
        hidden_states:     (B, S, T, H)
        source_embeddings: (B, S, C_st, H)
        attention_mask:    (B, 1, 1, T, C_st)
        st_dist_bias:      (B, S, 1, T, C_st)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer   = self.transpose_for_scores(self.key(source_embeddings))
        value_layer = self.transpose_for_scores(self.value(source_embeddings))

        if attention_mask is not None and st_dist_bias is not None:
            combined_mask = attention_mask + st_dist_bias.to(query_layer.dtype)
        elif attention_mask is not None:
            combined_mask = attention_mask
        elif st_dist_bias is not None:
            combined_mask = st_dist_bias.to(query_layer.dtype)
        else:
            combined_mask = None

        if output_attentions:
            attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attn_scores = attn_scores / math.sqrt(self.attention_head_size)
            if combined_mask is not None:
                attn_scores = attn_scores + combined_mask
            attn_probs = F.softmax(attn_scores, dim=-1)
            context    = torch.matmul(attn_probs, value_layer)
            context    = context.transpose(-2, -3).contiguous()
            context    = context.view(*context.size()[:-2] + (self.all_head_size,))
            return context, attn_probs
        else:
            context = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                attn_mask=combined_mask,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
                scale=1.0 / math.sqrt(self.attention_head_size),
            )
            context = context.transpose(-2, -3).contiguous()
            context = context.view(*context.size()[:-2] + (self.all_head_size,))
            return (context,)


class STCrossOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense   = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class STColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn    = STCrossAttention(config)
        self.output        = STCrossOutput(config)
        self.fire_spatial  = FIREDistanceBias(config.max_spatial_dist,  config.fire_hidden_size)
        self.fire_temporal = FIREDistanceBias(config.max_temporal_dist, config.fire_hidden_size)
        self.fire_seasonal = FIREDistanceBias(config.max_doy_dist,      config.fire_hidden_size)
        self.species_spatial_log_scale  = nn.Parameter(torch.zeros(config.num_species))
        self.species_temporal_log_scale = nn.Parameter(torch.zeros(config.num_species))

    def forward(self, hidden_states, source_embeddings,
                attention_mask=None, st_dist=None, st_doy_dist=None, output_attentions=False):
        """
        st_dist:     (B, T, C_st, 2) — [spatial_km, elapsed_days]
        st_doy_dist: (B, T, C_st)    — circular DOY distance to each cluster
        """
        st_dist_bias = None
        if st_dist is not None:
            spatial_bias  = self.fire_spatial(st_dist[..., 0])    # (B, T, C_st)
            temporal_bias = self.fire_temporal(st_dist[..., 1])   # (B, T, C_st)
            s_scale = self.species_spatial_log_scale.exp()         # (S,)
            t_scale = self.species_temporal_log_scale.exp()        # (S,)
            st_dist_bias = (
                spatial_bias[:, None, :, :]  * s_scale[None, :, None, None]
                + temporal_bias[:, None, :, :] * t_scale[None, :, None, None]
            )  # (B, S, T, C_st)
            if st_doy_dist is not None:
                seasonal_bias = self.fire_seasonal(st_doy_dist)    # (B, T, C_st)
                st_dist_bias  = st_dist_bias + seasonal_bias[:, None, :, :]
            st_dist_bias = st_dist_bias[:, :, None, :, :]          # (B, S, 1, T, C_st)

        cross_outputs = self.cross_attn(
            hidden_states, source_embeddings, attention_mask, st_dist_bias, output_attentions,
        )
        output = (self.output(cross_outputs[0]),)
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
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size       = self.num_attention_heads * self.attention_head_size

        self.query   = nn.Linear(config.hidden_size, self.all_head_size)
        self.key     = nn.Linear(config.hidden_size, self.all_head_size)
        self.value   = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):
        x = x.view(x.size()[:-1] + (self.num_attention_heads, self.attention_head_size))
        return x.transpose(-2, -3)

    def forward(self, hidden_states, eco_embeddings, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        eco_exp     = eco_embeddings.unsqueeze(1).expand(-1, hidden_states.size(1), -1, -1)
        key_layer   = self.transpose_for_scores(self.key(eco_exp))
        value_layer = self.transpose_for_scores(self.value(eco_exp))

        if output_attentions:
            attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attn_scores = attn_scores / math.sqrt(self.attention_head_size)
            attn_probs  = F.softmax(attn_scores, dim=-1)
            context     = torch.matmul(attn_probs, value_layer)
            context     = context.transpose(-2, -3).contiguous()
            context     = context.view(*context.size()[:-2] + (self.all_head_size,))
            return context, attn_probs
        else:
            context = F.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
                scale=1.0 / math.sqrt(self.attention_head_size),
            )
            context = context.transpose(-2, -3).contiguous()
            context = context.view(*context.size()[:-2] + (self.all_head_size,))
            return (context,)


class EcoCrossOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense   = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class EcoColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = EcoCrossAttention(config)
        self.output     = EcoCrossOutput(config)

    def forward(self, hidden_states, eco_embeddings, output_attentions=False):
        cross_outputs = self.cross_attn(hidden_states, eco_embeddings, output_attentions)
        output = (self.output(cross_outputs[0]),)
        if output_attentions:
            output = output + (cross_outputs[1],)
        return output


# =============================================================================
# Combined Attention Block
# =============================================================================

class JSDMAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.row_attention     = SpeciesRowAttention(config)
        self.st_col_attention  = STColAttention(config)
        self.eco_col_attention = EcoColAttention(config)
        self.combine_gate = nn.Parameter(torch.zeros(config.num_species))
        self.row_norm     = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_norm   = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, st_source_embeddings, eco_embeddings,
                st_attention_mask=None, st_dist=None, st_doy_dist=None,
                sinusoidal_pos=None, output_attentions=False):
        row_output = self.row_attention(
            hidden_states=self.row_norm(hidden_states).transpose(-2, -3),
            sinusoidal_pos=sinusoidal_pos,
            output_attentions=output_attentions,
        )
        h = hidden_states + row_output[0].transpose(-2, -3)

        h_normed   = self.cross_norm(h)
        st_output  = self.st_col_attention(h_normed, st_source_embeddings,
                                            st_attention_mask, st_dist, st_doy_dist,
                                            output_attentions)
        eco_output = self.eco_col_attention(h_normed, eco_embeddings, output_attentions)

        gate = torch.sigmoid(self.combine_gate)[None, :, None, None]
        h    = h + gate * st_output[0] + (1 - gate) * eco_output[0]

        out = (h,)
        if output_attentions:
            out = out + (
                row_output[1] if len(row_output) > 1 else None,
                st_output[1]  if len(st_output)  > 1 else None,
                eco_output[1] if len(eco_output) > 1 else None,
            )
        return out


# =============================================================================
# SwiGLU FFN
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.gate    = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up      = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down    = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# =============================================================================
# Layer and Encoder
# =============================================================================

class JSDMLayer(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.attention = JSDMAttention(config)
        self.ffn       = FeedForward(config)
        self.ffn_norm  = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, st_source_embeddings, eco_embeddings,
                st_attention_mask=None, st_dist=None, st_doy_dist=None,
                sinusoidal_pos=None, output_attentions=False):
        attn_outputs = self.attention(
            hidden_states, st_source_embeddings, eco_embeddings,
            st_attention_mask, st_dist, st_doy_dist, sinusoidal_pos, output_attentions,
        )
        h = attn_outputs[0]
        h = h + self.ffn(self.ffn_norm(h))
        return (h,) + attn_outputs[1:]


class JSDMEncoder(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.layers = nn.ModuleList([JSDMLayer(config) for _ in range(config.num_hidden_layers)])
        from transformers.models.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding
        head_size = config.hidden_size // config.num_attention_heads
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(config.num_species + 2, head_size)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, st_source_embeddings, eco_embeddings,
                st_attention_mask=None, st_dist=None, st_doy_dist=None,
                output_attentions=False, output_hidden_states=False):
        all_hidden   = () if output_hidden_states else None
        all_sp_attn  = () if output_attentions else None
        all_st_attn  = () if output_attentions else None
        all_eco_attn = () if output_attentions else None

        sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1], 0)[None, None, :, :]

        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, st_doy_dist, sinusoidal_pos, output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states, st_source_embeddings, eco_embeddings,
                    st_attention_mask, st_dist, st_doy_dist, sinusoidal_pos, output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_sp_attn  = all_sp_attn  + (layer_outputs[1],)
                all_st_attn  = all_st_attn  + (layer_outputs[2],)
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
# Full Model
# =============================================================================

class JSDMModel(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.config            = config
        self.target_input      = TargetInput(config)
        self.st_source_embedder = STSourceEmbedder(config)
        self.eco_source_module = EcoSourceModule(config)
        self.encoder           = JSDMEncoder(config)

    def forward(
        self,
        input_ids,                  # (B, S, T) long
        source_ids,                 # (B, S, N)
        source_idx,                 # (B, N) long
        target_site_idx,            # (B, T) long
        env_data,                   # (B, N, E)
        source_spatial_dist,        # (B, N) or (B, T, N)
        source_temporal_dist,       # (B, N) or (B, T, N)
        source_doy_dist,            # (B, N) or (B, T, N)
        source_time,                # (B, N)
        target_time,                # (B,) or (B, T)
        output_attentions=False,
        output_hidden_states=False,
    ):
        hidden_states = self.target_input(input_ids)  # (B, S, T, H)
        dev = hidden_states.device
        B, _, T = input_ids.shape

        source_idx = source_idx.to(dev)
        target_site_idx = target_site_idx.to(dev)
        st_source = self.st_source_embedder(source_ids)  # (B, S, N, H)

        eco_emb = self.eco_source_module(env_data)  # (B, C_eco, H)

        source_spatial_dist = source_spatial_dist.to(dev)
        source_temporal_dist = source_temporal_dist.to(dev)
        source_doy_dist = source_doy_dist.to(dev)
        source_time = source_time.to(dev)
        target_time = target_time.to(dev)

        if target_time.dim() == 1:
            target_time = target_time[:, None].expand(B, T)
        if source_time.dim() == 1:
            source_time = source_time[:, None]

        if source_spatial_dist.dim() == 2:
            st_spatial = source_spatial_dist[:, None, :].expand(B, T, -1)
            st_temporal = source_temporal_dist[:, None, :].expand(B, T, -1)
            st_doy_dist = source_doy_dist[:, None, :].expand(B, T, -1)
        else:
            st_spatial = source_spatial_dist
            st_temporal = source_temporal_dist
            st_doy_dist = source_doy_dist

        st_dist = torch.stack([st_spatial, st_temporal], dim=-1)  # (B, T, N, 2)

        is_future = source_time[:, None, :] > target_time[..., None]  # (B, T, N)
        same_site = source_idx[:, None, :].eq(target_site_idx)         # (B, T, N)
        mask = is_future | same_site
        attn_mask = mask[:, None, None, :, :].to(hidden_states.dtype)
        attn_mask = attn_mask * torch.finfo(hidden_states.dtype).min

        return self.encoder(
            hidden_states, st_source, eco_emb,
            st_attention_mask=attn_mask,
            st_dist=st_dist,
            st_doy_dist=st_doy_dist,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


# =============================================================================
# Masked prediction head and wrapper
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
        self.dense      = nn.Linear(config.hidden_size, config.hidden_size)
        self.act        = nn.SiLU()
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder    = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        return self.decoder(self.layer_norm(self.act(self.dense(hidden_states)))).squeeze(-1)


# =============================================================================
# Occupancy-Detection Head (commented out)
# Uncomment cls init and occupancy blocks in JSDMForMaskedSpeciesPrediction to activate.
# P(x_s=1) = ψ_s · p_s, where p_s = σ(species_detect_logit[s] + effort_proj(target_env)[s])
# =============================================================================
#
# class OccupancyDetectionHead(nn.Module):
#     def __init__(self, config: JSDMConfig):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.act = nn.SiLU()
#         self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.occupancy_decoder = nn.Linear(config.hidden_size, 1)
#         self.species_detect_logit = nn.Parameter(torch.full((config.num_species,), 2.2))
#         self.effort_proj = nn.Linear(config.num_env_vars, config.num_species)
#         nn.init.zeros_(self.effort_proj.weight)
#         nn.init.zeros_(self.effort_proj.bias)
#
#     def forward(self, hidden_states, target_env=None):
#         x = self.act(self.dense(hidden_states))
#         x = self.layer_norm(x)
#         psi_logit = self.occupancy_decoder(x).squeeze(-1)
#         p_detect_logit = self.species_detect_logit
#         if target_env is not None:
#             p_detect_logit = p_detect_logit.unsqueeze(0) + self.effort_proj(target_env)
#         return psi_logit, p_detect_logit


class JSDMForMaskedSpeciesPrediction(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.config = config
        self.model  = JSDMModel(config)
        self.cls    = JSDMPredictionHead(config)
        # self.occupancy_cls = OccupancyDetectionHead(config)

    def forward(self, labels=None, loss_weight=None, output_attentions=False, target_env=None, **kwargs):
        encoder_out = self.model(output_attentions=output_attentions, **kwargs)
        logits = self.cls(encoder_out.last_hidden_state)  # (B, S, T)

        # === Occupancy-detection model (uncomment to replace simple BCE) ===
        # psi_logit, p_detect_logit = self.occupancy_cls(encoder_out.last_hidden_state, target_env)
        # logits = psi_logit

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
                    w    = loss_weight.unsqueeze(-1).expand_as(labels)[mask]
                    loss = (per_el * w).sum() / w.sum()

                # ── Loss option A: SPML positive-only ─────────────────────────────────
                # pos_mask = mask & (labels == 1)
                # if pos_mask.any():
                #     loss = F.binary_cross_entropy_with_logits(
                #         logits[pos_mask].float(), labels[pos_mask].float()
                #     )

                # ── Loss option B: SPML AN-loss (asymmetric downweighting) ────────────
                # alpha = 0.1
                # y = labels[mask].float()
                # per_el = F.binary_cross_entropy_with_logits(logits[mask].float(), y, reduction="none")
                # w = torch.where(y == 1, torch.ones_like(y), y.new_full((), alpha))
                # loss = (per_el * w).mean()

                # ── Loss option C: Occupancy-detection composite likelihood ────────────
                # psi = torch.sigmoid(psi_logit)
                # p_det = torch.sigmoid(p_detect_logit)
                # if p_det.dim() == 1:
                #     p_det = p_det[None, :, None]
                # else:
                #     p_det = p_det.unsqueeze(-1)
                # p_obs = (psi * p_det).clamp(1e-6, 1 - 1e-6)
                # y = labels[mask].float()
                # loss = -(y * p_obs[mask].log() + (1 - y) * (1 - p_obs[mask]).log()).mean()

        return JSDMOutput(
            loss=loss, logits=logits,
            hidden_states=encoder_out.hidden_states,
            species_attentions=encoder_out.species_attentions,
            st_attentions=encoder_out.st_attentions,
            eco_attentions=encoder_out.eco_attentions,
        )


def extract_interaction_matrix(output: JSDMOutput, layer_idx: int = -1) -> torch.Tensor:
    """Extract S×S species interaction from self-attention weights. Returns (B, S, S)."""
    if output.species_attentions is None:
        raise ValueError("Run with output_attentions=True")
    attn = output.species_attentions[layer_idx]  # (B, 1, A, S, S)
    return attn.squeeze(1).mean(dim=1)
