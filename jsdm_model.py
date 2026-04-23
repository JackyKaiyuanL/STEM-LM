"""
S(pacial)T(empral)E(co)(JSD)M-L(anguage)M(odel)

Architecture:
    Row self-attention (along S species)  = species attend to species (INTERACTION TERM)
    Col cross-attention 1 (along N)       = target site attends directly to N source sites
                                            with per-species FIRE distance bias
    Col cross-attention 2 (env groups)    = target site attends to pooled env variable groups

Data layout:
    CSV columns: time, latitude, longitude, env1, env2, ..., species1, species2, ...
    Each row: one site-time observation with environmental covariates + species 0/1

Training:
    1. Sample a batch of target site-time rows
    2. For each target, sample N source sites weighted by proximity
    3. Collator masks 15% of species in the target row; blinds same-cluster source entries
    4. Self-attention along species axis captures species interactions
    5. Cross-attention to source site embeddings with FIRE ST distance bias
    6. Cross-attention to ecological variable group embeddings (env context)
    7. Predict masked species (BCE loss)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JSDMConfig:
    """Configuration for STEM-LM."""  # S(pacial)T(empral)E(co)(JSD)M-L(anguage)M(odel)
    # Species
    num_species: int = 100           # S: number of species columns

    # Source sites
    num_source_sites: int = 64       # N: number of source site-time obs per example

    # Distance normalization (set from data at training time)
    max_spatial_dist: float = 180.0
    max_temporal_dist: float = 365.0
    use_temporal: bool = True          # False for static datasets where all times are equal

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

    # Gate-MLP bottleneck for combining ST and Eco cross-attention outputs.
    # Default is 2 × hidden_size (set at __post_init__ if left None).
    gate_hidden_size: Optional[int] = None

    # Training — per-class mask rates. Each is a float in [0, 1] or a
    # 'rand[:lo,hi]' string (sample Uniform[lo, hi] per row; 'rand' = 'rand:0.0,1.0').
    p_pres: "float | str" = 0.15
    p_abs:  "float | str" = 0.15

    def __post_init__(self):
        if self.num_attention_heads < 2 or self.num_attention_heads % 2 != 0:
            raise ValueError(
                f"num_attention_heads must be even and >= 2 (splits into row + cross "
                f"sub-blocks); got {self.num_attention_heads}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.gate_hidden_size is None:
            # Historical default: H/8 (what older checkpoints were trained with).
            # Pass --gate_hidden_size explicitly (e.g. 2*hidden_size) in training
            # to widen the gate.
            self.gate_hidden_size = self.hidden_size // 8


# =============================================================================
# FIRE Distance Bias
# =============================================================================

class FIREDistanceBias(nn.Module):
    """
    Learned scalar bias applied to attention logits as a function of distance.
    Used here for spatial and (optionally) temporal distance between target
    and source sites.
    """

    def __init__(self, max_dist: float, fire_hidden_size: int = 32):
        super().__init__()
        self.log_c = nn.Parameter(torch.tensor(0.0))   # softplus(0)+1e-4 ≈ 0.693+1e-4
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
        c = F.softplus(self.log_c) + 1e-4
        denom = torch.log1p(c * self.max_dist)
        x = torch.log1p(c * dist) / denom
        return self.mlp(x).squeeze(-1)



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
# State embedding: 3 tokens (0=absent, 1=present, 2=mask) → H-dim vector.
# Species identity embedding: learned per-species vector (S × H), added to
#   state embedding so every species has a distinct input representation
#   independent of its state. Without this, two absent species are identical
#   vectors — the model has no handle on which species it is processing.
#   Species columns are a fixed named set, not an exchangeable sequence, so
#   permutation equivariance along the species axis is wrong. No positional
#   encoding since species ordering is arbitrary.
# =============================================================================

class TargetInput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_size)
        self.species_embedding = nn.Embedding(config.num_species, config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, S, T) long tensor — 0=absent, 1=present, 2=mask
        Returns:
            (B, S, T, H)
        """
        state_emb = self.embedding(input_ids)  # (B, S, T, H)
        species_idx = torch.arange(input_ids.size(1), device=input_ids.device)
        species_emb = self.species_embedding(species_idx)  # (S, H)
        return state_emb + species_emb[None, :, None, :]



# =============================================================================
# Target Environment Module
# Projects target site's own env vars into a single token appended to eco_emb.
# Two-layer MLP + RMSNorm matches the processing depth of eco_emb.
# =============================================================================

class TargetEnvModule(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.env_norm = nn.LayerNorm(config.num_env_vars)
        self.proj1    = nn.Linear(config.num_env_vars, config.hidden_size)
        self.act      = nn.SiLU()
        self.proj2    = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, target_env: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_env: (B, E)
        Returns:
            (B, 1, H) — one token concatenated onto eco_emb (B, C_eco, H)
        """
        x = self.proj1(self.env_norm(target_env))
        x = self.proj2(self.act(x))
        return self.out_norm(x).unsqueeze(1)


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
        self.env_norm = nn.LayerNorm(config.num_env_vars)  # normalize raw env values before projection
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
        site_emb = self.proj(self.env_norm(env_data))  # (B, N, H)
        k = self.key_proj(site_emb)
        v = self.value_proj(site_emb)
        q = self.group_query.unsqueeze(0).expand(B, -1, -1)  # (B, C_eco, H)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, v)
        return self.layer_norm(self.out_proj(pooled))


# =============================================================================
# Row Self-Attention along species axis
# Species attend to each other — the interaction term.
# =============================================================================

class SpeciesSelfAttention(nn.Module):
    """
    Species self-attention along the species axis.
    Input has been pre-transposed to (B, T, S, H). With T=1 hard-assumed
    throughout this model, Q/K/V all come from the single target row.
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.transpose(-2, -3)

    def forward(self, hidden_states, output_attentions=False):
        """
        Args:
            hidden_states: (B, T, S, H) transposed before calling
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))  # (B, T, A, S, D)
        key_layer   = self.transpose_for_scores(self.key(hidden_states))    # (B, T, A, S, D)
        value_layer = self.transpose_for_scores(self.value(hidden_states))  # (B, T, A, S, D)

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

    def forward(self, hidden_states, output_attentions=False):
        self_outputs = self.self_attn(hidden_states, output_attentions)
        output = (self.output(self_outputs[0]),)  # project+dropout only; residual in JSDMAttention
        if output_attentions:
            output = output + (self_outputs[1],)
        return output


# =============================================================================
# ST Cross-Attention: target species attend to source-site tokens
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
            attn_probs_drop = F.dropout(
                attn_probs, p=self.attention_probs_dropout_prob, training=self.training
            )
            context = torch.matmul(attn_probs_drop, value_layer)
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
        self.use_temporal = config.use_temporal
        self.fire_spatial = FIREDistanceBias(config.max_spatial_dist, config.fire_hidden_size)
        if self.use_temporal:
            self.fire_temporal = FIREDistanceBias(config.max_temporal_dist, config.fire_hidden_size)
        self.species_spatial_log_scale = nn.Parameter(torch.zeros(config.num_species))
        if self.use_temporal:
            self.species_temporal_log_scale = nn.Parameter(torch.zeros(config.num_species))

    def forward(
        self, hidden_states, source_embeddings,
        attention_mask=None, st_dist=None, output_attentions=False,
    ):
        # Compute per-species FIRE distance bias
        st_dist_bias = None
        if st_dist is not None:
            spatial_bias = self.fire_spatial(st_dist[..., 0])          # (B, T, C_st)
            s_scale = F.softplus(self.species_spatial_log_scale) + 1e-4
            st_dist_bias = spatial_bias[:, None, :, :] * s_scale[None, :, None, None]
            if self.use_temporal:
                temporal_bias = self.fire_temporal(st_dist[..., 1])  # (B, T, C_st)
                t_scale = F.softplus(self.species_temporal_log_scale) + 1e-4
                st_dist_bias = st_dist_bias + temporal_bias[:, None, :, :] * t_scale[None, :, None, None]
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
            attn_probs_drop = F.dropout(
                attn_probs, p=self.attention_probs_dropout_prob, training=self.training
            )
            context = torch.matmul(attn_probs_drop, value_layer)
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
# Combined Attention
#
# row_attention → (st_col ⊕ eco_col) → gate combine
# =============================================================================

class JSDMAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.row_attention = SpeciesRowAttention(config)
        self.st_col_attention = STColAttention(config)
        self.eco_col_attention = EcoColAttention(config)
        self.combine_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.gate_hidden_size),
            nn.SiLU(),
            nn.Linear(config.gate_hidden_size, 1),
        )
        # Pre-norm layers: applied before each sub-block; residual added here, not inside sub-modules
        self.row_norm  = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states, st_source_embeddings, eco_embeddings,
        st_attention_mask=None, st_dist=None,
        output_attentions=False,
    ):
        # 1. Pre-norm → species self-attention → residual  (B,S,T,H) ↔ (B,T,S,H)
        row_output = self.row_attention(
            hidden_states=self.row_norm(hidden_states).transpose(-2, -3),
            output_attentions=output_attentions,
        )
        h = hidden_states + row_output[0].transpose(-2, -3)

        # 2 & 3. Shared pre-norm → ST and eco cross-attention in parallel → gated residual
        h_normed = self.cross_norm(h)
        st_output  = self.st_col_attention(h_normed, st_source_embeddings, st_attention_mask, st_dist, output_attentions)
        eco_output = self.eco_col_attention(h_normed, eco_embeddings, output_attentions)

        gate = torch.sigmoid(self.combine_gate(h_normed))  # (B, S, T, 1)
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
        st_attention_mask=None, st_dist=None,
        output_attentions=False,
    ):
        attn_outputs = self.attention(
            hidden_states, st_source_embeddings, eco_embeddings,
            st_attention_mask, st_dist, output_attentions,
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
        self.config = config
        self.target_input = TargetInput(config)
        self.target_env_module = TargetEnvModule(config)
        self.eco_source_module = EcoSourceModule(config)
        self.encoder = JSDMEncoder(config)

    def forward(
        self,
        input_ids,              # (B, S, T) long — 0=absent, 1=present, 2=mask
        source_ids,             # (B, S, N) long — 0=absent, 1=present, 2=blinded
        source_idx,             # (B, N) long — indices into full distance matrices
        target_site_idx,        # (B, T) long
        env_data,               # (B, N, E) — source site env vars
        target_env,             # (B, E)    — target site's own env vars
        spatial_dist_pairwise,  # (N_total, N_total) float
        temporal_dist_pairwise, # (N_total, N_total) float
        output_attentions=False,
        output_hidden_states=False,
    ):
        # 1. Target input: token ids → (B, S, T, H)
        hidden_states = self.target_input(input_ids)

        # 2. Source site embedding: state token + species identity (same as target)
        #    source_ids (B, S, N) long → (B, S, N, H)
        species_idx = torch.arange(source_ids.size(1), device=source_ids.device)
        species_emb = self.target_input.species_embedding(species_idx)  # (S, H)
        source_emb = self.target_input.embedding(source_ids) + species_emb[None, :, None, :]

        # 3. Ecological context: source-site env groups + target site env as one extra token
        eco_emb = self.eco_source_module(env_data)                    # (B, C_eco, H)
        target_env_token = self.target_env_module(target_env)         # (B, 1, H)
        eco_emb = torch.cat([eco_emb, target_env_token], dim=1)       # (B, C_eco+1, H)

        # 4. Target-to-source distances via fancy indexing (avoids (B, T, N_total) intermediate)
        sp_pw = spatial_dist_pairwise.to(hidden_states.device)
        tp_pw = temporal_dist_pairwise.to(hidden_states.device)
        tgt_idx = target_site_idx[:, :, None]   # (B, T, 1)
        src_idx = source_idx[:, None, :]        # (B, 1, N)
        target_to_source_sp = sp_pw[tgt_idx, src_idx]  # (B, T, N)
        target_to_source_tp = tp_pw[tgt_idx, src_idx]  # (B, T, N)
        st_dist = torch.stack([target_to_source_sp, target_to_source_tp], dim=-1)  # (B, T, N, 2)

        # 5. Encode — no attention mask needed (blinding handled in collator)
        encoder_out = self.encoder(
            hidden_states, source_emb, eco_emb,
            st_attention_mask=None, st_dist=st_dist,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return encoder_out


# =============================================================================
# Masked prediction wrapper
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
