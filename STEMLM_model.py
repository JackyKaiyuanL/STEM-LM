import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput


_EARTH_RADIUS_KM = 6371.0

def _haversine_bt_n(lat_a, lon_a, lat_b, lon_b):
    lat_a_r = torch.deg2rad(lat_a); lon_a_r = torch.deg2rad(lon_a)
    lat_b_r = torch.deg2rad(lat_b); lon_b_r = torch.deg2rad(lon_b)
    dlat = lat_a_r - lat_b_r
    dlon = lon_a_r - lon_b_r
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat_a_r) * torch.cos(lat_b_r) * torch.sin(dlon / 2.0) ** 2
    return _EARTH_RADIUS_KM * 2.0 * torch.asin(torch.sqrt(a.clamp(min=0.0)))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class JSDMConfig:
    
    num_species: int = 100
    
    num_source_sites: int = 64
    
    max_spatial_dist: float = 180.0
    max_temporal_dist: float = 365.0
    use_temporal: bool = True
    
    num_env_vars: int = 10
    num_env_groups: int = 5
    
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_hidden_layers: int = 4
    intermediate_size: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6

    fire_hidden_size: int = 32

    temporal_fire_init_periods: Optional[Tuple[float, ...]] = None
    fire_zero_init_periodic: bool = True

    ablation: str = "full"  # full | no_st | no_env | no_st_env

    per_species_env_rank: int = 8

    # Training — per-row mask rate (float in [0,1], "unif[:lo,hi]" for Uniform, or "beta:a,b" for Beta).
    p: "float | str" = 0.15

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
        if self.ablation not in ("full", "no_st", "no_env", "no_st_env"):
            raise ValueError(
                f"ablation must be one of full/no_st/no_env/no_st_env; got {self.ablation!r}"
            )
        
        tfp = self.temporal_fire_init_periods
        if tfp is not None:
            tfp = tuple(float(p) for p in tfp) or None
            self.temporal_fire_init_periods = tfp

    @property
    def n_temporal_fire_freqs(self) -> int:
        return len(self.temporal_fire_init_periods) if self.temporal_fire_init_periods else 0


# =============================================================================
# FIRE Distance Bias
# =============================================================================

class FIREDistanceBias(nn.Module):

    def __init__(self, max_dist: float, fire_hidden_size: int = 32,
                 n_frequencies: int = 0,
                 freq_init_periods: Optional[Tuple[float, ...]] = None,
                 zero_init_periodic: bool = True):
        super().__init__()
        self.log_c = nn.Parameter(torch.tensor(0.0))
        self.max_dist = max_dist
        self.n_frequencies = int(n_frequencies)
        in_dim = 1 + 2 * self.n_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, fire_hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(fire_hidden_size, 1, bias=False),
        )
        if self.n_frequencies > 0:
            if freq_init_periods is None or len(freq_init_periods) != self.n_frequencies:
                raise ValueError(
                    f"freq_init_periods must have exactly n_frequencies={self.n_frequencies} entries."
                )
            periods = torch.tensor([float(p) for p in freq_init_periods], dtype=torch.float32)
            if (periods <= 0).any():
                raise ValueError("all freq_init_periods must be > 0.")
            self.log_omega = nn.Parameter(torch.log(2.0 * math.pi / periods))
            if zero_init_periodic:
                with torch.no_grad():
                    self.mlp[0].weight[:, 1:].zero_()

    def forward(self, dist: torch.Tensor):
        d = dist.unsqueeze(-1).float()
        c = F.softplus(self.log_c) + 1e-4
        denom = torch.log1p(c * self.max_dist)
        base = torch.log1p(c * d) / denom
        if self.n_frequencies > 0:
            omega = torch.exp(self.log_omega)
            phase = d * omega
            feats = torch.cat([base, torch.cos(phase), torch.sin(phase)], dim=-1)
        else:
            feats = base
        return self.mlp(feats).squeeze(-1)



# =============================================================================
# RMSNorm
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
# State embedding: 3 tokens (0=absent, 1=present, 2=mask).
# Species identity embedding: learned per-species vector (S × H), added to
#   state embedding so every species has a distinct input representation
#   independent of its state.
#
#   No positional encoding since species ordering is arbitrary.
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
# Projects target site's own env vars into a single token appended to env_emb.
# =============================================================================

class TargetEnvModule(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        E = config.num_env_vars
        self.env_norm = nn.LayerNorm(E)
        self.proj1    = nn.Linear(E, config.hidden_size)
        self.act      = nn.SiLU()
        self.proj2    = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, target_env: torch.Tensor) -> torch.Tensor:
        x = self.proj1(self.env_norm(target_env))
        x = self.proj2(self.act(x))
        return self.out_norm(x).unsqueeze(1)


# =============================================================================
# Environmental Source Module
# =============================================================================

class EnvSourceModule(nn.Module):
    """
    Encodes environmental covariates from source sites, pooled into groups.
    Provides environmental context for cross-attention 2.
    """

    def __init__(self, config: JSDMConfig):
        super().__init__()
        E = config.num_env_vars
        self.num_env_groups = config.num_env_groups
        self.env_norm = nn.LayerNorm(E)
        self.proj = nn.Linear(E, config.hidden_size)
        self.group_query = nn.Parameter(
            torch.randn(config.num_env_groups, config.hidden_size) * 0.02
        )
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, env_data: torch.Tensor) -> torch.Tensor:
        B = env_data.size(0)
        site_emb = self.proj(self.env_norm(env_data))  # (B, N, H)
        k = self.key_proj(site_emb)
        v = self.value_proj(site_emb)
        q = self.group_query.unsqueeze(0).expand(B, -1, -1)  # (B, C_env, H)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, v)
        return self.layer_norm(self.out_proj(pooled))


# =============================================================================
# Row Self-Attention along species axis
# Species attend to each other — the cooccurrence term.
# =============================================================================

class SpeciesSelfAttention(nn.Module):

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
        output = (self.output(self_outputs[0]),)
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
        self.fire_spatial = FIREDistanceBias(
            config.max_spatial_dist, config.fire_hidden_size, n_frequencies=0,
        )
        if self.use_temporal:
            periods = config.temporal_fire_init_periods
            self.fire_temporal = FIREDistanceBias(
                config.max_temporal_dist, config.fire_hidden_size,
                n_frequencies=len(periods) if periods else 0,
                freq_init_periods=periods,
                zero_init_periodic=config.fire_zero_init_periodic,
            )
        self.species_spatial_log_scale = nn.Parameter(torch.zeros(config.num_species))
        if self.use_temporal:
            self.species_temporal_log_scale = nn.Parameter(torch.zeros(config.num_species))

    def forward(
        self, hidden_states, source_embeddings,
        attention_mask=None, st_dist=None, output_attentions=False,
    ):
        st_dist_bias = None
        if st_dist is not None:
            spatial_bias = self.fire_spatial(st_dist[..., 0])
            s_scale = F.softplus(self.species_spatial_log_scale) + 1e-4
            st_dist_bias = spatial_bias[:, None, :, :] * s_scale[None, :, None, None]

            if self.use_temporal:
                temporal_bias = self.fire_temporal(st_dist[..., 1])
                t_scale = F.softplus(self.species_temporal_log_scale) + 1e-4
                st_dist_bias = st_dist_bias + temporal_bias[:, None, :, :] * t_scale[None, :, None, None]

            st_dist_bias = st_dist_bias[:, :, None, :, :]

        cross_outputs = self.cross_attn(
            hidden_states, source_embeddings, attention_mask, st_dist_bias, output_attentions,
        )
        output = (self.output(cross_outputs[0]),)
        if output_attentions:
            output = output + (cross_outputs[1],)
        return output


# =============================================================================
# Environmental Cross-Attention
# =============================================================================

class EnvCrossAttention(nn.Module):
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

    def forward(self, hidden_states, env_embeddings, output_attentions=False):
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        env_exp = env_embeddings.unsqueeze(1).expand(-1, hidden_states.size(1), -1, -1)
        key_layer = self.transpose_for_scores(self.key(env_exp))
        value_layer = self.transpose_for_scores(self.value(env_exp))

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


class EnvCrossOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class EnvColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = EnvCrossAttention(config)
        self.output = EnvCrossOutput(config)

    def forward(self, hidden_states, env_embeddings, output_attentions=False):
        cross_outputs = self.cross_attn(hidden_states, env_embeddings, output_attentions)
        output = (self.output(cross_outputs[0]),)
        if output_attentions:
            output = output + (cross_outputs[1],)
        return output


# =============================================================================
# Combined Attention
#
# row_attention → (st_col + env_col) sum residual
# =============================================================================

class JSDMAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.ablation = config.ablation
        self.row_attention = SpeciesRowAttention(config)

        self.use_st  = config.ablation in ("full", "no_env")
        self.use_env = config.ablation in ("full", "no_st")
        if self.use_st:
            self.st_col_attention = STColAttention(config)
        if self.use_env:
            self.env_col_attention = EnvColAttention(config)

        self.row_norm  = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        if self.use_st or self.use_env:
            self.cross_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states, st_source_embeddings, env_embeddings,
        st_attention_mask=None, st_dist=None,
        output_attentions=False,
    ):
        # 1. Pre-norm → species self-attention → residual  (B,S,T,H) ↔ (B,T,S,H)
        row_output = self.row_attention(
            hidden_states=self.row_norm(hidden_states).transpose(-2, -3),
            output_attentions=output_attentions,
        )
        h = hidden_states + row_output[0].transpose(-2, -3)

        # 2 & 3. Shared pre-norm → ST and/or Env cross-attention → summed residual
        st_attn  = None
        env_attn = None

        if self.use_st or self.use_env:
            h_normed = self.cross_norm(h)

        if self.use_st and self.use_env:
            st_out  = self.st_col_attention(
                h_normed, st_source_embeddings, st_attention_mask, st_dist, output_attentions,
            )
            env_out = self.env_col_attention(h_normed, env_embeddings, output_attentions)
            h = h + st_out[0] + env_out[0]
            if output_attentions:
                st_attn, env_attn = st_out[1], env_out[1]
        elif self.use_st:
            st_out = self.st_col_attention(
                h_normed, st_source_embeddings, st_attention_mask, st_dist, output_attentions,
            )
            h = h + st_out[0]
            if output_attentions:
                st_attn = st_out[1]
        elif self.use_env:
            env_out = self.env_col_attention(h_normed, env_embeddings, output_attentions)
            h = h + env_out[0]
            if output_attentions:
                env_attn = env_out[1]
        # else (no_st_env): pure species-self block, skip cross-attention

        out = (h,)
        if output_attentions:
            out = out + (row_output[1] if len(row_output) > 1 else None,
                         st_attn, env_attn)
        return out


# =============================================================================
# FFN
# =============================================================================

class FeedForward(nn.Module):
    
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
        self, hidden_states, st_source_embeddings, env_embeddings,
        st_attention_mask=None, st_dist=None,
        output_attentions=False,
    ):
        attn_outputs = self.attention(
            hidden_states, st_source_embeddings, env_embeddings,
            st_attention_mask, st_dist, output_attentions,
        )
        h = attn_outputs[0]
        h = h + self.ffn(self.ffn_norm(h))

        return (h,) + attn_outputs[1:]


class JSDMEncoder(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [JSDMLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, st_source_embeddings, env_embeddings,
        st_attention_mask=None, st_dist=None,
        output_attentions=False, output_hidden_states=False,
    ):
        all_hidden = () if output_hidden_states else None
        all_sp_attn = () if output_attentions else None
        all_st_attn = () if output_attentions else None
        all_env_attn = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden = all_hidden + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, st_source_embeddings, env_embeddings,
                    st_attention_mask, st_dist, output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states, st_source_embeddings, env_embeddings,
                    st_attention_mask, st_dist, output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_sp_attn = all_sp_attn + (layer_outputs[1],)
                all_st_attn = all_st_attn + (layer_outputs[2],)
                all_env_attn = all_env_attn + (layer_outputs[3],)

        if output_hidden_states:
            all_hidden = all_hidden + (hidden_states,)

        return JSDMEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden,
            species_attentions=all_sp_attn,
            st_attentions=all_st_attn,
            env_attentions=all_env_attn,
        )


@dataclass
class JSDMEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    species_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    st_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    env_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# =============================================================================
# Full Model
# =============================================================================

class JSDMModel(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.config = config
        self.target_input = TargetInput(config)
        self.use_env = config.ablation in ("full", "no_st")
        if self.use_env:
            self.target_env_module = TargetEnvModule(config)
            self.env_source_module = EnvSourceModule(config)
        self.encoder = JSDMEncoder(config)

    def forward(
        self,
        input_ids,       # (B, S, T)  0=abs, 1=pres, 2=mask
        source_ids,      # (B, S, N)  0=abs, 1=pres
        source_idx,      # (B, N)
        target_site_idx, # (B, T)
        env_data,        # (B, N, E)
        target_env,      # (B, E)
        site_lats,       # (N_total,) deg
        site_lons,       # (N_total,) deg
        site_times,      # (N_total,) days
        euclidean=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        hidden_states = self.target_input(input_ids)

        species_emb = self.target_input.species_embedding.weight
        source_emb = self.target_input.embedding(source_ids) + species_emb[None, :, None, :]

        if self.use_env:
            env_emb = torch.cat([
                self.env_source_module(env_data),
                self.target_env_module(target_env),
            ], dim=1)
        else:
            env_emb = None

        lat_t = site_lats[target_site_idx][:, :, None]
        lon_t = site_lons[target_site_idx][:, :, None]
        ti_t  = site_times[target_site_idx][:, :, None]
        lat_s = site_lats[source_idx][:, None, :]
        lon_s = site_lons[source_idx][:, None, :]
        ti_s  = site_times[source_idx][:, None, :]
        if euclidean:
            sp_dist = torch.sqrt((lat_t - lat_s) ** 2 + (lon_t - lon_s) ** 2)
        else:
            sp_dist = _haversine_bt_n(lat_t, lon_t, lat_s, lon_s)
        tp_dist = (ti_t - ti_s).abs()
        st_dist = torch.stack([sp_dist, tp_dist], dim=-1)

        encoder_out = self.encoder(
            hidden_states, source_emb, env_emb,
            st_attention_mask=None, st_dist=st_dist,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return encoder_out


# =============================================================================
# Masked prediction
# =============================================================================

@dataclass
class JSDMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    species_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    st_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    env_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class PerSpeciesEnvHead(nn.Module):
    def __init__(self, num_env_vars: int, num_species: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(num_env_vars, rank))
        self.B = nn.Parameter(torch.empty(rank, num_species))
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.zeros(num_species))

    def forward(self, target_env: torch.Tensor) -> torch.Tensor:
        return target_env @ self.A @ self.B + self.bias


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
        if config.per_species_env_rank > 0 and config.ablation in ("full", "no_st"):
            self.per_species_env_head = PerSpeciesEnvHead(
                num_env_vars=config.num_env_vars,
                num_species=config.num_species,
                rank=config.per_species_env_rank,
            )
        else:
            self.per_species_env_head = None

    def forward(self, labels=None, loss_weight=None,
                loss_type: str = "bce",
                focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                output_attentions=False, **kwargs):
        target_env = kwargs.get("target_env", None)
        encoder_out = self.model(output_attentions=output_attentions, **kwargs)
        logits = self.cls(encoder_out.last_hidden_state)  # (B, S, T)
        if self.per_species_env_head is not None and target_env is not None:
            logits = logits + self.per_species_env_head(target_env).unsqueeze(-1)

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                if loss_type == "bce":
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
                elif loss_type == "focal":
                    # Sigmoid focal loss (Lin et al. 2017, RetinaNet form):
                    #   FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
                    #   p_t     = p if y==1 else 1-p
                    #   alpha_t = alpha if y==1 else 1-alpha   (alpha<0 disables alpha)
                    x = logits[mask].float()
                    y = labels[mask].float()
                    ce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
                    p = torch.sigmoid(x)
                    p_t = p * y + (1.0 - p) * (1.0 - y)
                    modulator = (1.0 - p_t).clamp(min=0.0) ** focal_gamma
                    if 0.0 <= focal_alpha <= 1.0:
                        alpha_t = focal_alpha * y + (1.0 - focal_alpha) * (1.0 - y)
                        per_el = alpha_t * modulator * ce
                    else:
                        per_el = modulator * ce
                    if loss_weight is None:
                        loss = per_el.mean()
                    else:
                        w = loss_weight.unsqueeze(-1).expand_as(labels)[mask]
                        loss = (per_el * w).sum() / w.sum()
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type!r} "
                                     f"(expected 'bce' or 'focal')")

        return JSDMOutput(
            loss=loss, logits=logits,
            hidden_states=encoder_out.hidden_states,
            species_attentions=encoder_out.species_attentions,
            st_attentions=encoder_out.st_attentions,
            env_attentions=encoder_out.env_attentions,
        )


# =============================================================================
# Extract cooccurrence matrix
# =============================================================================

def extract_cooccurrence_matrix(output: JSDMOutput, layer_idx: int = -1) -> torch.Tensor:
    if output.species_attentions is None:
        raise ValueError("Run with output_attentions=True")
    attn = output.species_attentions[layer_idx]
    return attn.squeeze(1).mean(dim=1)
