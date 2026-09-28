import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_EARTH_RADIUS_KM = 6371.0

def _haversine_bt_n(lat_a, lon_a, lat_b, lon_b):
    lat_a_r = torch.deg2rad(lat_a)
    lon_a_r = torch.deg2rad(lon_a)
    lat_b_r = torch.deg2rad(lat_b)
    lon_b_r = torch.deg2rad(lon_b)
    dlat = lat_a_r - lat_b_r
    dlon = lon_a_r - lon_b_r
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat_a_r) * torch.cos(lat_b_r) * torch.sin(dlon / 2.0) ** 2
    return _EARTH_RADIUS_KM * 2.0 * torch.asin(torch.sqrt(a.clamp(min=0.0)))


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

    temporal_fire_init_periods: tuple[float, ...] | None = None

    ablation: str = "full"

    per_species_env_rank: int = 8

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


class FIREDistanceBias(nn.Module):

    def __init__(self, max_dist: float, fire_hidden_size: int = 32,
                 n_frequencies: int = 0,
                 freq_init_periods: tuple[float, ...] | None = None):
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
            periods = torch.tensor([float(p) for p in freq_init_periods], dtype=torch.float32)
            if (periods <= 0).any():
                raise ValueError("all freq_init_periods must be > 0.")
            self.log_omega = nn.Parameter(torch.log(2.0 * math.pi / periods))
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class TargetInput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.embedding = nn.Embedding(3, config.hidden_size)
        self.species_embedding = nn.Embedding(config.num_species, config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        state_emb = self.embedding(input_ids)          # (B, S, T, H)
        species_emb = self.species_embedding.weight    # (S, H)
        return state_emb + species_emb[None, :, None, :]


class TargetEnvModule(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        E = config.num_env_vars
        self.proj1    = nn.Linear(E, config.hidden_size)
        self.act      = nn.SiLU()
        self.proj2    = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, target_env: torch.Tensor) -> torch.Tensor:
        x = self.proj1(target_env)
        x = self.proj2(self.act(x))
        return self.out_norm(x).unsqueeze(1)


class EnvSourceModule(nn.Module):

    def __init__(self, config: JSDMConfig):
        super().__init__()
        E = config.num_env_vars
        self.num_env_groups = config.num_env_groups
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
        site_emb = self.proj(env_data)  # (B, N, H)
        k = self.key_proj(site_emb)
        v = self.value_proj(site_emb)
        q = self.group_query.unsqueeze(0).expand(B, -1, -1)  # (B, C_env, H)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attn = F.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, v)
        return self.layer_norm(self.out_proj(pooled))


class Attention(nn.Module):
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
        new_x_shape = (*x.size()[:-1], self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).transpose(-2, -3)

    def _merge_heads(self, context):
        context = context.transpose(-2, -3).contiguous()
        return context.view(*context.size()[:-2], self.all_head_size)


class AttentionOutput(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size // 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        return self.dropout(self.dense(hidden_states))


class SpeciesSelfAttention(Attention):

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer   = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        scale = 1.0 / math.sqrt(self.attention_head_size)

        # q/k/v are (B, T, heads, S, head_dim). The fused attention kernels only
        # accept 4-D inputs, so a 5-D call silently falls back to the math backend
        # and materialises the S x S scores — 18 GB at S=3000, where it OOMs. Fold
        # the leading dims into the batch axis (attention is independent across
        # them) so flash/mem-efficient kernels apply: 3.4x faster at S=200, 6.8x
        # at S=1000, and S>=3000 becomes feasible at all.
        lead = query_layer.shape[:-3]
        q, k, v = (x.reshape(-1, *x.shape[-3:])
                   for x in (query_layer, key_layer, value_layer))
        context = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            scale=scale,
        )
        context = context.reshape(*lead, *context.shape[-3:])
        return self._merge_heads(context)


class SpeciesRowAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.self_attn = SpeciesSelfAttention(config)
        self.output = AttentionOutput(config)

    def forward(self, hidden_states):
        return self.output(self.self_attn(hidden_states))


class STCrossAttention(Attention):

    def forward(self, hidden_states, source_embeddings, st_dist_bias):
        """Attend over source sites without ever materialising K or V."""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        basis, source_ids = source_embeddings
        n_bins, S, _ = basis.shape
        h, hd = self.num_attention_heads, self.attention_head_size
        k_tab = self.key(basis).view(n_bins, S, h, hd)
        v_tab = self.value(basis).view(n_bins, S, h, hd)

        qk = torch.einsum("bshtd,ishd->bshti", query_layer, k_tab)
        qk = qk / math.sqrt(hd)

        p = self.attention_probs_dropout_prob
        with torch.autocast(device_type=st_dist_bias.device.type, enabled=False):
            in_bin = F.one_hot(source_ids.long(), n_bins).bool()[:, :, None]
            masked = st_dist_bias.float()[..., None].masked_fill(~in_bin, torch.finfo(torch.float32).min) # float32 minimum, since -inf gives NaN in exp(masked - log_mass) for a state with no source sites.
            log_mass = torch.logsumexp(masked, dim=-2)
            bins = F.softmax(qk.float() + log_mass[:, :, None], dim=-1)
            if self.training and p > 0:
                B, _, T, N = st_dist_bias.shape
                keep = torch.empty(B, S, T, h, N, device=masked.device).bernoulli_(1 - p)
                kept = torch.matmul(keep, torch.exp(masked - log_mass[..., None, :])).transpose(2, 3)
                bins = bins * kept / (1 - p)
        context = torch.einsum("bshti,ishd->bshtd", bins.to(v_tab.dtype), v_tab)
        return self._merge_heads(context)


class STColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = STCrossAttention(config)
        self.output = AttentionOutput(config)
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
            )
        self.species_spatial_log_scale = nn.Parameter(torch.zeros(config.num_species))
        if self.use_temporal:
            self.species_temporal_log_scale = nn.Parameter(torch.zeros(config.num_species))
        self.species_gate_threshold = nn.Parameter(torch.zeros(config.num_species))

    def forward(self, hidden_states, source_embeddings, st_dist):
        spatial_bias = self.fire_spatial(st_dist[..., 0])
        s_scale = F.softplus(self.species_spatial_log_scale) + 1e-4
        st_dist_bias = spatial_bias[:, None, :, :] * s_scale[None, :, None, None]

        if self.use_temporal:
            temporal_bias = self.fire_temporal(st_dist[..., 1])
            t_scale = F.softplus(self.species_temporal_log_scale) + 1e-4
            st_dist_bias = st_dist_bias + temporal_bias[:, None, :, :] * t_scale[None, :, None, None]

        gate = torch.sigmoid(self.species_gate_threshold[None, :, None]
                             - torch.logsumexp(st_dist_bias.float(), dim=-1))
        context = self.cross_attn(hidden_states, source_embeddings, st_dist_bias)
        return self.output(context * (1.0 - gate).unsqueeze(-1).to(context.dtype))


class EnvCrossAttention(Attention):

    def forward(self, hidden_states, env_embeddings):
        B, S, T, _ = hidden_states.shape
        h, hd = self.num_attention_heads, self.attention_head_size
        q = self.query(hidden_states).view(B, S * T, h, hd).transpose(1, 2)
        k = self.key(env_embeddings).view(B, -1, h, hd).transpose(1, 2)
        v = self.value(env_embeddings).view(B, -1, h, hd).transpose(1, 2)
        context = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,
            scale=1.0 / math.sqrt(hd),
        )
        return context.transpose(1, 2).reshape(B, S, T, self.all_head_size)


class EnvColAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.cross_attn = EnvCrossAttention(config)
        self.output = AttentionOutput(config)

    def forward(self, hidden_states, env_embeddings):
        return self.output(self.cross_attn(hidden_states, env_embeddings))


class JSDMAttention(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
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

    def forward(self, hidden_states, st_source_embeddings, env_embeddings, st_dist):
        h = hidden_states + self.row_attention(self.row_norm(hidden_states).transpose(-2, -3)).transpose(-2, -3)
        if self.use_st or self.use_env:
            h_normed = self.cross_norm(h)
        if self.use_st:
            h = h + self.st_col_attention(h_normed, st_source_embeddings, st_dist)
        if self.use_env:
            h = h + self.env_col_attention(h_normed, env_embeddings)
        return h


class FeedForward(nn.Module):

    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class JSDMLayer(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.attention = JSDMAttention(config)
        self.ffn = FeedForward(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, st_source_embeddings, env_embeddings, st_dist):
        h = self.attention(hidden_states, st_source_embeddings, env_embeddings, st_dist)
        return h + self.ffn(self.ffn_norm(h))


class JSDMEncoder(nn.Module):
    def __init__(self, config: JSDMConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [JSDMLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states, st_source_embeddings, env_embeddings, st_dist):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, st_source_embeddings, env_embeddings, st_dist,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, st_source_embeddings, env_embeddings, st_dist)
        return hidden_states


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
    ):
        hidden_states = self.target_input(input_ids)

        # The source K/V inputs would be (B, S, N, H) — hundreds of millions of
        # elements — but they only ever take 2*S distinct values, since
        # source_emb[b,s,n] = state_emb[source_ids[b,s,n]] + species_emb[s].
        # Hand cross-attention that small basis plus the ids and let it gather
        # after projecting, so the (B, S, N, H) product is never materialised.
        species_emb = self.target_input.species_embedding.weight        # (S, H)
        state_emb = self.target_input.embedding.weight
        source_basis = state_emb[:2, None, :] + species_emb[None, :, :]
        source_emb = (source_basis, source_ids.long())

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
        sp_dist = _haversine_bt_n(lat_t, lon_t, lat_s, lon_s)
        tp_dist = (ti_t - ti_s).abs()
        st_dist = torch.stack([sp_dist, tp_dist], dim=-1)

        return self.encoder(hidden_states, source_emb, env_emb, st_dist)


@dataclass
class JSDMOutput:
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None


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

    def forward(self, labels=None, loss_type: str = "bce",
                focal_alpha: float = 0.25, focal_gamma: float = 2.0, **kwargs):
        logits = self.cls(self.model(**kwargs))
        if self.per_species_env_head is not None:
            logits = logits + self.per_species_env_head(kwargs["target_env"]).unsqueeze(-1)

        loss = None
        if labels is not None:
            mask = (labels != -100).float()
            y = labels.clamp(min=0).float()
            x = logits.float()
            per_el = F.binary_cross_entropy_with_logits(x, y, reduction="none")
            if loss_type == "focal":
                p = torch.sigmoid(x)
                p_t = p * y + (1.0 - p) * (1.0 - y)
                per_el = per_el * (1.0 - p_t).clamp(min=0.0) ** focal_gamma
                if 0.0 <= focal_alpha <= 1.0:
                    per_el = per_el * (focal_alpha * y + (1.0 - focal_alpha) * (1.0 - y))
            loss = (per_el * mask).sum() / mask.sum()

        return JSDMOutput(loss=loss, logits=logits)
