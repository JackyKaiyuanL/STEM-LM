"""Equivalence guard for the STCrossAttention source K/V gather.

The dense source embedding is (B, S, N, H) — hundreds of millions of elements at
training shapes — yet it only takes 3*S distinct values, because
    source_emb[b, s, n] = state_emb[source_ids[b, s, n]] + species_emb[s]
So cross-attention is handed the small (3, S, H) basis plus the ids and gathers
after projecting. This test pins that the gather path equals feeding the dense
tensor, so the optimisation can't silently change the model.
"""
import torch

from stemlm.model import JSDMConfig, JSDMModel, STCrossAttention


def _dense_source_emb(basis, source_ids):
    """The (B, S, N, H) tensor the gather path avoids building."""
    S = basis.size(1)
    flat = (source_ids.long() * S
            + torch.arange(S, device=source_ids.device)[None, :, None]).reshape(-1)
    return basis.reshape(-1, basis.size(-1))[flat].view(*source_ids.shape, -1)


def test_full_model_forward_runs_with_gather():
    """End-to-end: JSDMModel builds the basis internally and trains through it."""
    torch.manual_seed(2)
    S, N, E, NTOT = 8, 5, 3, 50
    cfg = JSDMConfig(num_species=S, num_source_sites=N, num_env_vars=E,
                     hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                     intermediate_size=32, num_env_groups=2)
    model = JSDMModel(cfg)
    B = 4
    out = model(
        input_ids=torch.randint(0, 3, (B, S, 1)),
        source_ids=torch.randint(0, 2, (B, S, N), dtype=torch.uint8),  # narrow, as collated
        source_idx=torch.randint(0, NTOT, (B, N)),
        target_site_idx=torch.randint(0, NTOT, (B, 1)),
        env_data=torch.randn(B, N, E),
        target_env=torch.randn(B, E),
        site_lats=torch.rand(NTOT) * 10 + 30,
        site_lons=torch.rand(NTOT) * 10 - 100,
        site_times=torch.rand(NTOT) * 100,
    )
    assert out.last_hidden_state.shape == (B, S, 1, cfg.hidden_size)
    out.last_hidden_state.sum().backward()
    grad = model.target_input.species_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_collapsed_matches_dense_projection():
    """The collapsed path (no K/V materialised) must equal the dense reference."""
    torch.manual_seed(3)
    cfg = JSDMConfig(hidden_size=64, num_attention_heads=8, num_species=10,
                     num_source_sites=7)
    mod = STCrossAttention(cfg).eval()  # eval => dropout off, deterministic
    B, S, N, T, H = 3, 10, 7, 1, 64
    hidden_states = torch.randn(B, S, T, H)
    basis = torch.randn(3, S, H)
    source_ids = torch.randint(0, 3, (B, S, N), dtype=torch.uint8)
    bias = torch.randn(B, S, 1, T, N)

    ref = mod(hidden_states, _dense_source_emb(basis, source_ids.long()),
              st_dist_bias=bias)[0]
    got = mod(hidden_states, (basis, source_ids.long()), st_dist_bias=bias)[0]
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_collapsed_matches_dense_attention_probs():
    """output_attentions must return the same probs as the dense path."""
    torch.manual_seed(4)
    cfg = JSDMConfig(hidden_size=32, num_attention_heads=4, num_species=6,
                     num_source_sites=5)
    mod = STCrossAttention(cfg).eval()
    B, S, N, T, H = 2, 6, 5, 1, 32
    hidden_states = torch.randn(B, S, T, H)
    basis = torch.randn(3, S, H)
    source_ids = torch.randint(0, 3, (B, S, N), dtype=torch.uint8)

    ref = mod(hidden_states, _dense_source_emb(basis, source_ids.long()),
              output_attentions=True)
    got = mod(hidden_states, (basis, source_ids.long()), output_attentions=True)
    for a, b in zip(ref, got, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)


def test_collapsed_no_source_dependence_on_absent_bins():
    """Sanity: bin weights sum to 1 across bins, so context is a convex mix."""
    torch.manual_seed(5)
    cfg = JSDMConfig(hidden_size=32, num_attention_heads=4, num_species=5,
                     num_source_sites=9)
    mod = STCrossAttention(cfg).eval()
    B, S, N, H = 2, 5, 9, 32
    # All sources absent => context must equal V[0, s] exactly.
    source_ids = torch.zeros(B, S, N, dtype=torch.uint8)
    basis = torch.randn(3, S, H)
    hidden_states = torch.randn(B, S, 1, H)
    got = mod(hidden_states, (basis, source_ids.long()))[0]
    v0 = mod.value(basis[0])  # (S, all_head)
    torch.testing.assert_close(got, v0[None, :, None, :].expand_as(got),
                               rtol=1e-4, atol=1e-5)


def test_species_attention_matches_explicit_reference():
    """Species self-attention folds (B, T) into the batch axis so the fused
    kernels apply; it must still equal an explicit softmax(QK^T)V."""
    import math

    import torch.nn.functional as F

    from stemlm.model import SpeciesSelfAttention

    torch.manual_seed(6)
    cfg = JSDMConfig(hidden_size=64, num_attention_heads=8, num_species=17)
    mod = SpeciesSelfAttention(cfg).eval()  # dropout off
    B, T, S, H = 3, 2, 17, 64
    x = torch.randn(B, T, S, H)

    got = mod(x)[0]

    q = mod.transpose_for_scores(mod.query(x))
    k = mod.transpose_for_scores(mod.key(x))
    v = mod.transpose_for_scores(mod.value(x))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(mod.attention_head_size)
    ctx = torch.matmul(F.softmax(scores, dim=-1), v).transpose(-2, -3).contiguous()
    ref = ctx.view(*ctx.size()[:-2], mod.all_head_size)

    assert got.shape == ref.shape == (B, T, S, mod.all_head_size)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_species_attention_independent_across_leading_dims():
    """Folding must not leak information between batch rows or target sites."""
    from stemlm.model import SpeciesSelfAttention

    torch.manual_seed(7)
    cfg = JSDMConfig(hidden_size=32, num_attention_heads=4, num_species=9)
    mod = SpeciesSelfAttention(cfg).eval()
    x = torch.randn(4, 3, 9, 32)
    full = mod(x)[0]
    for b in range(4):
        for t in range(3):
            one = mod(x[b:b + 1, t:t + 1])[0]
            torch.testing.assert_close(one[0, 0], full[b, t], rtol=1e-4, atol=1e-5)
