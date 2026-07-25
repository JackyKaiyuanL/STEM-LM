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
    s_ar = torch.arange(S, device=source_ids.device)[None, :, None]
    flat = (source_ids * S + s_ar).reshape(-1)
    return basis.reshape(-1, basis.size(-1))[flat].view(*source_ids.shape, -1)


def test_gather_matches_dense_projection():
    torch.manual_seed(0)
    cfg = JSDMConfig(hidden_size=64, num_attention_heads=8, num_species=10,
                     num_source_sites=7)
    mod = STCrossAttention(cfg).eval()

    B, S, N, T, H = 3, 10, 7, 1, 64
    hidden_states = torch.randn(B, S, T, H)
    basis = torch.randn(3, S, H)
    source_ids = torch.randint(0, 3, (B, S, N))
    bias = torch.randn(B, S, 1, T, N)

    dense = _dense_source_emb(basis, source_ids)
    ref = mod(hidden_states, dense, st_dist_bias=bias)[0]
    got = mod(hidden_states, (basis, source_ids), st_dist_bias=bias)[0]

    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_gather_matches_dense_with_attentions():
    """The output_attentions path must agree too (it takes a different branch)."""
    torch.manual_seed(1)
    cfg = JSDMConfig(hidden_size=32, num_attention_heads=4, num_species=6,
                     num_source_sites=5)
    mod = STCrossAttention(cfg).eval()
    B, S, N, T, H = 2, 6, 5, 1, 32
    hidden_states = torch.randn(B, S, T, H)
    basis = torch.randn(3, S, H)
    source_ids = torch.randint(0, 3, (B, S, N))

    dense = mod(hidden_states, _dense_source_emb(basis, source_ids), output_attentions=True)
    gathered = mod(hidden_states, (basis, source_ids), output_attentions=True)
    for a, b in zip(dense, gathered, strict=True):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)


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
        source_ids=torch.randint(0, 2, (B, S, N)),
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
