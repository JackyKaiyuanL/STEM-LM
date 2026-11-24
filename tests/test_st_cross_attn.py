import math

import torch
import torch.nn.functional as F

from stemlm.model import JSDMConfig, JSDMModel, STColAttention, STCrossAttention


def _reference(mod, hidden_states, basis, source_ids, bias):
    q = mod.transpose_for_scores(mod.query(hidden_states))
    S = basis.size(1)
    flat = (source_ids.long() * S
            + torch.arange(S, device=source_ids.device)[None, :, None]).reshape(-1)
    emb = basis.reshape(-1, basis.size(-1))[flat].view(*source_ids.shape, -1)
    k = mod.transpose_for_scores(mod.key(emb))
    v = mod.transpose_for_scores(mod.value(emb))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(mod.attention_head_size) + bias
    return mod._merge_heads(torch.matmul(F.softmax(scores, dim=-1), v))


def test_full_model_forward_runs_with_gather():
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
    assert out.shape == (B, S, 1, cfg.hidden_size)
    out.sum().backward()
    grad = model.target_input.species_embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_collapsed_matches_explicit_reference():
    torch.manual_seed(3)
    cfg = JSDMConfig(hidden_size=64, num_attention_heads=8, num_species=10,
                     num_source_sites=7)
    mod = STCrossAttention(cfg).eval()  # eval => dropout off, deterministic
    B, S, N, T, H = 3, 10, 7, 1, 64
    hidden_states = torch.randn(B, S, T, H)
    basis = torch.randn(3, S, H)
    source_ids = torch.randint(0, 3, (B, S, N), dtype=torch.uint8)
    bias = torch.randn(B, S, 1, T, N)

    ref = _reference(mod, hidden_states, basis, source_ids, bias)
    got = mod(hidden_states, (basis, source_ids.long()), bias[:, :, 0])
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_collapsed_dropout_matches_explicit_reference():
    torch.manual_seed(4)
    cfg = JSDMConfig(hidden_size=64, num_attention_heads=8, num_species=10,
                     num_source_sites=7, attention_probs_dropout_prob=0.3)
    mod = STCrossAttention(cfg).train()
    B, S, N, T, H = 3, 10, 7, 2, 64
    h, p = mod.num_attention_heads, cfg.attention_probs_dropout_prob
    hidden_states = torch.randn(B, S, T, H)
    basis = torch.randn(3, S, H)
    source_ids = torch.randint(0, 3, (B, S, N), dtype=torch.uint8)
    bias = torch.randn(B, S, T, N)

    torch.manual_seed(9)
    got = mod(hidden_states, (basis, source_ids.long()), bias)
    torch.manual_seed(9)
    keep = torch.empty(B, S, T, h, N).bernoulli_(1 - p).transpose(2, 3)

    q = mod.transpose_for_scores(mod.query(hidden_states))
    flat = (source_ids.long() * S + torch.arange(S)[None, :, None]).reshape(-1)
    emb = basis.reshape(-1, H)[flat].view(B, S, N, H)
    k = mod.transpose_for_scores(mod.key(emb))
    v = mod.transpose_for_scores(mod.value(emb))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(mod.attention_head_size) + bias[:, :, None]
    ref = mod._merge_heads(torch.matmul(F.softmax(scores, dim=-1) * keep / (1 - p), v))
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_gate_threshold_scales_the_context():
    torch.manual_seed(5)
    cfg = JSDMConfig(hidden_size=32, num_attention_heads=4, num_species=5,
                     num_source_sites=9, temporal_fire_init_periods=(365.0,))
    mod = STColAttention(cfg).eval()
    B, S, N, H = 2, 5, 9, 32
    basis = torch.randn(3, S, H)
    source_ids = torch.randint(0, 2, (B, S, N))
    hidden_states = torch.randn(B, S, 1, H)
    st_dist = torch.stack([torch.rand(B, 1, N) * 100.0, torch.rand(B, 1, N) * 365.0], dim=-1)

    with torch.no_grad():
        mod.species_gate_threshold.fill_(-50.0)
    open_gate = mod(hidden_states, (basis, source_ids), st_dist)
    with torch.no_grad():
        mod.species_gate_threshold.fill_(50.0)
    closed_gate = mod(hidden_states, (basis, source_ids), st_dist)

    torch.testing.assert_close(closed_gate, mod.output(torch.zeros(B, S, 1, mod.output.dense.in_features)),
                               rtol=0, atol=1e-6)
    assert not torch.allclose(open_gate, closed_gate, atol=1e-6)


def test_gate_responds_to_a_uniform_bias_shift():
    torch.manual_seed(6)
    cfg = JSDMConfig(hidden_size=32, num_attention_heads=4, num_species=5,
                     num_source_sites=6, use_temporal=False)
    mod = STColAttention(cfg).eval()
    B, S, N = 2, 5, 6
    bias = torch.randn(B, S, 1, N)
    g_a = torch.sigmoid(mod.species_gate_threshold[None, :, None] - torch.logsumexp(bias, dim=-1))
    g_b = torch.sigmoid(mod.species_gate_threshold[None, :, None] - torch.logsumexp(bias - 3.0, dim=-1))
    assert (g_b > g_a).all()


def test_species_attention_matches_explicit_reference():
    from stemlm.model import SpeciesSelfAttention

    torch.manual_seed(6)
    cfg = JSDMConfig(hidden_size=64, num_attention_heads=8, num_species=17)
    mod = SpeciesSelfAttention(cfg).eval()  # dropout off
    B, T, S, H = 3, 2, 17, 64
    x = torch.randn(B, T, S, H)

    got = mod(x)

    q = mod.transpose_for_scores(mod.query(x))
    k = mod.transpose_for_scores(mod.key(x))
    v = mod.transpose_for_scores(mod.value(x))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(mod.attention_head_size)
    ctx = torch.matmul(F.softmax(scores, dim=-1), v).transpose(-2, -3).contiguous()
    ref = ctx.view(*ctx.size()[:-2], mod.all_head_size)

    assert got.shape == ref.shape == (B, T, S, mod.all_head_size)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)


def test_species_attention_independent_across_leading_dims():
    from stemlm.model import SpeciesSelfAttention

    torch.manual_seed(7)
    cfg = JSDMConfig(hidden_size=32, num_attention_heads=4, num_species=9)
    mod = SpeciesSelfAttention(cfg).eval()
    x = torch.randn(4, 3, 9, 32)
    full = mod(x)
    for b in range(4):
        for t in range(3):
            one = mod(x[b:b + 1, t:t + 1])
            torch.testing.assert_close(one[0, 0], full[b, t], rtol=1e-4, atol=1e-5)
