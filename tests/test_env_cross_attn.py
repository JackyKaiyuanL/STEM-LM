"""Equivalence guard for EnvCrossAttention.

The module's env key/value projections must not depend on the species axis S:
env is identical across all S rows, so projecting once and broadcasting is
mathematically the same as expanding to (B,S,C_env,H) and projecting S times —
but far cheaper. This test pins that equivalence by computing the reference the
naive expand-then-project way and asserting the module matches it, so any
project-then-expand refactor stays honest.
"""
import math

import torch
import torch.nn.functional as F

from stemlm.model import EnvCrossAttention, JSDMConfig


def _reference_forward(mod, hidden_states, env_embeddings):
    """Naive: expand env across S, THEN project (the original formulation)."""
    heads, hd = mod.num_attention_heads, mod.attention_head_size

    def tfs(x):
        return x.view(*x.size()[:-1], heads, hd).transpose(-2, -3)

    S = hidden_states.size(1)
    q = tfs(mod.query(hidden_states))
    env_exp = env_embeddings.unsqueeze(1).expand(-1, S, -1, -1)
    k = tfs(mod.key(env_exp))
    v = tfs(mod.value(env_exp))
    ctx = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=1.0 / math.sqrt(hd))
    ctx = ctx.transpose(-2, -3).contiguous()
    return ctx.view(*ctx.size()[:-2], mod.all_head_size)


def test_env_cross_attn_matches_expand_then_project():
    torch.manual_seed(0)
    cfg = JSDMConfig(hidden_size=64, num_attention_heads=8, num_species=12)
    mod = EnvCrossAttention(cfg).eval()  # eval => dropout off, deterministic

    B, S, T, H = 3, 12, 5, 64
    C_env = cfg.num_env_groups + 1  # env source groups + target env token
    hidden_states = torch.randn(B, S, T, H)
    env = torch.randn(B, C_env, H)

    ref = _reference_forward(mod, hidden_states, env)
    got = mod(hidden_states, env, output_attentions=False)[0]

    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
