import pytest
import torch

from stemlm.model import JSDMConfig, JSDMForMaskedSpeciesPrediction

GROUPS = ("source_species", "source_env", "target_env", "positions", "times")
USED = {
    ("full", True): {"source_species", "source_env", "target_env", "positions", "times"},
    ("full", False): {"source_species", "source_env", "target_env", "positions"},
    ("no_source_env", True): {"source_species", "target_env", "positions", "times"},
    ("no_st", True): {"target_env"},
    ("no_env", True): {"source_species", "positions", "times"},
    ("no_st_env", True): set(),
}


def perturbed(inputs, group):
    out = dict(inputs)
    if group == "source_species":
        out["source_ids"] = 1 - inputs["source_ids"]
    elif group == "source_env":
        out["env_data"] = inputs["env_data"] + 1.0
    elif group == "target_env":
        out["target_env"] = inputs["target_env"] + 1.0
    elif group == "positions":
        out["site_lats"] = inputs["site_lats"].flip(0)
    else:
        out["site_times"] = inputs["site_times"].flip(0)
    return out


@pytest.mark.parametrize("ablation,use_temporal", sorted(USED))
def test_each_ablation_uses_only_its_inputs(ablation, use_temporal):
    torch.manual_seed(0)
    B, S, N, E, sites = 3, 6, 5, 4, 20
    config = JSDMConfig(num_species=S, num_source_sites=N, num_env_vars=E, hidden_size=32, num_attention_heads=4,
                        num_hidden_layers=2, intermediate_size=32, max_spatial_dist=5000.0, max_temporal_dist=3650.0,
                        temporal_fire_init_periods=(365.0,), ablation=ablation, use_temporal=use_temporal)
    model = JSDMForMaskedSpeciesPrediction(config).eval()
    inputs = {
        "input_ids": torch.randint(0, 3, (B, S, 1)),
        "source_ids": torch.randint(0, 2, (B, S, N)),
        "source_idx": torch.randint(0, sites, (B, N)),
        "target_site_idx": torch.randint(0, sites, (B, 1)),
        "env_data": torch.randn(B, N, E),
        "target_env": torch.randn(B, E),
        "site_lats": torch.empty(sites).uniform_(25.0, 50.0),
        "site_lons": torch.empty(sites).uniform_(-125.0, -65.0),
        "site_times": torch.empty(sites).uniform_(0.0, 3650.0),
    }
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.2)
        base = model(**inputs).logits
        assert torch.isfinite(base).all()
        for group in GROUPS:
            changed = not torch.equal(model(**perturbed(inputs, group)).logits, base)
            assert changed == (group in USED[ablation, use_temporal]), group
