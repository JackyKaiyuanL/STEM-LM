"""End-to-end: a real `stemlm train` run produces the expected artifacts."""
import json

import numpy as np
import pandas as pd

# Mirrors the synthetic dataset built in conftest.py.
SPECIES = [f"species_{i}" for i in range(6)]

EXPECTED_ARTIFACTS = [
    "best_model.pt",
    "best_model_by_cbi.pt",
    "config.json",
    "species_names.json",
    "splits.json",
    "training_log.csv",
    "test_results.csv",
    "per_species_auc.csv",
    "ablation_summary.json",
]


def test_all_artifacts_written(trained_run):
    missing = [f for f in EXPECTED_ARTIFACTS if not (trained_run / f).exists()]
    assert not missing, f"missing artifacts: {missing}"


def test_config_matches_dataset(trained_run):
    config = json.loads((trained_run / "config.json").read_text())
    assert config["num_species"] == len(SPECIES)


def test_species_names_preserved(trained_run):
    names = json.loads((trained_run / "species_names.json").read_text())
    assert names == SPECIES


def test_training_log_has_two_epochs(trained_run):
    log = pd.read_csv(trained_run / "training_log.csv")
    assert len(log) == 2


def test_exclusion_sweep_covers_the_grid_and_matches_the_fixed_pair(trained_run):
    sweep = pd.read_csv(trained_run / "test_sweep.csv")
    pairs = set(zip(sweep["eval_exclusion_km"], sweep["eval_exclusion_days"], sweep["p"], strict=True))
    assert pairs == {(r, t, p) for r in (0, 500) for t in (0, 30) for p in (0.5, 1.0)}
    fixed = (pd.read_csv(trained_run / "test_results.csv")
             .query("mask_scheme == 'uniform' and sources == 'heldout'").set_index("p"))
    none = sweep.query("eval_exclusion_km == 0 and eval_exclusion_days == 0").set_index("p")
    np.testing.assert_allclose(none["auc"], fixed.loc[none.index, "auc"], rtol=0, atol=1e-12)


def test_test_results_has_finite_metrics(trained_run):
    df = pd.read_csv(trained_run / "test_results.csv")
    assert len(df) > 0
    numeric = df.select_dtypes("number").to_numpy()
    assert np.isfinite(numeric).any()
