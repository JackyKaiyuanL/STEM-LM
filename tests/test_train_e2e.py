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
    "cooccurrence_matrix.npy",
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


def test_cooccurrence_matrix_shape(trained_run):
    matrix = np.load(trained_run / "cooccurrence_matrix.npy")
    assert matrix.shape == (len(SPECIES), len(SPECIES))


def test_training_log_has_two_epochs(trained_run):
    log = pd.read_csv(trained_run / "training_log.csv")
    assert len(log) == 2


def test_test_results_has_finite_metrics(trained_run):
    df = pd.read_csv(trained_run / "test_results.csv")
    assert len(df) > 0
    numeric = df.select_dtypes("number").to_numpy()
    assert np.isfinite(numeric).any()
