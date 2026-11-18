"""End-to-end: a real `stemlm train` run produces the expected artifacts."""
import json

import numpy as np
import pandas as pd

from stemlm.metric import compute_per_species_metrics, summarize_per_species_metrics

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
    "test_predictions.npz",
    "per_species_auc.csv",
    "ablation_summary.json",
    "run_info.json",
]


def test_all_artifacts_written(trained_run):
    missing = [f for f in EXPECTED_ARTIFACTS if not (trained_run / f).exists()]
    assert not missing, f"missing artifacts: {missing}"


def test_run_info_records_seed_and_wall_time(trained_run):
    info = json.loads((trained_run / "run_info.json").read_text())
    assert info["seed"] == 0 and info["wall_seconds"] > 0
    assert info["split"] == {"file": None, "resolution": 2, "seed": 0, "train_frac": 0.8, "test_frac": 0.1}


def test_predictions_file(trained_run):
    saved = np.load(trained_run / "test_predictions.npz")
    info = json.loads((trained_run / "run_info.json").read_text())
    fixed = (pd.read_csv(trained_run / "test_results.csv")
             .query("mask_scheme == 'uniform' and sources == 'heldout'").set_index("p"))
    for p in (0.5, 1.0):
        per_sp = compute_per_species_metrics(saved[f"logits_p{p:.2f}"].astype(np.float64) / saved["temperature"],
                                             saved[f"labels_p{p:.2f}"].astype(np.int64))
        np.testing.assert_allclose(summarize_per_species_metrics(per_sp)["mean_auc_roc"], fixed.loc[p, "auc"],
                                   rtol=0, atol=1e-12)
    assert saved["source_km"].shape == saved["source_days"].shape == (info["test_sites"], 8)
    assert len(saved["site_idx"]) == info["test_sites"] and set(info["test_seconds_by_p"]) == {"0.50", "1.00"}


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


def test_no_time_run_finishes_without_time_windows(no_time_run):
    assert (no_time_run / "test_results.csv").exists()
    assert not (no_time_run / "test_sweep.csv").exists()


def test_resume_keeps_training_log(resumed_run):
    assert len(pd.read_csv(resumed_run / "training_log.csv")) == 2


def test_resume_drops_log_rows_past_the_checkpoint(cli, tiny_csv, tmp_path):
    from conftest import TRAIN_ARGS

    args = ("train", str(tiny_csv), "--output_dir", str(tmp_path), *TRAIN_ARGS[2:], "--num_epochs")
    assert TRAIN_ARGS[:2] == ["--num_epochs", "2"]
    assert cli(*args, "1").returncode == 0
    log = tmp_path / "training_log.csv"
    first = log.read_text().splitlines()[1]
    log.write_text(log.read_text() + "2" + first[1:] + "\n2,0.5")
    (tmp_path / "test_results.csv").unlink()
    result = cli(*args, "2")
    assert result.returncode == 0, result.stderr
    assert pd.read_csv(log).epoch.tolist() == [1, 2]
    assert json.loads((tmp_path / "run_info.json").read_text())["resumed_at_epoch"] == 2


def test_test_results_has_finite_metrics(trained_run):
    df = pd.read_csv(trained_run / "test_results.csv")
    assert len(df) > 0
    numeric = df.select_dtypes("number").to_numpy()
    assert np.isfinite(numeric).any()
