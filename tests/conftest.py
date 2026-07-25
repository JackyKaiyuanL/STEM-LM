"""Shared fixtures for stemlm integration tests.

The heavy `trained_run` fixture invokes the real `stemlm train` CLI once (session
scope) on a tiny synthetic dataset, so every artifact assertion reuses one run.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

SPECIES = [f"species_{i}" for i in range(6)]
ENV_COLS = ["env_temp", "env_precip"]

# Small, fast, CPU-only training config. Enough rows/species for non-empty
# train/val/test under --fold random; tiny model + 2 epochs to stay quick.
TRAIN_ARGS = [
    "--num_epochs", "2",
    "--batch_size", "8",
    "--hidden_size", "32",
    "--num_attention_heads", "2",
    "--num_hidden_layers", "1",
    "--intermediate_size", "32",
    "--num_source_sites", "8",
    "--fold", "random",
    "--test_bag_K", "1",
    "--cooccurrence_extract_batches", "1",
    "--val_p_list", "0.5", "1.0",
    "--absence_mask_p_list", "0.5", "1.0",
    "--seed", "0",
]


def _write_synthetic_csv(path, n=96, seed=0):
    """Species presences loosely tied to env so metrics aren't degenerate."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(25.0, 55.0, n)
    lon = rng.uniform(-120.0, -70.0, n)
    time = rng.integers(0, 365, n)
    env = rng.normal(size=(n, len(ENV_COLS)))

    weights = rng.normal(size=(len(ENV_COLS), len(SPECIES)))
    prob = 1.0 / (1.0 + np.exp(-(env @ weights)))
    pres = (rng.uniform(size=prob.shape) < prob).astype(int)
    # Guarantee every species has at least one presence and one absence so
    # per-species AUROC is defined rather than NaN across the board.
    pres[0, :] = 1
    pres[1, :] = 0

    df = pd.DataFrame({"time": time, "latitude": lat, "longitude": lon})
    for k, col in enumerate(ENV_COLS):
        df[col] = env[:, k]
    for j, sp in enumerate(SPECIES):
        df[sp] = pres[:, j]
    df.to_csv(path, index=False)
    return path


def _run_cli(*args):
    """Invoke the CLI via the current interpreter so it uses the test venv."""
    return subprocess.run(
        [sys.executable, "-m", "stemlm.cli", *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def cli():
    """Callable that runs `stemlm <args>` and returns the CompletedProcess."""
    return _run_cli


@pytest.fixture(scope="session")
def tiny_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "tiny.csv"
    return _write_synthetic_csv(path)


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory, tiny_csv):
    """Run `stemlm train` once; return the output directory Path."""
    out_dir = tmp_path_factory.mktemp("run") / "out"
    result = _run_cli("train", str(tiny_csv), "--output_dir", str(out_dir), *TRAIN_ARGS)
    assert result.returncode == 0, (
        f"stemlm train exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return out_dir
