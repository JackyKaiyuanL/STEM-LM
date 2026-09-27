import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from stemlm.metric import (
    auc_roc_and_pr,
    compute_per_species_metrics,
    safe_auc_pr,
    safe_auc_roc,
    safe_cbi,
    safe_ece,
)


def _ece_maskwise(labels, preds, n_bins=15):
    """The previous ECE: one boolean mask per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(preds, edges) - 1, 0, n_bins - 1)
    err, n = 0.0, preds.size
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        err += (m.sum() / n) * abs(labels[m].mean() - preds[m].mean())
    return float(err)


def _cbi_looped(labels, preds, n_windows=101, bin_width_frac=0.1):
    u = (preds - preds.min()) / (preds.max() - preds.min())
    pres = u[labels == 1]
    half_w = 0.5 * bin_width_frac
    centers = np.linspace(0.0, 1.0, n_windows)
    pe = np.full(n_windows, np.nan)
    for i, ctr in enumerate(centers):
        a, b = ctr - half_w, ctr + half_w
        e = ((u >= a) & (u <= b)).sum() / u.size
        if e == 0:
            continue
        pe[i] = (((pres >= a) & (pres <= b)).sum() / pres.size) / e
    ok = np.isfinite(pe)
    return float(spearmanr(centers[ok], pe[ok]).statistic)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("prevalence", [0.5, 0.05, 0.005])
def test_auc_roc_and_pr_match_sklearn(seed, prevalence):
    rng = np.random.default_rng(seed)
    n = 4000
    y = (rng.random(n) < prevalence).astype(np.int64)
    if y.sum() in (0, n):
        pytest.skip("degenerate label vector")
    # signal + noise so scores are informative but not separable
    p = np.clip(0.5 * y + rng.normal(0, 0.3, n), 0, 1)
    got_roc, got_pr = auc_roc_and_pr(y, p)
    assert got_roc == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert got_pr == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_auc_handles_heavy_ties():
    """Tied scores are where mid-rank AUROC and grouped AP could diverge."""
    rng = np.random.default_rng(5)
    y = (rng.random(2000) < 0.2).astype(np.int64)
    p = np.round(rng.random(2000), 2)          # only 101 distinct values
    got_roc, got_pr = auc_roc_and_pr(y, p)
    assert got_roc == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert got_pr == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_auc_all_scores_identical():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    p = np.full(4, 0.7)
    got_roc, got_pr = auc_roc_and_pr(y, p)
    assert got_roc == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert got_pr == pytest.approx(average_precision_score(y, p), abs=1e-12)


@pytest.mark.parametrize("y", [
    np.zeros(50, dtype=np.int64),          # no positives
    np.ones(50, dtype=np.int64),           # no negatives
    np.array([], dtype=np.int64),          # empty
])
def test_auc_degenerate_returns_nan_like_safe_versions(y):
    p = np.full(y.size, 0.5)
    roc, pr = auc_roc_and_pr(y, p)
    assert np.isnan(roc) and np.isnan(pr)
    assert np.isnan(safe_auc_roc(y, p)) and np.isnan(safe_auc_pr(y, p))


def test_auc_nan_scores_return_nan():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    p = np.array([0.1, np.nan, 0.3, 0.9])
    roc, pr = auc_roc_and_pr(y, p)
    assert np.isnan(roc) and np.isnan(pr)


@pytest.mark.parametrize("seed", [0, 3])
def test_ece_matches_maskwise(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    y = (rng.random(n) < 0.1).astype(np.int64)
    p = rng.random(n)
    assert safe_ece(y, p) == pytest.approx(_ece_maskwise(y, p), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("seed", [0, 4])
def test_cbi_matches_looped(seed):
    rng = np.random.default_rng(seed)
    n = 6000
    y = (rng.random(n) < 0.08).astype(np.int64)
    p = np.clip(0.4 * y + rng.normal(0, 0.25, n), 0, 1)
    assert safe_cbi(y, p) == pytest.approx(_cbi_looped(y, p), rel=1e-12, abs=1e-15)


def _per_species_serial_sklearn(logits, labels):
    from stemlm.metric import safe_brier
    S = logits.shape[1]
    out = {k: {} for k in ("auc_roc", "auc_pr", "cbi", "brier", "ece")}
    for s in range(S):
        mask = labels[:, s] != -100
        y = labels[mask, s].astype(np.int64)
        z = logits[mask, s].astype(np.float64)
        if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
            continue
        p = expit(z)
        out["auc_roc"][s] = safe_auc_roc(y, p)
        out["auc_pr"][s] = safe_auc_pr(y, p)
        out["cbi"][s] = safe_cbi(y, z)
        out["brier"][s] = safe_brier(y, p)
        out["ece"][s] = safe_ece(y, p)
    return out


def _random_logits(rng, n, S, prevalence):
    labels = (rng.random((n, S)) < prevalence).astype(np.int64)
    return 2.0 * labels - 3.0 + rng.normal(0, 1.0, (n, S)), labels


def test_per_species_metrics_match_serial_sklearn():
    rng = np.random.default_rng(11)
    n = 3000
    logits, labels = _random_logits(rng, n, 12, 0.08)
    labels[: n // 3, 2] = -100
    labels[:, 5] = 0
    got = compute_per_species_metrics(logits, labels)
    ref = _per_species_serial_sklearn(logits, labels)
    assert got.keys() == ref.keys()
    for name in ref:
        assert got[name].keys() == ref[name].keys(), name
        for s, v in ref[name].items():
            assert got[name][s] == pytest.approx(v, rel=1e-9, abs=1e-12), (name, s)


def test_per_species_metrics_thread_count_does_not_change_results():
    rng = np.random.default_rng(12)
    logits, labels = _random_logits(rng, 1500, 9, 0.1)
    serial = compute_per_species_metrics(logits, labels, max_workers=1)
    parallel = compute_per_species_metrics(logits, labels, max_workers=8)
    for name in serial:
        assert serial[name] == parallel[name], name


@pytest.mark.parametrize("T", [0.5, 2.0, 3.7])
def test_ranking_and_cbi_invariant_to_temperature(T):
    rng = np.random.default_rng(13)
    logits, labels = _random_logits(rng, 4000, 10, 0.06)
    base = compute_per_species_metrics(logits, labels)
    scaled = compute_per_species_metrics(logits / T, labels)
    for name in ("auc_roc", "auc_pr", "cbi"):
        for s, v in base[name].items():
            assert scaled[name][s] == pytest.approx(v, rel=1e-12, abs=1e-12), (name, s)
    assert any(scaled["brier"][s] != v for s, v in base["brier"].items())
