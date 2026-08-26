"""The fast metric paths must agree with the implementations they replaced.

compute_per_species_metrics is the per-epoch validation cost (7 min per pass at
71M-scale val), so AUROC/AP now share one sort instead of calling sklearn twice,
ECE bins with bincount instead of one mask per bin, CBI counts windows with
searchsorted instead of rescanning, and species run on a thread pool. Each is
checked here against a direct implementation of the original.
"""
import numpy as np
import pytest
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
    """The previous CBI: rescan both arrays for every window."""
    pres = preds[labels == 1]
    lo, hi = float(preds.min()), float(preds.max())
    half_w = 0.5 * bin_width_frac * (hi - lo)
    centers = np.linspace(lo, hi, n_windows)
    pe = np.full(n_windows, np.nan)
    for i, ctr in enumerate(centers):
        a, b = ctr - half_w, ctr + half_w
        e = ((preds >= a) & (preds <= b)).sum() / preds.size
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


def _per_species_serial_sklearn(probs, labels):
    """The previous per-species loop, sklearn metrics, single threaded."""
    from stemlm.metric import safe_brier
    S = probs.shape[1]
    out = {k: {} for k in ("auc_roc", "auc_pr", "cbi", "brier", "ece")}
    for s in range(S):
        mask = labels[:, s] != -100
        y = labels[mask, s].astype(np.int64)
        p = probs[mask, s].astype(np.float64)
        if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
            continue
        out["auc_roc"][s] = safe_auc_roc(y, p)
        out["auc_pr"][s] = safe_auc_pr(y, p)
        out["cbi"][s] = safe_cbi(y, p)
        out["brier"][s] = safe_brier(y, p)
        out["ece"][s] = safe_ece(y, p)
    return out


def test_per_species_metrics_match_serial_sklearn():
    rng = np.random.default_rng(11)
    n, S = 3000, 12
    labels = (rng.random((n, S)) < 0.08).astype(np.int64)
    probs = np.clip(0.4 * labels + rng.normal(0, 0.25, (n, S)), 0, 1)
    # a couple of species get partially masked, and one is degenerate
    labels[: n // 3, 2] = -100
    labels[:, 5] = 0
    got = compute_per_species_metrics(probs, labels)
    ref = _per_species_serial_sklearn(probs, labels)
    assert got.keys() == ref.keys()
    for name in ref:
        assert got[name].keys() == ref[name].keys(), name
        for s, v in ref[name].items():
            assert got[name][s] == pytest.approx(v, rel=1e-9, abs=1e-12), (name, s)


def test_per_species_metrics_thread_count_does_not_change_results():
    rng = np.random.default_rng(12)
    n, S = 1500, 9
    labels = (rng.random((n, S)) < 0.1).astype(np.int64)
    probs = np.clip(0.4 * labels + rng.normal(0, 0.3, (n, S)), 0, 1)
    serial = compute_per_species_metrics(probs, labels, max_workers=1)
    parallel = compute_per_species_metrics(probs, labels, max_workers=8)
    for name in serial:
        assert serial[name] == parallel[name], name
