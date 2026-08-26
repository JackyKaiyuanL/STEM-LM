
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from stemlm.data import FixedPValCollator, seed_worker

# Species metrics are independent; cap the pool so a many-species run does
# not spawn hundreds of threads on a large node.
_METRIC_MAX_WORKERS = 32


def safe_auc_roc(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    if np.isnan(preds).any():
        return float("nan")
    return float(roc_auc_score(labels, preds))


def safe_auc_pr(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    if np.isnan(preds).any():
        return float("nan")
    return float(average_precision_score(labels, preds))


def auc_roc_and_pr(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    """AUROC and average precision from a single sort.

    Equivalent to ``safe_auc_roc`` and ``safe_auc_pr`` but shares one argsort and
    skips sklearn's curve construction: AUROC is the mid-rank Mann-Whitney
    statistic (identical to trapezoidal ROC integration) and AP is the step-wise
    sum over thresholds with tied scores grouped, as sklearn does.
    """
    n = labels.size
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n == 0 or n_pos == 0 or n_neg == 0 or np.isnan(preds).any():
        return float("nan"), float("nan")

    order = np.argsort(preds)
    p_sorted = preds[order]
    y_sorted = labels[order].astype(np.float64)

    # Contiguous runs of equal scores; both metrics treat a run as one threshold.
    new_run = np.empty(n, dtype=bool)
    new_run[0] = True
    np.not_equal(p_sorted[1:], p_sorted[:-1], out=new_run[1:])
    run_of = np.cumsum(new_run) - 1
    run_sizes = np.bincount(run_of)
    run_starts = np.concatenate(([0], np.cumsum(run_sizes)[:-1]))

    # AUROC: mean rank of the positives, ties sharing their average rank.
    mid_ranks = run_starts + (run_sizes + 1) / 2.0
    rank_sum = float(np.dot(mid_ranks[run_of], y_sorted))
    auroc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # AP: walk thresholds from the highest score down, one point per run.
    pos_upto = np.cumsum(y_sorted)[run_starts + run_sizes - 1]
    n_upto = run_starts + run_sizes
    tp = n_pos - np.concatenate(([0.0], pos_upto[:-1]))
    predicted = n - np.concatenate(([0], n_upto[:-1]))
    recall = np.concatenate((tp / n_pos, [0.0]))
    precision = np.concatenate((tp / predicted, [1.0]))
    ap = float(-np.sum(np.diff(recall) * precision[:-1]))
    return float(auroc), ap


def safe_brier(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or np.isnan(preds).any():
        return float("nan")
    return float(np.mean((preds - labels.astype(np.float64)) ** 2))


def safe_ece(labels: np.ndarray, preds: np.ndarray, n_bins: int = 15) -> float:
    if labels.size == 0 or np.isnan(preds).any():
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(preds, edges) - 1, 0, n_bins - 1)
    # Bin sums in one pass instead of one boolean mask per bin.
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    label_sums = np.bincount(idx, weights=labels.astype(np.float64), minlength=n_bins)
    pred_sums = np.bincount(idx, weights=preds.astype(np.float64), minlength=n_bins)
    nz = counts > 0
    gap = np.abs(label_sums[nz] - pred_sums[nz]) / counts[nz]
    return float(np.sum((counts[nz] / preds.size) * gap))


def safe_cbi(labels: np.ndarray, preds: np.ndarray,
             n_windows: int = 101, bin_width_frac: float = 0.1) -> float:
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    if np.isnan(preds).any():
        return float("nan")
    pres_preds = preds[labels == 1]
    all_preds = preds
    lo, hi = float(preds.min()), float(preds.max())
    if hi <= lo:
        return float("nan")
    half_w = 0.5 * bin_width_frac * (hi - lo)
    centers = np.linspace(lo, hi, n_windows)
    # Sort once, then every window's inclusive count is a pair of searchsorted
    # lookups — no rescan of the predictions per window.
    pres_sorted = np.sort(pres_preds)
    all_sorted = np.sort(all_preds)
    lo_i, hi_i = centers - half_w, centers + half_w
    e_count = (np.searchsorted(all_sorted, hi_i, side="right")
               - np.searchsorted(all_sorted, lo_i, side="left"))
    p_count = (np.searchsorted(pres_sorted, hi_i, side="right")
               - np.searchsorted(pres_sorted, lo_i, side="left"))
    pe = np.full(n_windows, np.nan, dtype=np.float64)
    hit = e_count > 0
    pe[hit] = ((p_count[hit] / pres_preds.size)
               / (e_count[hit] / all_preds.size))
    ok = np.isfinite(pe)
    if ok.sum() < 3 or np.unique(pe[ok]).size < 2:
        return float("nan")
    rho = spearmanr(centers[ok], pe[ok]).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _species_metrics(probs: np.ndarray, labels: np.ndarray, s: int):
    mask = labels[:, s] != -100
    y = labels[mask, s].astype(np.int64)
    p = probs[mask, s].astype(np.float64)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return None
    auc_roc, auc_pr = auc_roc_and_pr(y, p)
    return auc_roc, auc_pr, safe_cbi(y, p), safe_brier(y, p), safe_ece(y, p)


def compute_per_species_metrics(probs: np.ndarray,
                                labels: np.ndarray,
                                max_workers: int | None = None,
                                ) -> dict[str, dict[int, float]]:
    """Per-species metrics, computed across species in parallel.

    Species are independent, and the work is numpy sorts and reductions that
    release the GIL, so threads give real speedup without copying the (rows x
    species) arrays into worker processes.
    """
    if probs.shape != labels.shape:
        raise ValueError(f"probs {probs.shape} != labels {labels.shape}")
    S = probs.shape[1]
    if max_workers is None:
        max_workers = min(_METRIC_MAX_WORKERS, os.cpu_count() or 1, max(S, 1))

    if max_workers <= 1:
        results = [_species_metrics(probs, labels, s) for s in range(S)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda s: _species_metrics(probs, labels, s),
                                    range(S)))

    names = ("auc_roc", "auc_pr", "cbi", "brier", "ece")
    out: dict[str, dict[int, float]] = {name: {} for name in names}
    for s, res in enumerate(results):
        if res is None:
            continue
        for name, value in zip(names, res, strict=True):
            out[name][s] = value
    return out


def summarize_per_species_metrics(per_sp: dict[str, dict[int, float]]) -> dict[str, float]:
    def _clean(d):
        return [v for v in d.values() if np.isfinite(v)]
    aucs = _clean(per_sp.get("auc_roc", {}))
    prs = _clean(per_sp.get("auc_pr", {}))
    cbis = _clean(per_sp.get("cbi", {}))
    briers = _clean(per_sp.get("brier", {}))
    eces = _clean(per_sp.get("ece", {}))
    return {
        "mean_auc_roc": float(np.mean(aucs)) if aucs else float("nan"),
        "mean_auc_pr":  float(np.mean(prs)) if prs else float("nan"),
        "mean_cbi":     float(np.mean(cbis)) if cbis else float("nan"),
        "mean_brier":   float(np.mean(briers)) if briers else float("nan"),
        "mean_ece":     float(np.mean(eces)) if eces else float("nan"),
        "auc_roc_q25":  float(np.quantile(aucs, 0.25)) if aucs else float("nan"),
        "auc_roc_q50":  float(np.quantile(aucs, 0.50)) if aucs else float("nan"),
        "auc_roc_q75":  float(np.quantile(aucs, 0.75)) if aucs else float("nan"),
        "n_species":    len(aucs),
    }

def run_forward(model, batch, dist_info, device, output_attentions=False):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return model(
        input_ids=batch["input_ids"],
        source_ids=batch["source_ids"],
        source_idx=batch["source_idx"],
        target_site_idx=batch["target_site_idx"],
        env_data=batch["env_data"],
        target_env=batch["target_env"],
        labels=batch.get("labels"),
        site_lats=dist_info["site_lats"],
        site_lons=dist_info["site_lons"],
        site_times=dist_info["site_times"],
        euclidean=dist_info.get("euclidean", False),
        output_attentions=output_attentions,
    )


def _move_dist_info(dist_info, device):
    out = dict(dist_info)
    for k in ("site_lats", "site_lons", "site_times"):
        out[k] = out[k].to(device)
    return out


@torch.no_grad()
def bagged_evaluate_at_p(model, dataset, eval_indices, dist_info, p_value: float,
                         bag_K: int, batch_size: int, device,
                         num_workers: int = 0, base_seed: int = 0,
                         amp_dtype=None, distributed_sampler: bool = False,
                         collator_cls=None) -> dict:
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    dist_info_dev = _move_dist_info(dist_info, device)

    sum_probs: dict[int, np.ndarray] = {}
    label_for_idx: dict[int, np.ndarray] = {}

    is_distributed = bool(distributed_sampler) and torch.distributed.is_initialized()

    mask_seed = base_seed + round(p_value * 1000)
    cls = collator_cls if collator_cls is not None else FixedPValCollator
    collator = cls(p=p_value, base_seed=mask_seed)
    subset = Subset(dataset, eval_indices)

    for k in range(bag_K):
        np.random.seed(mask_seed + 7919 * k)
        torch.manual_seed(mask_seed + 7919 * k)
        if is_distributed:
            sampler = DistributedSampler(subset, shuffle=False)
            loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                                collate_fn=collator, num_workers=num_workers, pin_memory=True,
                                worker_init_fn=seed_worker)
        else:
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                collate_fn=collator, num_workers=num_workers, pin_memory=True,
                                worker_init_fn=seed_worker)

        for batch in loader:
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = run_forward(model, batch, dist_info_dev, device)
            else:
                out = run_forward(model, batch, dist_info_dev, device)
            probs = torch.sigmoid(out.logits.float().squeeze(-1)).cpu().numpy()
            labels = batch["labels"].squeeze(-1).cpu().numpy()
            target_idx = batch["target_site_idx"].squeeze(-1).cpu().numpy()
            for b, ti in enumerate(target_idx):
                ti = int(ti)
                if ti not in sum_probs:
                    sum_probs[ti] = probs[b].astype(np.float64)
                    label_for_idx[ti] = labels[b]
                else:
                    sum_probs[ti] += probs[b]

    if is_distributed:
        objs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(objs, (sum_probs, label_for_idx))
        merged_sum, merged_lab = {}, {}
        for sp, lb in objs:
            for ti, p in sp.items():
                if ti not in merged_sum:
                    merged_sum[ti] = p.copy()
                    merged_lab[ti] = lb[ti]
                else:
                    merged_sum[ti] += p
        sum_probs, label_for_idx = merged_sum, merged_lab

    indices = sorted(sum_probs.keys())
    if not indices:
        return {"p": float(p_value), "K": int(bag_K),
                "summary": summarize_per_species_metrics({}), "per_species": {}}

    avg_probs = np.stack([sum_probs[i] / bag_K for i in indices], axis=0)
    labels_arr = np.stack([label_for_idx[i] for i in indices], axis=0)
    per_sp_bag = compute_per_species_metrics(avg_probs, labels_arr)

    return {
        "p": float(p_value),
        "K": int(bag_K),
        "summary": summarize_per_species_metrics(per_sp_bag),
        "per_species": per_sp_bag,
    }


@torch.no_grad()
def gather_logits_at_p(model, dataset, eval_indices, dist_info, p_value: float,
                       batch_size: int, device,
                       num_workers: int = 0, base_seed: int = 0,
                       amp_dtype=None, distributed_sampler: bool = False,
                       collator_cls=None) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass forward at fixed mask rate `p_value`; returns
    (logits, labels) aligned by target_site_idx, both shape (N_eval, S).
    Used for Guo-style temperature scaling fitting and T-cal ECE evaluation.
    Distributed-safe (gathers across ranks)."""
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    dist_info_dev = _move_dist_info(dist_info, device)
    is_distributed = bool(distributed_sampler) and torch.distributed.is_initialized()

    mask_seed = base_seed + round(p_value * 1000)
    cls = collator_cls if collator_cls is not None else FixedPValCollator
    collator = cls(p=p_value, base_seed=mask_seed)
    subset = Subset(dataset, eval_indices)
    np.random.seed(mask_seed)
    torch.manual_seed(mask_seed)
    if is_distributed:
        sampler = DistributedSampler(subset, shuffle=False)
        loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                            collate_fn=collator, num_workers=num_workers, pin_memory=True,
                            worker_init_fn=seed_worker)
    else:
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            collate_fn=collator, num_workers=num_workers, pin_memory=True,
                            worker_init_fn=seed_worker)

    logits_by_idx: dict[int, np.ndarray] = {}
    labels_by_idx: dict[int, np.ndarray] = {}
    for batch in loader:
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = run_forward(model, batch, dist_info_dev, device)
        else:
            out = run_forward(model, batch, dist_info_dev, device)
        z = out.logits.float().squeeze(-1).cpu().numpy()
        labels = batch["labels"].squeeze(-1).cpu().numpy()
        target_idx = batch["target_site_idx"].squeeze(-1).cpu().numpy()
        for b, ti in enumerate(target_idx):
            ti = int(ti)
            logits_by_idx[ti] = z[b]
            labels_by_idx[ti] = labels[b]

    if is_distributed:
        objs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(objs, (logits_by_idx, labels_by_idx))
        merged_z, merged_lab = {}, {}
        for zd, ld in objs:
            merged_z.update(zd)
            merged_lab.update(ld)
        logits_by_idx, labels_by_idx = merged_z, merged_lab

    indices = sorted(logits_by_idx.keys())
    if not indices:
        S = dataset.num_species
        return np.zeros((0, S), dtype=np.float32), np.zeros((0, S), dtype=np.int64)
    logits_arr = np.stack([logits_by_idx[i] for i in indices], axis=0).astype(np.float32)
    labels_arr = np.stack([labels_by_idx[i] for i in indices], axis=0).astype(np.int64)
    return logits_arr, labels_arr


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray,
                    max_iter: int = 200) -> float:
    """Guo et al. 2017 §4.2: fit single positive scalar T* by minimizing BCE
    on masked positions of validation logits. Returns T* (float)."""
    mask = (val_labels != -100)
    z = torch.from_numpy(val_logits[mask]).float()
    y = torch.from_numpy(val_labels[mask].astype(np.float32))
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_T.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_T.exp().item())


def compute_per_species_ece_from_logits(logits: np.ndarray, labels: np.ndarray,
                                        T: float = 1.0, n_bins: int = 15) -> float:
    """Per-species mean ECE on T-scaled probabilities. Same convention as
    summarize_per_species_metrics(safe_ece(...)) — average across species
    after dropping species with no positive or no negative."""
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64) / T))
    eces = []
    for s in range(labels.shape[1]):
        m = labels[:, s] != -100
        y = labels[m, s].astype(np.int64)
        p = probs[m, s]
        if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
            continue
        eces.append(safe_ece(y, p, n_bins=n_bins))
    return float(np.mean(eces)) if eces else float("nan")
