import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from stemlm.data import FixedPValCollator, seed_worker

# Species metrics are independent; cap the pool so a many-species run does
# not spawn hundreds of threads on a large node.
_METRIC_MAX_WORKERS = 32
_METRIC_THREAD_MIN_ROWS = 25_000


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


def safe_cbi(labels: np.ndarray, logits: np.ndarray,
             n_windows: int = 101, bin_width_frac: float = 0.1) -> float:
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    if np.isnan(logits).any():
        return float("nan")
    lo, hi = float(logits.min()), float(logits.max())
    if hi <= lo:
        return float("nan")
    u = (logits - lo) / (hi - lo) # The extreme scores sit exactly on outer window edges; only this form keeps their membership fixed under z / T.
    half_w = 0.5 * bin_width_frac
    centers = np.linspace(0.0, 1.0, n_windows)
    pres_sorted = np.sort(u[labels == 1])
    all_sorted = np.sort(u)
    lo_i, hi_i = centers - half_w, centers + half_w
    e_count = (np.searchsorted(all_sorted, hi_i, side="right")
               - np.searchsorted(all_sorted, lo_i, side="left"))
    p_count = (np.searchsorted(pres_sorted, hi_i, side="right")
               - np.searchsorted(pres_sorted, lo_i, side="left"))
    pe = np.full(n_windows, np.nan, dtype=np.float64)
    hit = e_count > 0
    pe[hit] = ((p_count[hit] / pres_sorted.size)
               / (e_count[hit] / all_sorted.size))
    ok = np.isfinite(pe)
    if ok.sum() < 3 or np.unique(pe[ok]).size < 2:
        return float("nan")
    rho = spearmanr(centers[ok], pe[ok]).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _species_metrics(logits: np.ndarray, labels: np.ndarray, s: int):
    mask = labels[:, s] != -100
    y = labels[mask, s].astype(np.int64)
    z = logits[mask, s].astype(np.float64)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return None
    p = expit(z)
    auc_roc, auc_pr = auc_roc_and_pr(y, z)
    return auc_roc, auc_pr, safe_cbi(y, z), safe_brier(y, p), safe_ece(y, p)


def compute_per_species_metrics(logits: np.ndarray,
                                labels: np.ndarray,
                                max_workers: int | None = None,
                                ) -> dict[str, dict[int, float]]:
    S = logits.shape[1]
    if max_workers is None:
        max_workers = (1 if logits.shape[0] < _METRIC_THREAD_MIN_ROWS
                       else min(_METRIC_MAX_WORKERS, os.cpu_count() or 1, S))

    if max_workers <= 1:
        results = [_species_metrics(logits, labels, s) for s in range(S)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda s: _species_metrics(logits, labels, s),
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

def run_forward(model, batch, dist_info, **loss_kw):
    return model(
        input_ids=batch["input_ids"],
        source_ids=batch["source_ids"],
        source_idx=batch["source_idx"],
        target_site_idx=batch["target_site_idx"],
        env_data=batch["env_data"],
        target_env=batch["target_env"],
        site_lats=dist_info["site_lats"],
        site_lons=dist_info["site_lons"],
        site_times=dist_info["site_times"],
        **loss_kw,
    )


def move_dist_info_to_device(dist_info, device):
    out = dict(dist_info)
    for k in ("site_lats", "site_lons", "site_times"):
        out[k] = out[k].to(device)
    return out


@torch.no_grad()
def evaluate_at_p(model, dataset, eval_indices, dist_info, p_value: float,
                  batch_size: int, device,
                  num_workers: int = 0, base_seed: int = 0,
                  amp_dtype=None, distributed_sampler: bool = False,
                  collator_cls=None, temperature: float = 1.0) -> dict:
    z, y = gather_logits_at_p(model, dataset, eval_indices, dist_info, p_value, batch_size, device,
                              num_workers=num_workers, base_seed=base_seed, amp_dtype=amp_dtype,
                              distributed_sampler=distributed_sampler, collator_cls=collator_cls)
    per_sp = compute_per_species_metrics(z.astype(np.float64) / temperature, y)
    return {"p": float(p_value), "summary": summarize_per_species_metrics(per_sp), "per_species": per_sp}


def _mask_seed(base_seed: int, p_value: float) -> int:
    return base_seed + round(p_value * 1000)


def fixed_p_masks(eval_indices, num_species: int, p_value: float, base_seed: int,
                  batch_size: int) -> np.ndarray:
    collator = FixedPValCollator(p=p_value, base_seed=_mask_seed(base_seed, p_value))
    return np.concatenate([collator.draw(torch.as_tensor(eval_indices[s:s + batch_size]), num_species).numpy()
                           for s in range(0, len(eval_indices), batch_size)])


@torch.no_grad()
def gather_logits_at_p(model, dataset, eval_indices, dist_info, p_value: float,
                       batch_size: int, device,
                       num_workers: int = 0, base_seed: int = 0,
                       amp_dtype=None, distributed_sampler: bool = False,
                       collator_cls=None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    dist_info_dev = move_dist_info_to_device(dist_info, device)
    is_distributed = bool(distributed_sampler) and torch.distributed.is_initialized()

    mask_seed = _mask_seed(base_seed, p_value)
    collator = (collator_cls or FixedPValCollator)(p=p_value, base_seed=mask_seed)
    subset = Subset(dataset, eval_indices)
    np.random.seed(mask_seed)
    torch.manual_seed(mask_seed)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        sampler=DistributedSampler(subset, shuffle=False) if is_distributed else None,
                        collate_fn=collator, num_workers=num_workers, pin_memory=True,
                        worker_init_fn=seed_worker)

    logits_by_idx: dict[int, np.ndarray] = {}
    labels_by_idx: dict[int, np.ndarray] = {}
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = run_forward(model, batch, dist_info_dev)
        z = out.logits.float().squeeze(-1).cpu().numpy()
        labels = batch["labels"].squeeze(-1).cpu().numpy()
        for b, ti in enumerate(batch["target_site_idx"].squeeze(-1).cpu().numpy()):
            logits_by_idx[int(ti)] = z[b]
            labels_by_idx[int(ti)] = labels[b]

    if is_distributed:
        objs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(objs, (logits_by_idx, labels_by_idx))
        for zd, ld in objs:
            logits_by_idx.update(zd)
            labels_by_idx.update(ld)

    indices = sorted(logits_by_idx)
    return (np.stack([logits_by_idx[i] for i in indices]).astype(np.float32),
            np.stack([labels_by_idx[i] for i in indices]).astype(np.int64))


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
