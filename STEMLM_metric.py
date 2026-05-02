from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from STEMLM_data import FixedPValCollator

def safe_auc_roc(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        return float("nan")
    if np.isnan(preds).any():
        return float("nan")
    try:
        return float(roc_auc_score(labels, preds))
    except Exception:
        return float("nan")


def safe_auc_pr(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    if np.isnan(preds).any():
        return float("nan")
    try:
        return float(average_precision_score(labels, preds))
    except Exception:
        return float("nan")


def safe_brier(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0 or np.isnan(preds).any():
        return float("nan")
    return float(np.mean((preds - labels.astype(np.float64)) ** 2))


def safe_ece(labels: np.ndarray, preds: np.ndarray, n_bins: int = 15) -> float:
    if labels.size == 0 or np.isnan(preds).any():
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(preds, edges) - 1, 0, n_bins - 1)
    err = 0.0
    n = preds.size
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        err += (m.sum() / n) * abs(labels[m].mean() - preds[m].mean())
    return float(err)


def safe_cbi(labels: np.ndarray, preds: np.ndarray,
             n_windows: int = 101, width: float = 0.1,
             min_per_window: int = 10) -> float:
    # Continuous Boyce Index (Hirzel et al. 2006)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    if np.isnan(preds).any():
        return float("nan")
    pres_preds = preds[labels == 1]
    bg_preds = preds[labels == 0]
    if bg_preds.size == 0 or pres_preds.size == 0:
        return float("nan")
    centers = np.linspace(0.0, 1.0, n_windows)
    half_w = width / 2.0
    pe = np.full(n_windows, np.nan, dtype=np.float64)
    for i, ctr in enumerate(centers):
        lo, hi = ctr - half_w, ctr + half_w
        n_bg = int(((bg_preds >= lo) & (bg_preds <= hi)).sum())
        if n_bg < min_per_window:
            continue
        e_frac = n_bg / bg_preds.size
        if e_frac == 0:
            continue
        p_frac = ((pres_preds >= lo) & (pres_preds <= hi)).sum() / pres_preds.size
        pe[i] = p_frac / e_frac
    ok = np.isfinite(pe)
    if ok.sum() < 3 or np.unique(pe[ok]).size < 2:
        return float("nan")
    try:
        rho = spearmanr(centers[ok], pe[ok]).statistic
        return float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        return float("nan")


def compute_per_species_metrics(probs: np.ndarray,
                                labels: np.ndarray) -> Dict[str, Dict[int, float]]:
    if probs.shape != labels.shape:
        raise ValueError(f"probs {probs.shape} != labels {labels.shape}")
    S = probs.shape[1]
    out: Dict[str, Dict[int, float]] = {
        "auc_roc": {}, "auc_pr": {}, "cbi": {},
        "brier": {}, "ece": {},
    }
    for s in range(S):
        mask = labels[:, s] != -100
        y = labels[mask, s].astype(np.int64)
        p = probs[mask, s].astype(np.float64)
        if y.size == 0 or 0 == y.sum() or y.sum() == y.size:
            continue
        out["auc_roc"][s] = safe_auc_roc(y, p)
        out["auc_pr"][s] = safe_auc_pr(y, p)
        out["cbi"][s] = safe_cbi(y, p)
        out["brier"][s] = safe_brier(y, p)
        out["ece"][s] = safe_ece(y, p)
    return out


def summarize_per_species_metrics(per_sp: Dict[str, Dict[int, float]]) -> Dict[str, float]:
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
        if hasattr(out[k], "to"):
            out[k] = out[k].to(device)
    return out


@torch.no_grad()
def evaluate_loader(model, loader, device, dist_info, amp_dtype=None):
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    dist_info_dev = _move_dist_info(dist_info, device)

    total_loss, num_batches = 0.0, 0
    total_correct, total_masked = 0, 0
    probs_by_idx: Dict[int, np.ndarray] = {}
    labels_by_idx: Dict[int, np.ndarray] = {}

    for batch in loader:
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = run_forward(model, batch, dist_info_dev, device)
        else:
            out = run_forward(model, batch, dist_info_dev, device)

        if out.loss is not None:
            total_loss += out.loss.item()
            num_batches += 1

        probs = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
        labels = batch["labels"].squeeze(-1).cpu().numpy() if "labels" in batch \
                 else np.full_like(probs, fill_value=-100, dtype=np.int64)
        target_idx = batch["target_site_idx"].squeeze(-1).cpu().numpy()

        mask_arr = labels != -100
        total_correct += int(((probs > 0.5) == labels)[mask_arr].sum())
        total_masked += int(mask_arr.sum())

        for b, ti in enumerate(target_idx):
            ti = int(ti)
            probs_by_idx[ti] = probs[b]
            labels_by_idx[ti] = labels[b]

    indices = sorted(probs_by_idx.keys())
    if indices:
        probs_arr = np.stack([probs_by_idx[i] for i in indices], axis=0)
        labels_arr = np.stack([labels_by_idx[i] for i in indices], axis=0)
        per_sp = compute_per_species_metrics(probs_arr, labels_arr)
    else:
        per_sp = {}
    summary = summarize_per_species_metrics(per_sp)
    avg_loss = total_loss / max(num_batches, 1)
    acc = total_correct / max(total_masked, 1)
    return avg_loss, acc, summary, per_sp, probs_by_idx, labels_by_idx


@torch.no_grad()
def bagged_evaluate_at_p(model, dataset, eval_indices, dist_info, p_value: float,
                         bag_K: int, batch_size: int, device,
                         num_workers: int = 0, base_seed: int = 0,
                         amp_dtype=None, distributed_sampler: bool = False,
                         collator_cls=None) -> Dict:
    model.eval()
    use_amp = amp_dtype is not None and device.type == "cuda"
    dist_info_dev = _move_dist_info(dist_info, device)

    sum_probs: Dict[int, np.ndarray] = {}
    label_for_idx: Dict[int, np.ndarray] = {}
    single_pass_probs: Dict[int, np.ndarray] = {}

    is_distributed = bool(distributed_sampler) and torch.distributed.is_initialized()

    mask_seed = base_seed + int(round(p_value * 1000))
    cls = collator_cls if collator_cls is not None else FixedPValCollator
    collator = cls(
        p=p_value,
        site_lats=dist_info["site_lats"],
        site_lons=dist_info["site_lons"],
        site_times=dist_info["site_times"],
        spatial_scale_km=dist_info["spatial_scale_km"],
        euclidean=dist_info.get("euclidean", False),
        base_seed=mask_seed,
    )
    subset = Subset(dataset, eval_indices)

    for k in range(bag_K):
        np.random.seed(mask_seed + 7919 * k)
        torch.manual_seed(mask_seed + 7919 * k)
        if is_distributed:
            sampler = DistributedSampler(subset, shuffle=False)
            loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                                collate_fn=collator, num_workers=num_workers, pin_memory=True)
        else:
            loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                                collate_fn=collator, num_workers=num_workers, pin_memory=True)

        for batch in loader:
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = run_forward(model, batch, dist_info_dev, device)
            else:
                out = run_forward(model, batch, dist_info_dev, device)
            probs = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
            labels = batch["labels"].squeeze(-1).cpu().numpy()
            target_idx = batch["target_site_idx"].squeeze(-1).cpu().numpy()
            for b, ti in enumerate(target_idx):
                ti = int(ti)
                if ti not in sum_probs:
                    sum_probs[ti] = probs[b].astype(np.float64)
                    label_for_idx[ti] = labels[b]
                    if k == 0:
                        single_pass_probs[ti] = probs[b].copy()
                else:
                    sum_probs[ti] += probs[b]

    if is_distributed:
        objs = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(objs, (sum_probs, label_for_idx, single_pass_probs))
        merged_sum, merged_lab, merged_single = {}, {}, {}
        for sp, lb, sg in objs:
            for ti, p in sp.items():
                if ti not in merged_sum:
                    merged_sum[ti] = p.copy()
                    merged_lab[ti] = lb[ti]
                    if ti in sg:
                        merged_single[ti] = sg[ti]
                else:
                    merged_sum[ti] += p
        sum_probs, label_for_idx, single_pass_probs = merged_sum, merged_lab, merged_single

    indices = sorted(sum_probs.keys())
    if not indices:
        empty = {"mean_auc_roc": float("nan"), "mean_auc_pr": float("nan"),
                 "mean_cbi": float("nan"),
                 "mean_brier": float("nan"), "mean_ece": float("nan"),
                 "auc_roc_q25": float("nan"), "auc_roc_q50": float("nan"),
                 "auc_roc_q75": float("nan"),
                 "n_species": 0}
        return {"p": p_value, "K": bag_K, "summary": empty, "per_species": {},
                "single_pass_summary": empty}

    avg_probs = np.stack([sum_probs[i] / bag_K for i in indices], axis=0)
    labels_arr = np.stack([label_for_idx[i] for i in indices], axis=0)
    per_sp_bag = compute_per_species_metrics(avg_probs, labels_arr)
    summary_bag = summarize_per_species_metrics(per_sp_bag)

    if single_pass_probs:
        sp_indices = sorted(single_pass_probs.keys())
        sp_probs = np.stack([single_pass_probs[i] for i in sp_indices], axis=0)
        sp_labels = np.stack([label_for_idx[i] for i in sp_indices], axis=0)
        summary_single = summarize_per_species_metrics(
            compute_per_species_metrics(sp_probs, sp_labels)
        )
    else:
        summary_single = summary_bag

    return {
        "p": float(p_value),
        "K": int(bag_K),
        "summary": summary_bag,
        "per_species": per_sp_bag,
        "single_pass_summary": summary_single,
    }
