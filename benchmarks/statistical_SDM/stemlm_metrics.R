# Shared metric helpers — match STEMLM_metric.py.
suppressPackageStartupMessages({
  library(pROC)
  library(PRROC)
})

safe_auc_roc <- function(labels, preds) {
  if (length(unique(labels)) < 2) return(NA_real_)
  if (any(is.na(preds))) return(NA_real_)
  tryCatch(as.numeric(roc(labels, preds, quiet = TRUE)$auc), error = function(e) NA_real_)
}

safe_auc_pr <- function(labels, preds) {
  if (sum(labels) == 0 || sum(labels) == length(labels)) return(NA_real_)
  if (any(is.na(preds))) return(NA_real_)
  tryCatch(pr.curve(scores.class0 = preds, weights.class0 = labels)$auc.integral,
           error = function(e) NA_real_)
}

safe_brier <- function(labels, preds) {
  if (length(labels) == 0 || any(is.na(preds))) return(NA_real_)
  mean((preds - as.numeric(labels))^2)
}

safe_ece <- function(labels, preds, n_bins = 15) {
  if (length(labels) == 0 || any(is.na(preds))) return(NA_real_)
  edges <- seq(0, 1, length.out = n_bins + 1)
  # Match np.clip(np.digitize - 1, 0, n_bins - 1) using 1-indexed findInterval.
  idx <- pmin(pmax(findInterval(preds, edges), 1L), n_bins)
  err <- 0
  n <- length(preds)
  for (b in seq_len(n_bins)) {
    m <- idx == b
    if (!any(m)) next
    err <- err + (sum(m) / n) * abs(mean(as.numeric(labels[m])) - mean(preds[m]))
  }
  err
}

# CBI: bg-only "expected" denominator, min_per_window floor — matches STEMLM_metric.safe_cbi.
safe_cbi <- function(labels, preds, n_windows = 101, width = 0.1, min_per_window = 10) {
  if (sum(labels) == 0 || sum(labels) == length(labels)) return(NA_real_)
  if (any(is.na(preds))) return(NA_real_)
  pres_preds <- preds[labels == 1]
  bg_preds   <- preds[labels == 0]
  if (length(bg_preds) == 0 || length(pres_preds) == 0) return(NA_real_)
  centers <- seq(0, 1, length.out = n_windows)
  half_w  <- width / 2
  pe <- vapply(centers, function(ctr) {
    n_bg <- sum(bg_preds >= (ctr - half_w) & bg_preds <= (ctr + half_w))
    if (n_bg < min_per_window) return(NA_real_)
    e_frac <- n_bg / length(bg_preds)
    if (e_frac == 0) return(NA_real_)
    p_frac <- sum(pres_preds >= (ctr - half_w) & pres_preds <= (ctr + half_w)) / length(pres_preds)
    p_frac / e_frac
  }, numeric(1))
  ok <- is.finite(pe)
  if (sum(ok) < 3 || length(unique(pe[ok])) < 2) return(NA_real_)
  tryCatch(cor(centers[ok], pe[ok], method = "spearman"), error = function(e) NA_real_)
}

compute_per_species_row <- function(species, cov_set, split, labels, preds) {
  data.frame(
    species = species, cov_set = cov_set, split = split,
    n = length(labels), n_pres = sum(labels), prevalence = mean(labels),
    auc_roc = safe_auc_roc(labels, preds),
    auc_pr  = safe_auc_pr(labels, preds),
    cbi   = safe_cbi(labels, preds),
    brier = safe_brier(labels, preds),
    ece   = safe_ece(labels, preds),
    stringsAsFactors = FALSE
  )
}

summarize_metrics <- function(rows) {
  q <- function(x, p) if (length(x) >= 1) quantile(x, p, na.rm = TRUE, names = FALSE) else NA_real_
  aucs <- rows$auc_roc[!is.na(rows$auc_roc)]
  data.frame(
    n_species          = sum(!is.na(rows$auc_roc)),
    mean_auc_roc       = mean(rows$auc_roc, na.rm = TRUE),
    auc_roc_q25        = q(aucs, 0.25),
    auc_roc_q50        = q(aucs, 0.50),
    auc_roc_q75        = q(aucs, 0.75),
    mean_auc_pr        = mean(rows$auc_pr, na.rm = TRUE),
    mean_cbi           = mean(rows$cbi,   na.rm = TRUE),
    mean_brier         = mean(rows$brier, na.rm = TRUE),
    mean_ece           = mean(rows$ece,   na.rm = TRUE),
    n_na_auc_roc       = sum(is.na(rows$auc_roc)),
    n_na_cbi           = sum(is.na(rows$cbi)),
    stringsAsFactors = FALSE
  )
}
