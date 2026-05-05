suppressPackageStartupMessages({
  library(jsonlite)
  library(parallel)
})
N_CORES <- as.integer(Sys.getenv("N_CORES", unset = "32"))

args_all    <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args_all[grep("^--file=", args_all)])
SCRIPT_DIR  <- if (length(script_path)) dirname(normalizePath(script_path)) else "."
PROJECT_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."))

source(file.path(SCRIPT_DIR, "stemlm_metrics.R"))

DATASET     <- Sys.getenv("DATASET", unset = "ebutterfly")
results_dir <- file.path(PROJECT_DIR, DATASET, "results", "logistic_regression")
metrics_dir <- file.path(results_dir, "metrics")
dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)

COV_SETS <- c("env", "spatiotemporal", "full")
SPLITS   <- c("train", "val", "test")

load_predictions <- function(cov_set) {
  combined_files <- file.path(results_dir, cov_set,
                              paste0("predictions_", SPLITS, "_all.csv"))
  dat <- do.call(rbind, lapply(combined_files, read.csv, check.names = FALSE))
  if (!"cov_set" %in% names(dat)) dat$cov_set <- cov_set
  dat
}

compute_metrics <- function(cov_set) {
  dat     <- load_predictions(cov_set)
  species <- unique(dat$species)
  cat(sprintf("\nProcessing %s: %d species, %d rows  (mc.cores=%d)\n",
              cov_set, length(species), nrow(dat), N_CORES))
  # Pre-split once so worker doesn't carry the full data.frame.
  by_sp_split <- split(dat[, c("species","split","actual","predicted")],
                       list(dat$species, dat$split), drop = TRUE)
  rows <- mclapply(names(by_sp_split), function(key) {
    sub <- by_sp_split[[key]]
    if (nrow(sub) == 0) return(NULL)
    compute_per_species_row(sub$species[1], cov_set, sub$split[1],
                            sub$actual, sub$predicted)
  }, mc.cores = N_CORES, mc.preschedule = FALSE)
  do.call(rbind, rows[!sapply(rows, is.null)])
}

all_metrics <- vector("list", length(COV_SETS))
for (i in seq_along(COV_SETS)) {
  cs <- COV_SETS[i]
  if (!dir.exists(file.path(results_dir, cs))) {
    cat(sprintf("Skipping %s (no results dir)\n", cs)); next
  }
  met <- compute_metrics(cs)
  all_metrics[[i]] <- met
  write.csv(met, file.path(metrics_dir, paste0("metrics_", cs, ".csv")), row.names = FALSE)
}

all_met <- do.call(rbind, all_metrics)

summary_rows <- list(); idx <- 1L
for (cs in COV_SETS) {
  for (split in SPLITS) {
    sub <- all_met[all_met$cov_set == cs & all_met$split == split, ]
    if (nrow(sub) == 0) next
    s <- summarize_metrics(sub)
    s$cov_set <- cs; s$split <- split
    summary_rows[[idx]] <- s
    idx <- idx + 1L
  }
}
summary_df <- do.call(rbind, summary_rows)
summary_df <- summary_df[, c("cov_set", "split",
                             setdiff(names(summary_df), c("cov_set", "split")))]
write.csv(summary_df, file.path(metrics_dir, "metrics_summary.csv"), row.names = FALSE)

cat("\n--- Logistic regression: full STEM-LM metric set (mean across species) ---\n")
print(summary_df[, c("cov_set", "split", "n_species",
                     "mean_auc_roc", "auc_roc_q25", "auc_roc_q50", "auc_roc_q75",
                     "mean_auc_pr",
                     "mean_cbi", "mean_brier", "mean_ece")],
      row.names = FALSE, digits = 3)
cat("\nDone.\n")
