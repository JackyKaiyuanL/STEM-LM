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
results_dir <- file.path(PROJECT_DIR, DATASET, "results", "maxnet")
metrics_dir <- file.path(results_dir, "metrics")
dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)

SPLITS <- c("train", "val", "test")

load_predictions <- function() {
  combined_files <- file.path(results_dir, "env",
                              paste0("predictions_", SPLITS, "_all.csv"))
  do.call(rbind, lapply(combined_files, read.csv, check.names = FALSE))
}

cat("Loading maxnet predictions...\n")
dat <- load_predictions()
if (!"cov_set" %in% names(dat)) dat$cov_set <- "env"
species <- unique(dat$species)
cat(sprintf("%d species, %d total rows\n", length(species), nrow(dat)))

cat(sprintf("Computing per-species metrics across %d cores...\n", N_CORES))
by_sp_split <- split(dat[, c("species","split","actual","predicted")],
                     list(dat$species, dat$split), drop = TRUE)
rows <- mclapply(names(by_sp_split), function(key) {
  sub <- by_sp_split[[key]]
  if (nrow(sub) == 0) return(NULL)
  compute_per_species_row(sub$species[1], "env", sub$split[1],
                          sub$actual, sub$predicted)
}, mc.cores = N_CORES, mc.preschedule = FALSE)
met <- do.call(rbind, rows[!sapply(rows, is.null)])
write.csv(met, file.path(metrics_dir, "metrics_env.csv"), row.names = FALSE)

summary_rows <- list(); idx <- 1L
for (split in SPLITS) {
  sub <- met[met$split == split, ]
  if (nrow(sub) == 0) next
  s <- summarize_metrics(sub)
  s$cov_set <- "env"; s$split <- split
  summary_rows[[idx]] <- s
  idx <- idx + 1L
}
summary_df <- do.call(rbind, summary_rows)
summary_df <- summary_df[, c("cov_set", "split",
                             setdiff(names(summary_df), c("cov_set", "split")))]
write.csv(summary_df, file.path(metrics_dir, "metrics_summary.csv"), row.names = FALSE)

cat("\n--- MaxNet: full STEM-LM metric set (mean across species) ---\n")
print(summary_df[, c("cov_set", "split", "n_species",
                     "mean_auc_roc", "auc_roc_q25", "auc_roc_q50", "auc_roc_q75",
                     "mean_auc_pr",
                     "mean_cbi", "mean_brier", "mean_ece")],
      row.names = FALSE, digits = 3)
cat("\nDone.\n")
