suppressPackageStartupMessages({
  library(maxnet)
  library(jsonlite)
  library(pROC)
  library(parallel)
})

N_CORES <- as.integer(Sys.getenv("N_CORES", unset = "32"))
cat(sprintf("Using %d parallel workers\n", N_CORES))

args_all    <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args_all[grep("^--file=", args_all)])
SCRIPT_DIR  <- if (length(script_path)) dirname(normalizePath(script_path)) else "."
PROJECT_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."))
REPO_ROOT   <- normalizePath(file.path(PROJECT_DIR, "..", ".."))

DATASET     <- Sys.getenv("DATASET",     unset = "ebutterfly")
DATA_FILE   <- Sys.getenv("DATA_FILE",   unset = file.path(REPO_ROOT, "data", "ebutterfly_na_2011_2025.csv"))
SPLITS_FILE <- Sys.getenv("SPLITS_FILE", unset = file.path(REPO_ROOT, "data", "ebutterfly_splits.json"))

results_dir <- file.path(PROJECT_DIR, DATASET, "results", "maxnet")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

dat          <- read.csv(DATA_FILE, check.names = FALSE)
dat$time_num <- as.numeric(as.Date(dat$time))

splits    <- fromJSON(SPLITS_FILE)
train_idx <- splits$train + 1L
val_idx   <- splits$val   + 1L
test_idx  <- splits$test  + 1L

train_dat <- dat[train_idx, ]
val_dat   <- dat[val_idx,   ]
test_dat  <- dat[test_idx,  ]

env_cols <- grep("^env_", names(dat), value = TRUE)
non_sp   <- c("time", "latitude", "longitude", "time_num", env_cols)
all_sp   <- setdiff(names(dat), non_sp)

cat(sprintf("Data: %d rows  |  %d species  |  %d env features\n",
            nrow(dat), length(all_sp), length(env_cols)))
cat(sprintf("Split sizes -- train: %d  val: %d  test: %d\n",
            length(train_idx), length(val_idx), length(test_idx)))

feat_train_full <- train_dat[, env_cols, drop = FALSE]
feat_val_full   <- val_dat[,   env_cols, drop = FALSE]
feat_test_full  <- test_dat[,  env_cols, drop = FALSE]

nz_mask    <- apply(feat_train_full, 2, var) > 0
feat_train <- feat_train_full[, nz_mask, drop = FALSE]
feat_val   <- feat_val_full[,   nz_mask, drop = FALSE]
feat_test  <- feat_test_full[,  nz_mask, drop = FALSE]
cat(sprintf("Env features after removing zero-variance: %d\n", ncol(feat_train)))

safe_auc_roc <- function(labels, preds) {
  if (length(unique(labels)) < 2) return(NA_real_)
  tryCatch(as.numeric(roc(labels, preds, quiet = TRUE)$auc), error = function(e) NA_real_)
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

# PART 1: optimize reg_mult on 20 random species.
set.seed(42)
opt_species     <- sample(all_sp, 20)
reg_mult_values <- c(1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32)

cat("\n--- Optimizing reg_mult on", length(opt_species), "species ---\n")
cat(sprintf("Grid: %s\n", paste(reg_mult_values, collapse = ", ")))
cat("Selection criterion: MAX val AUC-ROC (averaged across species)\n\n")

opt_jobs <- list()
for (rm in reg_mult_values) for (sp in opt_species) {
  opt_jobs[[length(opt_jobs)+1L]] <- list(rm = rm, sp = sp)
}

opt_one <- function(job) {
  rm <- job$rm; sp <- job$sp
  y_tr <- train_dat[[sp]]; y_va <- val_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  model <- tryCatch(maxnet(p = y_tr, data = feat_train, regmult = rm),
                    error = function(e) NULL)
  if (is.null(model)) return(NULL)
  p_tr <- as.numeric(predict(model, newdata = feat_train, type = "logistic", clamp = TRUE))
  p_va <- as.numeric(predict(model, newdata = feat_val,   type = "logistic", clamp = TRUE))
  data.frame(
    reg_mult      = rm, species = sp,
    auc_roc_train = safe_auc_roc(y_tr, p_tr),
    auc_roc_val   = safe_auc_roc(y_va, p_va),
    cbi_train     = safe_cbi(y_tr,    p_tr),
    cbi_val       = safe_cbi(y_va,    p_va)
  )
}

t0 <- Sys.time()
results_list <- mclapply(opt_jobs, opt_one,
                         mc.cores = N_CORES, mc.preschedule = FALSE)
cat(sprintf("Reg-mult sweep done in %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
opt_results <- do.call(rbind, results_list[!sapply(results_list, is.null)])
write.csv(opt_results, file.path(results_dir, "reg_mult_optimization.csv"), row.names = FALSE)

agg <- do.call(rbind, lapply(reg_mult_values, function(rm) {
  sub <- opt_results[opt_results$reg_mult == rm, ]
  data.frame(
    reg_mult       = rm,
    n_species      = nrow(sub),
    mean_val_roc   = mean(sub$auc_roc_val, na.rm = TRUE),
    mean_val_cbi   = mean(sub$cbi_val,     na.rm = TRUE),
    mean_gap_roc   = mean(abs(sub$auc_roc_val - sub$auc_roc_train), na.rm = TRUE)
  )
}))

cat("\nReg-mult sweep summary:\n")
print(agg, row.names = FALSE, digits = 4)

best_rm <- agg$reg_mult[which.max(agg$mean_val_roc)]
cat(sprintf("\nSelected reg_mult by MAX val AUC-ROC: %d (val AUC-ROC = %.4f)\n",
            best_rm, max(agg$mean_val_roc, na.rm = TRUE)))
cat(sprintf("  (for ref) selection by MAX val CBI would give:       %d\n",
            agg$reg_mult[which.max(agg$mean_val_cbi)]))
cat(sprintf("  (for ref) selection by MIN val-train gap would give: %d\n",
            agg$reg_mult[which.min(agg$mean_gap_roc)]))

if (best_rm == max(reg_mult_values)) {
  cat(sprintf("\n*** WARNING: best reg_mult (%d) is at the TOP of the grid. ***\n", best_rm))
  cat("    Consider extending the grid further before declaring this final.\n")
}
if (best_rm == min(reg_mult_values)) {
  cat(sprintf("\n*** WARNING: best reg_mult (%d) is at the BOTTOM of the grid. ***\n", best_rm))
}

writeLines(as.character(best_rm), file.path(results_dir, "best_reg_mult.txt"))

# PART 2: fit all species at best reg_mult.
cat(sprintf("\n--- Running all %d species with reg_mult=%d ---\n", length(all_sp), best_rm))

out_dir <- file.path(results_dir, "env", "per_species")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fit_one_species <- function(sp) {
  y_tr <- train_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  model <- tryCatch(maxnet(p = y_tr, data = feat_train, regmult = best_rm),
                    error = function(e) NULL)
  if (is.null(model)) return(NULL)
  p_tr <- as.numeric(predict(model, newdata = feat_train, type = "logistic", clamp = TRUE))
  p_va <- as.numeric(predict(model, newdata = feat_val,   type = "logistic", clamp = TRUE))
  p_te <- as.numeric(predict(model, newdata = feat_test,  type = "logistic", clamp = TRUE))

  sp_safe <- gsub("[^A-Za-z0-9]", "_", sp)
  for (tag in list(list("train", train_idx, p_tr, y_tr),
                   list("val",   val_idx,   p_va, val_dat[[sp]]),
                   list("test",  test_idx,  p_te, test_dat[[sp]]))) {
    write.csv(
      data.frame(row_index = tag[[2]] - 1L, species = sp, cov_set = "env",
                 split = tag[[1]], predicted = tag[[3]], actual = tag[[4]]),
      file.path(out_dir, paste0(sp_safe, "_", tag[[1]], ".csv")),
      row.names = FALSE
    )
  }
  TRUE
}

t0 <- Sys.time()
invisible(mclapply(all_sp, fit_one_species,
                   mc.cores = N_CORES, mc.preschedule = FALSE))
cat(sprintf("All %d species fit at reg_mult=%d in %.1f min\n",
            length(all_sp), best_rm,
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

for (split in c("train", "val", "test")) {
  files    <- list.files(out_dir, pattern = paste0("_", split, "\\.csv$"), full.names = TRUE)
  if (length(files) == 0) next
  combined <- do.call(rbind, lapply(files, read.csv, check.names = FALSE))
  out_path <- file.path(results_dir, "env", paste0("predictions_", split, "_all.csv"))
  write.csv(combined, out_path, row.names = FALSE)
  cat(sprintf("Combined %s -> %s\n", split, out_path))
}
unlink(out_dir, recursive = TRUE)

cat("\nAll maxnet runs complete.\n")
