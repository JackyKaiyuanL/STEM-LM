suppressPackageStartupMessages({
  library(jsonlite)
  library(maxnet)
  library(parallel)
  library(pROC)
})

need <- function(name) {
  value <- Sys.getenv(name)
  if (!nzchar(value)) stop(name, " is not set")
  value
}
DATA_FILE   <- need("DATA_FILE")
SPLITS_FILE <- need("SPLITS_FILE")
results_dir <- need("RESULTS_DIR")
N_CORES     <- as.integer(Sys.getenv("N_CORES", unset = "8"))
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

SPLITS          <- c("train", "val", "test")
REG_MULT_VALUES <- c(1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32)

dat      <- read.csv(DATA_FILE, check.names = FALSE)
env_cols <- grep("^env_", names(dat), value = TRUE)
all_sp   <- readLines(need("SPECIES_FILE"))

splits    <- fromJSON(SPLITS_FILE)
idx       <- list(train = splits$train + 1L, val = splits$val + 1L, test = splits$test + 1L)
for (v in env_cols) dat[[v]][is.na(dat[[v]])] <- mean(dat[[v]][idx$train], na.rm = TRUE)
train_dat <- dat[idx$train, ]
val_dat   <- dat[idx$val, ]
features  <- env_cols[apply(train_dat[, env_cols, drop = FALSE], 2, var) > 0]

cat(sprintf("Data: %d rows | %d species | %d env (%d with variance) | train %d val %d test %d\n",
            nrow(dat), length(all_sp), length(env_cols), length(features),
            length(idx$train), length(idx$val), length(idx$test)))

val_auc <- function(labels, preds) {
  if (length(unique(labels)) < 2) return(NA_real_)
  as.numeric(roc(labels, preds, quiet = TRUE)$auc)
}

set.seed(42)
opt_species <- sample(all_sp, min(20L, length(all_sp)))
opt_jobs    <- expand.grid(rm = REG_MULT_VALUES, sp = opt_species, stringsAsFactors = FALSE)
opt_one <- function(i) {
  rm <- opt_jobs$rm[i]; sp <- opt_jobs$sp[i]
  y_tr <- train_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  model <- tryCatch(maxnet(p = y_tr, data = train_dat[, features, drop = FALSE], regmult = rm),
                    error = function(e) NULL)
  if (is.null(model)) return(NULL)
  p_va <- as.numeric(predict(model, newdata = val_dat[, features, drop = FALSE],
                             type = "logistic", clamp = TRUE))
  data.frame(reg_mult = rm, species = sp, auc_roc_val = val_auc(val_dat[[sp]], p_va))
}
opt <- do.call(rbind, mclapply(seq_len(nrow(opt_jobs)), opt_one,
                               mc.cores = N_CORES, mc.preschedule = FALSE))
write.csv(opt, file.path(results_dir, "reg_mult_optimization.csv"), row.names = FALSE)
agg     <- aggregate(auc_roc_val ~ reg_mult, data = opt, FUN = mean)
best_rm <- agg$reg_mult[which.max(agg$auc_roc_val)]
print(agg, row.names = FALSE, digits = 4)
cat(sprintf("Selected reg_mult %g by mean validation AUROC over %d species\n",
            best_rm, length(opt_species)))
if (best_rm %in% range(REG_MULT_VALUES)) cat("WARNING: selected reg_mult is at the edge of the grid\n")
writeLines(as.character(best_rm), file.path(results_dir, "best_reg_mult.txt"))

out_dir <- file.path(results_dir, "env", "per_species")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fit_one_species <- function(sp) {
  y_tr <- train_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  model <- tryCatch(maxnet(p = y_tr, data = train_dat[, features, drop = FALSE], regmult = best_rm),
                    error = function(e) NULL)
  if (is.null(model)) return(NULL)
  sp_safe <- gsub("[^A-Za-z0-9]", "_", sp)
  for (split in SPLITS) {
    rows <- dat[idx[[split]], ]
    write.csv(data.frame(row_index = idx[[split]] - 1L, species = sp, cov_set = "env", split = split,
                         logit = qlogis(as.numeric(predict(model, newdata = rows[, features, drop = FALSE],
                                                           type = "logistic", clamp = TRUE))),
                         actual = rows[[sp]]),
              file.path(out_dir, paste0(sp_safe, "_", split, ".csv")), row.names = FALSE)
  }
  TRUE
}
t0 <- Sys.time()
invisible(mclapply(all_sp, fit_one_species, mc.cores = N_CORES, mc.preschedule = FALSE))
for (split in SPLITS) {
  files <- list.files(out_dir, pattern = paste0("_", split, "\\.csv$"), full.names = TRUE)
  write.csv(do.call(rbind, lapply(files, read.csv, check.names = FALSE)),
            file.path(results_dir, "env", paste0("predictions_", split, "_all.csv")), row.names = FALSE)
}
unlink(out_dir, recursive = TRUE)
cat(sprintf("env: %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
