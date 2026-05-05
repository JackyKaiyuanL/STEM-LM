suppressPackageStartupMessages({
  library(jsonlite)
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

results_dir <- file.path(PROJECT_DIR, DATASET, "results", "logistic_regression")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

COV_SETS <- c("env", "spatiotemporal", "full")
SPLITS   <- c("train", "val", "test")

# Fourier(DOY) at periods matching STEM-LM's FIRE basis.
DOY_PERIODS <- as.numeric(strsplit(
  Sys.getenv("DOY_PERIODS", unset = "365,182,122,91"), ","
)[[1]])

dat <- read.csv(DATA_FILE, check.names = FALSE)
has_time <- "time" %in% names(dat)
doy_cols <- character(0)
if (has_time) {
  doy <- as.POSIXlt(as.Date(dat$time))$yday + 1L
  has_time_var <- length(unique(doy)) > 1
  if (has_time_var) {
    for (P in DOY_PERIODS) {
      sn <- sprintf("doy_sin_%d", P); cn <- sprintf("doy_cos_%d", P)
      dat[[sn]] <- sin(2 * pi * doy / P)
      dat[[cn]] <- cos(2 * pi * doy / P)
      doy_cols <- c(doy_cols, sn, cn)
    }
  }
} else {
  has_time_var <- FALSE
}

splits    <- fromJSON(SPLITS_FILE)
train_idx <- splits$train + 1L
val_idx   <- splits$val   + 1L
test_idx  <- splits$test  + 1L

train_dat <- dat[train_idx, ]
val_dat   <- dat[val_idx,   ]
test_dat  <- dat[test_idx,  ]

env_cols <- grep("^env_", names(dat), value = TRUE)
non_sp   <- c("time", "latitude", "longitude", env_cols, doy_cols)
all_sp   <- setdiff(names(dat), non_sp)

cat(sprintf("Time encoding: %s\n",
            if (has_time_var) sprintf("Fourier(DOY) periods=%s", paste(DOY_PERIODS, collapse=","))
            else "absent"))
cat(sprintf("Data: %d rows  |  %d species  |  %d env features\n",
            nrow(dat), length(all_sp), length(env_cols)))
cat(sprintf("Split sizes -- train: %d  val: %d  test: %d\n",
            length(train_idx), length(val_idx), length(test_idx)))

.doy_term <- if (has_time_var) paste("+", paste(doy_cols, collapse = " + ")) else ""

writeLines(c(
  paste0("data_file=",   DATA_FILE),
  paste0("splits_file=", SPLITS_FILE),
  paste0("results_dir=", results_dir),
  paste0("cov_sets=",    paste(COV_SETS, collapse = " ")),
  paste0("n_species=",   length(all_sp)),
  paste0("n_env=",       length(env_cols)),
  paste0("time_encoding=", if (has_time_var)
                             sprintf("Fourier(DOY) periods=%s", paste(DOY_PERIODS, collapse=","))
                           else "absent"),
  paste0("run_date=",    format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
  "model_env=glm(y ~ env_*, binomial)",
  paste0("model_spatiotemporal=glm(y ~ latitude * longitude", .doy_term, ", binomial)"),
  paste0("model_full=glm(y ~ env_* + latitude * longitude", .doy_term, ", binomial)")
), file.path(results_dir, "run_params.txt"))

fit_one <- function(cov_set, y_tr, train_dat) {
  fml <- if (cov_set == "env") {
    as.formula(paste("y ~", paste(env_cols, collapse = " + ")))
  } else if (cov_set == "spatiotemporal") {
    as.formula(paste("y ~ latitude * longitude", .doy_term))
  } else {
    as.formula(paste("y ~", paste(env_cols, collapse = " + "),
                     "+ latitude * longitude", .doy_term))
  }
  m <- tryCatch(
    glm(fml, data = cbind(train_dat, y = y_tr), family = binomial(link = "logit"),
        control = glm.control(maxit = 100)),
    error = function(e) { warning(conditionMessage(e)); NULL }
  )
  if (is.null(m)) return(list(model = NULL, converged = FALSE, kind = "glm"))
  list(model = m, converged = isTRUE(m$converged), kind = "glm")
}

predict_resp <- function(fit, newdat) {
  as.numeric(predict(fit$model, newdata = newdat, type = "response"))
}

fit_predict_species <- function(sp, cs, out_dir) {
  y_tr <- train_dat[[sp]]
  y_va <- val_dat[[sp]]
  y_te <- test_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) {
    return(NULL)
  }
  fit <- fit_one(cs, y_tr, train_dat)
  if (is.null(fit$model)) {
    return(data.frame(species = sp, cov_set = cs,
                      converged = FALSE, kind = fit$kind,
                      stringsAsFactors = FALSE))
  }
  p_tr <- predict_resp(fit, train_dat)
  p_va <- predict_resp(fit, val_dat)
  p_te <- predict_resp(fit, test_dat)

  sp_safe <- gsub("[^A-Za-z0-9]", "_", sp)
  write.csv(data.frame(row_index = train_idx - 1L, species = sp, cov_set = cs,
                       split = "train", predicted = p_tr, actual = y_tr),
            file.path(out_dir, paste0(sp_safe, "_train.csv")), row.names = FALSE)
  write.csv(data.frame(row_index = val_idx - 1L, species = sp, cov_set = cs,
                       split = "val", predicted = p_va, actual = y_va),
            file.path(out_dir, paste0(sp_safe, "_val.csv")), row.names = FALSE)
  write.csv(data.frame(row_index = test_idx - 1L, species = sp, cov_set = cs,
                       split = "test", predicted = p_te, actual = y_te),
            file.path(out_dir, paste0(sp_safe, "_test.csv")), row.names = FALSE)

  data.frame(species = sp, cov_set = cs,
             converged = fit$converged, kind = fit$kind,
             stringsAsFactors = FALSE)
}

conv_log <- list()
for (cs in COV_SETS) {
  cat(sprintf("\n=== Cov set: %s (mc.cores=%d, n_species=%d) ===\n",
              cs, N_CORES, length(all_sp)))
  out_dir <- file.path(results_dir, cs, "per_species")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  t0 <- Sys.time()
  res_list <- mclapply(all_sp, fit_predict_species,
                       cs = cs, out_dir = out_dir,
                       mc.cores = N_CORES, mc.preschedule = FALSE)
  cat(sprintf("  Cov set %s done in %.1f min\n", cs,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))

  for (r in res_list) {
    if (!is.null(r)) conv_log[[length(conv_log) + 1L]] <- r
  }

  for (split in SPLITS) {
    files <- list.files(out_dir, pattern = paste0("_", split, "\\.csv$"), full.names = TRUE)
    if (length(files) == 0) next
    combined <- do.call(rbind, lapply(files, read.csv, check.names = FALSE))
    out_path <- file.path(results_dir, cs, paste0("predictions_", split, "_all.csv"))
    write.csv(combined, out_path, row.names = FALSE)
    cat(sprintf("  Combined %s -> %s\n", split, out_path))
  }
  unlink(out_dir, recursive = TRUE)
}

conv_df <- do.call(rbind, conv_log)
write.csv(conv_df, file.path(results_dir, "convergence_log.csv"), row.names = FALSE)

cat("\nConvergence summary:\n")
print(table(conv_df$cov_set, conv_df$converged, dnn = c("cov_set", "converged")))

cat("\nDone.\n")
