suppressPackageStartupMessages({
  library(jsonlite)
  library(mgcv)
  library(parallel)
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

COV_SETS <- c("env", "spatiotemporal", "full")
SPLITS   <- c("test")

dat      <- read.csv(DATA_FILE, check.names = FALSE)
env_cols <- grep("^env_", names(dat), value = TRUE)
all_sp   <- readLines(need("SPECIES_FILE"))

doy_term <- character(0)
if ("time" %in% names(dat)) {
  dat$doy <- as.POSIXlt(as.Date(dat$time))$yday + 1L
  if (length(unique(dat$doy)) > 1) doy_term <- "s(doy, bs = 'cc', k = 12)"
}

splits    <- fromJSON(SPLITS_FILE)
idx       <- list(train = splits$train + 1L, val = splits$val + 1L, test = splits$test + 1L)
for (v in env_cols) dat[[v]][is.na(dat[[v]])] <- mean(dat[[v]][idx$train], na.rm = TRUE)
train_dat <- dat[idx$train, ]

env_terms <- paste(vapply(env_cols, function(v) {
  k <- min(10L, length(unique(train_dat[[v]])) - 1L)
  if (k >= 3L) sprintf("s(%s, k = %d)", v, k) else v
}, character(1)), collapse = " + ")
st_terms <- paste(c("s(latitude, longitude, k = 50)", doy_term), collapse = " + ")
formulas <- list(
  env            = paste("y ~", env_terms),
  spatiotemporal = paste("y ~", st_terms),
  full           = paste("y ~", env_terms, "+", st_terms)
)

cat(sprintf("Data: %d rows | %d species | %d env | train %d val %d test %d\n",
            nrow(dat), length(all_sp), length(env_cols),
            length(idx$train), length(idx$val), length(idx$test)))
writeLines(c(paste0("data_file=", DATA_FILE), paste0("splits_file=", SPLITS_FILE),
             paste0("model_", names(formulas), "=bam(", unlist(formulas), ", binomial, discrete)")),
           file.path(results_dir, "run_params.txt"))

fit_predict_species <- function(sp, cs, out_dir) {
  y_tr <- train_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  m <- bam(as.formula(formulas[[cs]]), data = cbind(train_dat, y = y_tr),
           family = binomial(link = "logit"), discrete = TRUE, nthreads = 1)
  sp_safe <- gsub("[^A-Za-z0-9]", "_", sp)
  for (split in SPLITS) {
    rows <- dat[idx[[split]], ]
    write.csv(data.frame(row_index = idx[[split]] - 1L, species = sp, cov_set = cs, split = split,
                         logit = as.numeric(predict(m, newdata = rows, type = "link")),
                         actual = rows[[sp]]),
              file.path(out_dir, paste0(sp_safe, "_", split, ".csv")), row.names = FALSE)
  }
  data.frame(species = sp, cov_set = cs, converged = isTRUE(m$converged))
}

conv_log <- list()
for (cs in COV_SETS) {
  out_dir <- file.path(results_dir, cs, "per_species")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  t0 <- Sys.time()
  res <- mclapply(all_sp, fit_predict_species, cs = cs, out_dir = out_dir,
                  mc.cores = N_CORES, mc.preschedule = FALSE)
  bad <- vapply(res, inherits, logical(1), "try-error")
  if (any(bad)) stop(paste(all_sp[bad], unlist(res[bad]), sep = ": ", collapse = "\n"))
  conv_log <- c(conv_log, res[!sapply(res, is.null)])
  for (split in SPLITS) {
    files <- list.files(out_dir, pattern = paste0("_", split, "\\.csv$"), full.names = TRUE)
    write.csv(do.call(rbind, lapply(files, read.csv, check.names = FALSE)),
              file.path(results_dir, cs, paste0("predictions_", split, "_all.csv")), row.names = FALSE)
  }
  unlink(out_dir, recursive = TRUE)
  cat(sprintf("%s: %.1f min\n", cs, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}
conv_df <- do.call(rbind, conv_log)
write.csv(conv_df, file.path(results_dir, "convergence_log.csv"), row.names = FALSE)
print(table(conv_df$cov_set, conv_df$converged, dnn = c("cov_set", "converged")))
