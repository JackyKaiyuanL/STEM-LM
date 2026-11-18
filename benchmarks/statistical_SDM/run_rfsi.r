suppressPackageStartupMessages({
  library(jsonlite)
  library(parallel)
  library(ranger)
})

need <- function(name) {
  value <- Sys.getenv(name)
  if (!nzchar(value)) stop(name, " is not set")
  value
}
results_dir <- need("RESULTS_DIR")
N_CORES     <- as.integer(Sys.getenv("N_CORES", unset = "8"))
N_OBS       <- as.integer(need("N_OBS"))
HAS_TIME    <- need("HAS_TIME") == "1"
SHARES      <- need("SHARES") == "1"
SOURCES     <- need("AUTOCOV_SOURCES")
SEED        <- as.integer(Sys.getenv("SEED", unset = "42"))
DOY_PERIODS <- as.numeric(strsplit(Sys.getenv("DOY_PERIODS", unset = "365,182,122,91"), ",")[[1]])
NUM_TREES   <- 250L

dat      <- read.csv(need("DATA_FILE"), check.names = FALSE)
env_cols <- grep("^env_", names(dat), value = TRUE)
all_sp   <- readLines(need("SPECIES_FILE"))
prefix   <- need("NEIGHBOR_PREFIX")
all_idx  <- as.matrix(read.csv(paste0(prefix, "_idx.csv"), header = FALSE)) + 1L
all_dist <- as.matrix(read.csv(paste0(prefix, "_dist.csv"), header = FALSE))
all_gap  <- as.matrix(read.csv(paste0(prefix, "_gap.csv"), header = FALSE))
nb_idx   <- all_idx[, seq_len(N_OBS), drop = FALSE]
nb_dist  <- all_dist[, seq_len(N_OBS), drop = FALSE]
nb_gap   <- all_gap[, seq_len(N_OBS), drop = FALSE]
colnames(nb_dist) <- sprintf("dist%d", seq_len(N_OBS))
colnames(nb_gap)  <- sprintf("gap%d", seq_len(N_OBS))

inverse_weights <- function(d) {
  zero <- d == 0
  w <- ifelse(zero, 0, 1 / d)
  at_zero <- rowSums(zero) > 0
  w[at_zero, ] <- zero[at_zero, ] * 1
  w / rowSums(w)
}
w_dist <- inverse_weights(all_dist)
w_gap  <- if (HAS_TIME) inverse_weights(all_gap)

splits <- fromJSON(need("SPLITS_FILE"))
idx    <- list(train = splits$train + 1L, test = splits$test + 1L)
for (v in env_cols) dat[[v]][is.na(dat[[v]])] <- mean(dat[[v]][idx$train], na.rm = TRUE)

st <- cbind(latitude = dat$latitude, longitude = dat$longitude)
if (HAS_TIME) {
  doy <- as.POSIXlt(as.Date(dat$time))$yday + 1L
  for (P in DOY_PERIODS) {
    st <- cbind(st, sin(2 * pi * doy / P), cos(2 * pi * doy / P))
    colnames(st)[ncol(st) - 1:0] <- sprintf(c("doy_sin_%d", "doy_cos_%d"), P)
  }
}
neighbors <- if (HAS_TIME) cbind(nb_dist, nb_gap) else nb_dist
covariates <- list(env = cbind(as.matrix(dat[, env_cols]), neighbors),
                   full = cbind(as.matrix(dat[, env_cols]), st, neighbors))

cat(sprintf("Data: %d rows | %d species | %d env | train %d test %d | n.obs %d | trees %d\n",
            nrow(dat), length(all_sp), length(env_cols), length(idx$train), length(idx$test), N_OBS, NUM_TREES))
writeLines(c(paste0("data_file=", need("DATA_FILE")), paste0("splits_file=", need("SPLITS_FILE")),
             sprintf("model=ranger probability forest, %d trees, seed %d", NUM_TREES, SEED),
             paste0("inputs_", names(covariates), "=", vapply(covariates, function(m) paste(colnames(m), collapse = " "), ""),
                    sprintf(" obs1..obs%d", N_OBS), if (SHARES) " auto auto_idw" else "",
                    if (SHARES && HAS_TIME) " auto_itw" else "")),
           file.path(results_dir, "run_params.txt"))

fit_predict_species <- function(sp, cs, out_dir) {
  y <- dat[[sp]]
  obs <- matrix(y[nb_idx], nrow = length(y), ncol = N_OBS, dimnames = list(NULL, sprintf("obs%d", seq_len(N_OBS))))
  x <- cbind(covariates[[cs]], obs)
  if (SHARES) {
    sources <- matrix(y[all_idx], ncol = ncol(all_idx))
    x <- cbind(x, auto = rowMeans(sources), auto_idw = rowSums(w_dist * sources))
    if (HAS_TIME) x <- cbind(x, auto_itw = rowSums(w_gap * sources))
  }
  m <- ranger(x = x[idx$train, ], y = factor(y[idx$train], levels = c(0, 1)), probability = TRUE,
              num.trees = NUM_TREES, num.threads = 1, seed = SEED)
  p <- predict(m, data = x[idx$test, ], num.threads = 1)$predictions[, "1"]
  p <- pmin(pmax(p, 0.5 / NUM_TREES), 1 - 0.5 / NUM_TREES)
  write.csv(data.frame(row_index = idx$test - 1L, species = sp, cov_set = cs, split = "test", sources = SOURCES,
                       logit = qlogis(p), actual = y[idx$test]),
            file.path(out_dir, paste0(gsub("[^A-Za-z0-9]", "_", sp), ".csv")), row.names = FALSE)
  NULL
}

for (cs in names(covariates)) {
  out_dir <- file.path(results_dir, cs, "per_species")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  t0 <- Sys.time()
  res <- mclapply(all_sp, fit_predict_species, cs = cs, out_dir = out_dir, mc.cores = N_CORES, mc.preschedule = FALSE)
  bad <- vapply(res, inherits, logical(1), "try-error")
  if (any(bad)) stop(paste(all_sp[bad], unlist(res[bad]), sep = ": ", collapse = "\n"))
  files <- list.files(out_dir, full.names = TRUE)
  write.csv(do.call(rbind, lapply(files, read.csv, check.names = FALSE)),
            file.path(results_dir, cs, paste0("predictions_test_", SOURCES, "_all.csv")), row.names = FALSE)
  unlink(out_dir, recursive = TRUE)
  cat(sprintf("%s: %.1f min\n", cs, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}
