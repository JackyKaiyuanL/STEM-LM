suppressPackageStartupMessages({
  library(jsonlite)
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
DOY_PERIODS <- as.numeric(strsplit(Sys.getenv("DOY_PERIODS", unset = "365,182,122,91"), ",")[[1]])
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

COV_SETS <- c("env", "spatiotemporal", "full")
SPLITS   <- c("test")

dat      <- read.csv(DATA_FILE, check.names = FALSE)
env_cols <- grep("^env_", names(dat), value = TRUE)
all_sp   <- readLines(need("SPECIES_FILE"))
dat      <- cbind(dat, read.csv(need("AUTOCOV_FILE"), check.names = FALSE))
auto_train <- read.csv(need("AUTOCOV_TRAIN_FILE"), check.names = FALSE)
SOURCES  <- c("heldout", "train")

doy_cols <- character(0)
if ("time" %in% names(dat)) {
  doy <- as.POSIXlt(as.Date(dat$time))$yday + 1L
  if (length(unique(doy)) > 1) {
    for (P in DOY_PERIODS) {
      sn <- sprintf("doy_sin_%d", P); cn <- sprintf("doy_cos_%d", P)
      dat[[sn]] <- sin(2 * pi * doy / P)
      dat[[cn]] <- cos(2 * pi * doy / P)
      doy_cols <- c(doy_cols, sn, cn)
    }
  }
}

splits    <- fromJSON(SPLITS_FILE)
idx       <- list(train = splits$train + 1L, val = splits$val + 1L, test = splits$test + 1L)
for (v in env_cols) dat[[v]][is.na(dat[[v]])] <- mean(dat[[v]][idx$train], na.rm = TRUE)
train_dat <- dat[idx$train, ]

env_terms <- paste(env_cols, collapse = " + ")
st_terms  <- paste(c("latitude * longitude", doy_cols), collapse = " + ")
formulas  <- list(
  env            = paste("y ~ auto +", env_terms),
  spatiotemporal = paste("y ~ auto +", st_terms),
  full           = paste("y ~ auto +", env_terms, "+", st_terms)
)

cat(sprintf("Data: %d rows | %d species | %d env | train %d val %d test %d\n",
            nrow(dat), length(all_sp), length(env_cols),
            length(idx$train), length(idx$val), length(idx$test)))
writeLines(c(paste0("data_file=", DATA_FILE), paste0("splits_file=", SPLITS_FILE),
             paste0("model_", names(formulas), "=glm(", unlist(formulas), ", binomial)")),
           file.path(results_dir, "run_params.txt"))

fit_predict_species <- function(sp, cs, out_dir) {
  y_tr <- train_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  sp_safe <- gsub("[^A-Za-z0-9]", "_", sp)
  auto_col <- paste0("auto_", sp_safe)
  m <- glm(as.formula(formulas[[cs]]), data = cbind(train_dat, y = y_tr, auto = train_dat[[auto_col]]),
           family = binomial(link = "logit"), control = glm.control(maxit = 200))
  for (split in SPLITS) {
    rows <- dat[idx[[split]], ]
    for (src in SOURCES) {
      rows$auto <- if (src == "heldout") rows[[auto_col]] else auto_train[idx[[split]], auto_col]
      write.csv(data.frame(row_index = idx[[split]] - 1L, species = sp, cov_set = cs, split = split, sources = src,
                           logit = as.numeric(predict(m, newdata = rows, type = "link")),
                           actual = rows[[sp]]),
                file.path(out_dir, paste0(sp_safe, "_", split, "_", src, ".csv")), row.names = FALSE)
    }
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
    for (src in SOURCES) {
      files <- list.files(out_dir, pattern = paste0("_", split, "_", src, "\\.csv$"), full.names = TRUE)
      write.csv(do.call(rbind, lapply(files, read.csv, check.names = FALSE)),
                file.path(results_dir, cs, paste0("predictions_", split, "_", src, "_all.csv")), row.names = FALSE)
    }
  }
  unlink(out_dir, recursive = TRUE)
  cat(sprintf("%s: %.1f min\n", cs, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}
conv_df <- do.call(rbind, conv_log)
write.csv(conv_df, file.path(results_dir, "convergence_log.csv"), row.names = FALSE)
print(table(conv_df$cov_set, conv_df$converged, dnn = c("cov_set", "converged")))
