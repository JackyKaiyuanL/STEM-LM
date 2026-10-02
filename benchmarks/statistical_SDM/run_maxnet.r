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

SPLITS          <- c("test")
REG_MULT_VALUES <- c(0.125, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32)

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

design_matrix <- function(f, data) {
  tt <- terms(f)
  vars <- as.list(attr(tt, "variables"))[-1]
  fac <- attr(tt, "factors")
  labels <- attr(tt, "term.labels")
  env <- environment(f)
  width <- vapply(vars, function(v) NCOL(eval(v, data[1, , drop = FALSE], env)), 1L)
  term_width <- vapply(seq_along(labels), function(j) as.integer(prod(width[fac[, j] > 0])), 1L)
  mm <- matrix(0, nrow(data), sum(term_width))
  names_out <- character(ncol(mm))
  at <- 0L
  for (j in seq_along(labels)) {
    used <- which(fac[, j] > 0)
    block <- eval(vars[[used[1]]], data, env)
    for (u in used[-1]) block <- block * eval(vars[[u]], data, env)
    cols <- at + seq_len(term_width[j])
    mm[, cols] <- block
    names_out[cols] <- if (is.matrix(block)) paste0(labels[j], colnames(block)) else labels[j]
    at <- at + term_width[j]
  }
  dimnames(mm) <- list(row.names(data), names_out)
  attr(mm, "assign") <- rep(seq_along(labels), term_width)
  mm
}

regularization_exact <- function(p, m, range_m) {
  isproduct <- function(x) grepl(":", x) & !grepl("\\(", x)
  isquadratic <- function(x) grepl("^I\\(.*\\^2\\)", x)
  ishinge <- function(x) grepl("^hinge\\(", x)
  isthreshold <- function(x) grepl("^thresholds\\(", x)
  iscategorical <- function(x) grepl("^categorical\\(", x)
  regtable <- function(name, default) {
    if (ishinge(name)) return(list(c(0, 1), c(0.5, 0.5)))
    if (iscategorical(name)) return(list(c(0, 10, 17), c(0.65, 0.5, 0.25)))
    if (isthreshold(name)) return(list(c(0, 100), c(2, 1)))
    default
  }
  lregtable <- list(c(0, 10, 30, 100), c(1, 1, 0.2, 0.05))
  qregtable <- list(c(0, 10, 17, 30, 100), c(1.3, 0.8, 0.5, 0.25, 0.05))
  pregtable <- list(c(0, 10, 17, 30, 100), c(2.6, 1.6, 0.9, 0.55, 0.05))
  mm <- m[p == 1, ]
  np <- nrow(mm)
  lqpreg <- lregtable
  if (sum(isquadratic(colnames(mm)))) lqpreg <- qregtable
  if (sum(isproduct(colnames(mm)))) lqpreg <- pregtable
  classregularization <- sapply(colnames(mm), function(n) {
    t <- regtable(n, lqpreg)
    approx(t[[1]], t[[2]], np, rule = 2)$y
  })/sqrt(np)
  ishinge <- grepl("^hinge\\(", colnames(mm))
  hmindev <- sapply(1:ncol(mm), function(i) {
    if (!ishinge[i]) return(0)
    std <- max(sd(mm[, i]), 1/sqrt(np))
    std * 0.5/sqrt(np)
  })
  tmindev <- sapply(1:ncol(mm), function(i) {
    ifelse(isthreshold(colnames(mm)[i]) && (sum(mm[, i]) == 0 || sum(mm[, i]) == nrow(mm)), 1, 0)
  })
  pmax(0.001 * range_m, hmindev, tmindev, apply(as.matrix(mm), 2, sd) * classregularization)
}

glmnet_exact <- local({
  src <- deparse(glmnet::glmnet, width.cutoff = 500L)
  src <- sub("if (any(is.na(x)))", "if (anyNA(x))", src, fixed = TRUE)
  src <- sub("storage.mode(x) <- \"double\"", "if (storage.mode(x) != \"double\") storage.mode(x) <- \"double\"", src, fixed = TRUE)
  f <- eval(parse(text = src))
  environment(f) <- asNamespace("glmnet")
  f
})

maxnet_exact <- function(p, data, f = maxnet.formula(p, data), regmult = 1, addsamplestobackground = TRUE, ...) {
  if (anyNA(data))
    stop("NA values in data table. Please remove them and rerun.")
  if (addsamplestobackground) {
    pdata <- data[p == 1, ]
    ndata <- data[p == 0, ]
    row_key <- function(d) do.call(paste, c(lapply(as.data.frame(as.matrix(d) + 0), sprintf, fmt = "%a"), sep = "|"))
    toadd <- !(row_key(pdata) %in% row_key(ndata))
    p <- c(p, rep(0, sum(toadd)))
    data <- rbind(data, pdata[toadd, ])
  }
  mm <- design_matrix(f, data)
  feature_range <- vapply(seq_len(ncol(mm)), function(j) range(mm[, j]), numeric(2))
  featuremins <- setNames(feature_range[1, ], colnames(mm))
  featuremaxs <- setNames(feature_range[2, ], colnames(mm))
  reg <- regularization_exact(p, mm, featuremaxs - featuremins) * regmult
  weights <- p + (1 - p) * 100
  gc()
  glmnet::glmnet.control(pmin = 1e-08, fdev = 0)
  model <- glmnet_exact(x = mm, y = as.factor(p), family = "binomial", standardize = F, penalty.factor = reg,
                        lambda = 10^(seq(4, 0, length.out = 200)) * sum(reg)/length(reg) * sum(p)/sum(weights),
                        weights = weights, ...)
  rm(mm)
  class(model) <- c("maxnet", class(model))
  if (length(model$lambda) < 200) {
    msg <- "Error: glmnet failed to complete regularization path.  Model may be infeasible."
    if (!addsamplestobackground)
      msg <- paste(msg, " Try re-running with addsamplestobackground=T.")
    stop(msg)
  }
  bb <- model$beta[, 200]
  model$betas <- bb[bb != 0]
  model$alpha <- 0
  rr <- maxnet:::predict.maxnet(model, data[p == 0, , drop = FALSE], type = "exponent", clamp = F)
  raw <- rr/sum(rr)
  model$entropy <- -sum(raw * log(raw))
  model$alpha <- -log(sum(rr))
  model$penalty.factor <- reg
  model$featuremins <- featuremins
  model$featuremaxs <- featuremaxs
  vv <- (sapply(data, class) != "factor")
  model$varmin <- apply(data[, vv, drop = FALSE], 2, min)
  model$varmax <- apply(data[, vv, drop = FALSE], 2, max)
  means <- apply(data[p == 1, vv, drop = FALSE], 2, mean)
  majorities <- sapply(names(data)[!vv], function(n) which.max(table(data[p == 1, n, drop = FALSE])))
  names(majorities) <- names(data)[!vv]
  model$samplemeans <- unlist(c(means, majorities))
  model$levels <- lapply(data, levels)
  model
}

parallel_fit <- function(X, f) {
  res <- mclapply(X, f, mc.cores = N_CORES, mc.preschedule = FALSE)
  bad <- vapply(res, inherits, logical(1), "try-error")
  if (any(bad)) stop(paste(sprintf("job %d: %s", which(bad), unlist(res[bad])), collapse = "\n"))
  res
}

val_auc <- function(labels, preds) {
  if (length(unique(labels)) < 2) return(NA_real_)
  as.numeric(roc(labels, preds, quiet = TRUE)$auc)
}

opt_one <- function(i) {
  rm <- opt_jobs$rm[i]; sp <- opt_jobs$sp[i]
  y_tr <- train_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  model <- tryCatch(maxnet_exact(p = y_tr, data = train_dat[, features, drop = FALSE], regmult = rm),
                    error = conditionMessage)
  if (is.character(model)) return(data.frame(reg_mult = rm, species = sp, auc_roc_val = NA_real_, error = model))
  p_va <- as.numeric(predict(model, newdata = val_dat[, features, drop = FALSE],
                             type = "logistic", clamp = TRUE))
  data.frame(reg_mult = rm, species = sp, auc_roc_val = val_auc(val_dat[[sp]], p_va), error = NA_character_)
}
if (nzchar(Sys.getenv("MAXNET_REG_MULT"))) {
  best_rm <- as.numeric(Sys.getenv("MAXNET_REG_MULT"))
  cat(sprintf("Using reg_mult %g from MAXNET_REG_MULT\n", best_rm))
} else {
  set.seed(42)
  opt_species <- sample(all_sp, min(20L, length(all_sp)))
  opt_jobs    <- expand.grid(rm = REG_MULT_VALUES, sp = opt_species, stringsAsFactors = FALSE)
  opt <- do.call(rbind, parallel_fit(seq_len(nrow(opt_jobs)), opt_one))
  write.csv(opt, file.path(results_dir, "reg_mult_optimization.csv"), row.names = FALSE)
  opt_ok  <- opt[!opt$species %in% opt$species[!is.na(opt$error)], ]
  agg     <- aggregate(auc_roc_val ~ reg_mult, data = opt_ok, FUN = mean)
  best_rm <- agg$reg_mult[which.max(agg$auc_roc_val)]
  print(agg, row.names = FALSE, digits = 4)
  cat(sprintf("Selected reg_mult %g by mean validation AUROC over %d species (%d excluded after a failed fit)\n",
              best_rm, length(unique(opt_ok$species)), length(opt_species) - length(unique(opt_ok$species))))
  if (best_rm %in% range(REG_MULT_VALUES)) cat("WARNING: selected reg_mult is at the edge of the grid\n")
}
writeLines(as.character(best_rm), file.path(results_dir, "best_reg_mult.txt"))

out_dir <- file.path(results_dir, "env", "per_species")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fit_one_species <- function(sp) {
  y_tr <- train_dat[[sp]]
  if (sum(y_tr) == 0 || sum(y_tr) == length(y_tr)) return(NULL)
  sp_safe <- gsub("[^A-Za-z0-9]", "_", sp)
  if (all(file.exists(file.path(out_dir, paste0(sp_safe, "_", SPLITS, ".csv"))))) return(NULL)
  model <- tryCatch(maxnet_exact(p = y_tr, data = train_dat[, features, drop = FALSE], regmult = best_rm),
                    error = conditionMessage)
  if (is.character(model)) return(data.frame(species = sp, reg_mult = best_rm, error = model))
  for (split in SPLITS) {
    rows <- dat[idx[[split]], c(features, sp)]
    write.csv(data.frame(row_index = idx[[split]] - 1L, species = sp, cov_set = "env", split = split,
                         logit = qlogis(as.numeric(predict(model, newdata = rows[, features, drop = FALSE],
                                                           type = "logistic", clamp = TRUE))),
                         actual = rows[[sp]]),
              file.path(out_dir, paste0(sp_safe, "_", split, ".csv")), row.names = FALSE)
  }
  NULL
}
t0 <- Sys.time()
failed <- do.call(rbind, parallel_fit(all_sp, fit_one_species))
if (!is.null(failed)) {
  write.csv(failed, file.path(results_dir, "failed_species.csv"), row.names = FALSE)
  cat(sprintf("Excluded %d species whose maxnet fit failed:\n", nrow(failed)))
  print(failed, row.names = FALSE)
}
for (split in SPLITS) {
  files <- list.files(out_dir, pattern = paste0("_", split, "\\.csv$"), full.names = TRUE)
  write.csv(do.call(rbind, lapply(files, read.csv, check.names = FALSE)),
            file.path(results_dir, "env", paste0("predictions_", split, "_all.csv")), row.names = FALSE)
}
unlink(out_dir, recursive = TRUE)
cat(sprintf("env: %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
