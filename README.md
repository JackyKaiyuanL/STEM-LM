# STEM-LM

Masked-species Transformer for joint species distribution modelling. Each
observation is a site–time with a binary vector over species and environmental
covariates. A random subset of the species is masked and predicted from the
rest of the vector, from the K nearest other observations in space (their
species vectors and their environments), and from the target's own environment.

## Data

One CSV per dataset, one row per observation:

```
time, latitude, longitude, env_*, <species columns>
```

`time` is an ISO date or a number of days; `latitude`/`longitude` are degrees.
Environmental columns carry the `env_` prefix or are named with `--env_cols`;
every other column is a species with 0/1 entries. Preparation scripts for
eButterfly, NEUS, sPlotOpen and the monarch grid are under
`data_processing/`.

## Install and run

```bash
uv sync
uv run stemlm train data.csv --output_dir out --splits_path splits.json \
    --temporal_fire_init_periods 365 182 122 91 --mixed_precision bf16
```

`uv run pytest` runs the test suite.

## Sources

For every target the K nearest candidate rows (`--num_source_sites`, default 128)
are the sources. Training targets take candidates from the training rows only;
validation and test targets also take the other rows of their own split outside
their H3 cell at `--source_cell_resolution` (see the options table). During training,
`--train_exclusion` (default on) drops candidate sources within a radius and, independently,
within a time window of the target: each is 0 with probability 1/2 and otherwise
drawn log-uniformly between the target's nearest and farthest candidate. This
puts the geometry of evaluation and deployment into training without a tuned
scale. Validation, checkpoint selection and `test_results.csv` use no exclusion
(`test_results.csv` carries a `sources` column, `heldout` for the primary regime
and `train` for training rows as the only sources).
`--eval_exclusion_km` and `--eval_exclusion_days` take lists: the selected
checkpoint is also scored on the test set at every radius and window pair, written
to `test_sweep.csv`, giving performance as a function of distance to the nearest
source; by default the sweep holds the windows 0 and 1 day with no radius, so
same-day sources are removed in one row. `--causal_context` restricts sources to
earlier dates.

## Splits

`--resolution R` assigns whole H3 cells to train, validation and test
(`--train_frac`, `--test_frac`). Resolution 2 (cells of about 183 km edge)
tests extrapolation to unsampled regions; a fine resolution tests prediction at
new sites near data. `--splits_path` reuses a saved split.

`--min_train_presences M` (default 100) keeps the species with at least M
presences in the training rows; the baseline runners take the same option. Build
tables with a `--min_presences` at or below M, so that test rows do not decide
which species are kept.

## Model and training options

| Option | Default | Meaning |
|---|---|---|
| `--hidden_size`, `--num_attention_heads`, `--num_hidden_layers`, `--intermediate_size` | 256, 8, 4, 512 | Transformer size |
| `--num_env_groups` | 5 | learned queries pooling the sources' environments |
| `--per_species_env_rank` | 8 | rank of the per-species linear environmental head |
| `--temporal_fire_init_periods` | none | periods (days) of the periodic terms in the temporal distance bias; omit for a static dataset |
| `--no_time` | off | ignore the time column |
| `--ablation` | `full` | `no_st`, `no_env` or `no_st_env` remove cross-attention pathways |
| `--p` | `unif:0.0,1.0` | mask rate per row: a number or `unif:lo,hi` |
| `--loss_type` | `focal` | `focal` (`--focal_alpha 0.25 --focal_gamma 2.0`) or `bce` |
| `--batch_size`, `--num_epochs`, `--learning_rate`, `--weight_decay` | 32, 50, 1e-4, 0.01 | AdamW with cosine decay |
| `--mixed_precision` | `none` | `bf16` or `fp16` |
| `--grad_accum_steps`, `--gradient_checkpointing`, `--compile` | 1, off, off | memory and speed |
| `--val_p_list` | 0.25 0.5 0.75 1.0 | mask rates for validation and test |
| `--source_cell_resolution` | 7 | validation and test targets use the training rows and the other rows of their own split as sources, outside the target's H3 cell at this resolution; the uniform test scheme is also scored with training rows as the only sources |
| `--absence_mask_p_list`, `--no_absence_mask_eval` | 0.25 0.5 0.75 1.0 | presence-only evaluation block |
| `--temperature_scaling` | on | fit a temperature on validation logits at p = 1; every test metric uses the scaled logits |
| `--seed`, `--num_workers`, `--output_dir` | 42, cores, `./STEMLM_output` | |

Multi-GPU: `torchrun --nproc_per_node=N -m stemlm.cli train ...`; batch size is
per GPU; `latest_checkpoint.pt` resumes an interrupted run.

## Outputs

`best_model.pt` (selected by validation AUROC averaged over `--val_p_list`),
`best_model_by_cbi.pt`, `config.json`, `species_names.json`, `splits.json`,
`training_log.csv`, `test_results.csv` and `per_species_auc.csv` (per masking
rate and mask scheme), `ablation_summary.json`, and `temperature.json`.
Evaluation is deterministic: sources are fixed and
masks are seeded per batch.

## Library use

`stemlm.metric.evaluate_at_p` evaluates a model on a set of rows at one mask
rate; `compute_per_species_metrics` and `summarize_per_species_metrics` give
per-species and mean AUROC, AUPRC, CBI, Brier and ECE. Load `config.json` into
`JSDMConfig`, set `dataset.source_pool` to the run's training rows, and keep the
species order of `species_names.json`.
