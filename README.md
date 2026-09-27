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
    --p unif:0.0,1.0 --temporal_fire_init_periods 365 182 122 91 \
    --train_exclusion --mixed_precision bf16 --temperature_scaling
```

`uv run pytest` runs the test suite.

## Sources

For every target the K nearest training rows (`--num_source_sites`, default 64)
are the sources; no held-out row is ever a source. During training,
`--train_exclusion` drops candidate sources within a radius and, independently,
within a time window of the target: each is 0 with probability 1/2 and otherwise
drawn log-uniformly between the target's nearest and farthest candidate. This
puts the geometry of evaluation and deployment into training without a tuned
scale. At evaluation `--eval_exclusion_km` and `--eval_exclusion_days` apply a
fixed radius and window to every validation and test target; sweeping them
gives performance as a function of distance to the nearest data.
`--causal_context` restricts sources to earlier dates.

## Splits

`--resolution R` assigns whole H3 cells to train, validation and test
(`--train_frac`, `--test_frac`). Resolution 2 (cells of about 183 km edge)
tests extrapolation to unsampled regions; a fine resolution tests prediction at
new sites near data. `--splits_path` reuses a saved split.

## Model and training options

| Option | Default | Meaning |
|---|---|---|
| `--hidden_size`, `--num_attention_heads`, `--num_hidden_layers`, `--intermediate_size` | 256, 4, 3, 1024 | Transformer size |
| `--num_env_groups` | 5 | learned queries pooling the sources' environments |
| `--per_species_env_rank` | 8 | rank of the per-species linear environmental head |
| `--temporal_fire_init_periods` | none | periods (days) of the periodic terms in the temporal distance bias; omit for a static dataset |
| `--no_time` | off | ignore the time column |
| `--ablation` | `full` | `no_st`, `no_env` or `no_st_env` remove cross-attention pathways |
| `--p` | 0.15 | mask rate per row: a number, `unif:lo,hi` or `beta:a,b` |
| `--loss_type` | `focal` | `focal` (`--focal_alpha 0.25 --focal_gamma 2.0`) or `bce` |
| `--batch_size`, `--num_epochs`, `--learning_rate`, `--weight_decay` | 32, 50, 1e-4, 0.01 | AdamW with cosine decay |
| `--mixed_precision` | `none` | `bf16` or `fp16` |
| `--grad_accum_steps`, `--gradient_checkpointing`, `--compile` | 1, off, off | memory and speed |
| `--val_p_list` | 0.25 0.5 0.75 1.0 | mask rates for validation and test |
| `--absence_mask_p_list`, `--no_absence_mask_eval` | 0.25 0.5 0.75 1.0 | presence-only evaluation block |
| `--temperature_scaling` | off | fit a temperature on validation logits at p = 1 and report calibrated ECE |
| `--seed`, `--num_workers`, `--output_dir` | 42, cores, `./STEMLM_output` | |

Multi-GPU: `torchrun --nproc_per_node=N -m stemlm.cli train ...`; batch size is
per GPU; `latest_checkpoint.pt` resumes an interrupted run.

## Outputs

`best_model.pt` (selected by validation AUROC averaged over `--val_p_list`),
`best_model_by_cbi.pt`, `config.json`, `species_names.json`, `splits.json`,
`training_log.csv`, `test_results.csv` and `per_species_auc.csv` (per masking
rate and mask scheme), `ablation_summary.json`, and `temperature.json` with
`--temperature_scaling`. Evaluation is deterministic: sources are fixed and
masks are seeded per batch.

## Library use

`stemlm.metric.evaluate_at_p` evaluates a model on a set of rows at one mask
rate; `compute_per_species_metrics` and `summarize_per_species_metrics` give
per-species and mean AUROC, AUPRC, CBI, Brier and ECE. Load `config.json` into
`JSDMConfig`, set `dataset.source_pool` to the run's training rows, and keep the
species order of `species_names.json`.
