# STEM-LM

**Input CSV**: `time, latitude, longitude, env_*, species_*` — one row per site–time
observation. Env columns must be prefixed `env_*` or passed via `--env_cols`.
Species values must be 0/1.

## Files
| File | Purpose |
|---|---|
| `jsdm_model.py` | Model (species self-attn, ST + eco cross-attn, FIRE distance bias) |
| `jsdm_data.py` | Dataset, collator, H3 / grid block CV splits |
| `jsdm_train.py` | Training; also hosts shared `train_epoch` / `evaluate` helpers |
| `jsdm_ablation.py` | Ablations: `full`, `no_st`, `no_eco`, `no_st_eco` |
| `jsdm_inference.py` | `predict` (100%-mask, train-only sources) and `interactions` |

## Options (shared across scripts unless noted)

**Data / split**
- `--fold {random,h3,grid}` — split strategy. `h3` is spatial block CV (real lat/lon); `grid` is for euclidean/simulated coords.
- `--h3_resolution` (default `2`, ~183 km cells) — finer = more, smaller cells.
- `--train_frac` / `--test_frac` (default `0.8` / `0.1`) — val is the remainder.
- `--splits_path` — path to a `splits.json` from a prior training run; bypasses fold recomputation. Use this to keep train/val/test partitions identical across ablations and inference. Row indices are stored, so the CSV must not have been reordered or resized.
- `--no_save_splits` (train / ablation) — suppress the automatic `splits.json` dump written next to `best_model.pt`.
- `--num_source_sites` (default `64`) — N source sites per target, weighted by proximity.
- `--blind_percentile` (default `2.0`) — proximity threshold below which source entries are masked for masked species (prevents trivial copying).
- `--env_cols col1 col2 ...` — explicit env column list; otherwise all columns with `env_` prefix.
- `--euclidean_coords` — use Euclidean distance (for simulated data).
- `--no_time` — purely spatial model (disable temporal FIRE bias).

**Model** (train / ablation only)
- `--hidden_size` (`256`), `--num_attention_heads` (`8`, must be even), `--num_hidden_layers` (`4`), `--intermediate_size` (`512`), `--dropout` (`0.1`).
- `--num_env_groups` (`3`) — learned eco query groups in the eco cross-attention.

**Training**
- `--batch_size` (`32`), `--num_epochs` (`50`), `--learning_rate` (`1e-4`), `--weight_decay` (`0.01`).
- `--mlm_probability` (`0.15`) — fraction of species masked per example during training.
- `--max_grad_norm` (`1.0`) — gradient clipping.
- `--class_weighting` + `--class_weighting_beta` (`0.999`) — up-weight rare species (effective-number).
- `--gradient_checkpointing` — trade compute for memory.

**Ablation**
- `--ablation {full,no_st,no_eco,no_st_eco}` — which cross-attention branches to keep. Param counts differ across modes; a note is written to `ablation_summary.json`.

**Inference**
- `predict` subcommand flags: `--eval_split {val,test}` (default `test`), plus the data/split flags above. Pass `--splits_path <model_dir>/splits.json` to reuse the exact train/val/test partition from training; otherwise supply matching `--h3_resolution`/`--train_frac`/`--test_frac`/`--seed` so the H3 split is reproduced.
- `interactions` subcommand also accepts `--splits_path` to restrict the source pool to the original training rows.
- Species ordering in the CSV must match the trained model's `species_names.json`; the script asserts this and errors on mismatch.

## Outputs
Training writes to `--output_dir`: `best_model.pt`, `config.json`, `species_names.json`,
`splits.json` (unless `--no_save_splits`), `training_log.csv`, `per_species_auc_jsdm.csv`,
`interaction_matrix.npy`, and periodic checkpoints. Inference writes predictions as a parquet
with per-row lat/lon plus one column per species, and a `per_species_auc_{val,test}.csv`.
