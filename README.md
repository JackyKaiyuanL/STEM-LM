# STEM-LM

**Input CSV**: `time, latitude, longitude, env_*, species_*` — one row per site–time
observation. Env columns must be prefixed `env_*` or passed via `--env_cols`.
Species values must be 0/1.

## Files
| File | Purpose |
|---|---|
| `jsdm_model.py` | Model (species self-attn, ST + env cross-attn, FIRE distance bias). Single `JSDMConfig.ablation` field selects `full`, `no_st`, `no_env`, or `no_st_env` — ablated branches are not instantiated. |
| `jsdm_data.py` | Dataset, collator, H3 / grid block CV splits. Pairwise distances are never materialized; `|Δt|` and haversine are computed on-the-fly from stored site coords. |
| `jsdm_train.py` | Training; also hosts shared `train_epoch` / `evaluate` helpers and the final fixed-p test eval. Takes `--ablation` directly. |
| `jsdm_ablation.py` | Thin master: shells out to `jsdm_train.py` once per mode, shares splits across modes, aggregates `ablation_summary.json` files into `ablation_comparison.json`. No model or training code. |
| `jsdm_inference.py` | `predict` (per-p fixed-mask AUC) and `interactions` (species × species attention matrix). |

## Options (shared across scripts unless noted)

**Data / split**
- `--fold {random,h3,grid}` — split strategy. `h3` is spatial block CV on real lat/lon (each H3 cell is assigned entirely to one split; no temporal blocking — the paper claim is spatial transfer, and season-blocking over-blinds migratory taxa). `grid` is the euclidean-coords analogue.
- `--resolution` — block resolution. For `--fold h3`, H3 resolution `0..15` (default `2`, ~183 km edge). For `--fold grid`, grid side length (any positive integer; default `20` → `20×20` cells). Must not be set with `--fold random`.
- `--train_frac` / `--test_frac` (default `0.8` / `0.1`) — val is the remainder.
- `--splits_path` — path to a `splits.json` from a prior training run; bypasses fold recomputation. Use this to keep train/val/test partitions identical across ablations and inference. Row indices are stored, so the CSV must not have been reordered or resized.
- `--no_save_splits` (train / ablation) — suppress the automatic `splits.json` dump written next to `best_model.pt`.
- `--num_source_sites` (default `64`) — N source sites per target, weighted by proximity.
- `--blind_percentile` (default `auto`) — percentile of normalized **spatial** pairwise distance below which source entries are masked for masked species. Time is not part of the blinding metric (seasonally-separated obs at the same site are legitimate different samples; blocking them would over-blind migratory taxa). Under `auto` (default), the percentile is picked via a half-decay rule on the dataset's Jaccard(distance) curve — the blind radius is set to where mean pairwise Jaccard drops to the midpoint between its nearest-neighbor value and the dataset's background. Pass a float to override.
- `--env_cols col1 col2 ...` — explicit env column list; otherwise all columns with `env_` prefix.
- `--euclidean_coords` — use Euclidean distance.
- `--no_time` — purely spatial model (disable temporal FIRE bias).

**Model** (train only)
- `--hidden_size` (`256`), `--num_attention_heads` (`8`, must be even), `--num_hidden_layers` (`4`), `--intermediate_size` (`512`), `--dropout` (`0.1`).
- `--num_env_groups` (`3`) — learned environmental query groups in the environmental cross-attention.
- `--temporal_fire_init_periods 365 180 ...` — list of init periods (days) for learnable sin/cos channels in the temporal FIRE bias. The list length is the number of periodic channels K; each ω_k is learnable so seasonality is *discovered*, not hard-coded. Omit (or pass empty) to disable periodicity and fall back to legacy monotone FIRE. Recommended values: `365 180 730 1825` (K=4: annual / semi / biennial / 5-yr) or `365 180 120 91 730 1825` (K=6: adds sub-annual for multivoltine taxa). Year-to-year phenological anomalies are handled separately by the Env head via env covariates.
- `--fire_no_zero_init_periodic` — flag (off by default). When zero-init is on (the default), the periodic (sin/cos) weights in FIRE's MLP start at zero so the module is arithmetically identical to legacy FIRE at step 0 — the optimizer has to *earn* periodic contribution. Disable only for legacy reproducibility.
- `--gate_hidden_size` (default `hidden_size // 8`) — bottleneck width of the ST/Env gate MLP. Old checkpoints use the default; widen to `2 * hidden_size` (or more) for sharper per-species routing.
- `--gate_init_bias` (default `0.0`) — initial bias of the gate MLP's final Linear. Negative values start training with the gate favoring Env (`-2.0` → `sigmoid≈0.12`), so ST has to *earn* its weight. Prevents a useless ST head from bleeding into outputs at convergence.
- `--gate_l1` (default `0.0`) — L1 weight on `ReLU(gate_logit).mean()` added to the loss. Penalizes the positive side only; species where ST genuinely helps still push their gate up, while noise-species gates stay near the Env-biased init. Recommended: `1e-4`.

**Training**
- `--batch_size` (`32`), `--num_epochs` (`50`), `--learning_rate` (`1e-4`), `--weight_decay` (`0.01`).
- `--p` (default `0.15`) — per-row mask rate applied to both presences and absences. Accepts a float in `[0, 1]` or a `rand[:lo,hi]` string (sample `Uniform[lo, hi]` per row; bare `rand` = `rand:0.0,1.0`). Example regimes:
  - `--p 0.15` — fixed 15% random masking (classical JSDM, current default).
  - `--p rand` — sample a single mask rate per row from `Uniform[0, 1]`; covers the full range in training.
  - `--p 1.0` — always 100% mask (matches the `predict` deployment regime).
- `--max_grad_norm` (`1.0`) — gradient clipping.
- `--class_weighting` (`0.999`) — up-weight rare species (effective-number).
- `--gradient_checkpointing` — trade compute for memory.
- `--val_p_list` (default `0.25 0.5 0.75 1.0`) — fixed mask rates evaluated each epoch with deterministic per-batch seeding (`FixedPValCollator`). Per-p val AUC and AUPRC are logged to `training_log.csv` as `val_auc_p... / val_auprc_p...`; their mean AUPRC (`val_auprc_mean`) drives `best_model.pt` selection — AUPRC is preferred over AUROC because most JSDM species are rare and AUROC is inflated by the easy common-species tail. AUC is still reported as a secondary metric.

**Ablation**
- `jsdm_train.py --ablation {full,no_st,no_env,no_st_env}` — which cross-attention branches to keep. Param counts differ across modes. Ablated branches are not instantiated, so `no_env` / `no_st_env` have no environmental modules at all (no dead weight). Every training run writes `ablation_summary.json` alongside the checkpoint.
- `jsdm_ablation.py` is a master script that calls `jsdm_train.py` once per mode, shares splits across modes automatically, and aggregates results:
  - `--modes full no_st no_env no_st_env` (default = all four) — subset of modes to run.
  - `--shared_splits_from path/to/splits.json` — optional; otherwise the first completed mode's `splits.json` is reused by the remaining modes.
  - All other flags are passed through verbatim to `jsdm_train.py` (everything in the **Data / split**, **Model**, **Training** sections above). The master auto-injects `--ablation $mode` and `--output_dir <parent>/<mode>/` per run, so **do not** pass either of those in the passthrough section.
- Usage pattern A — direct per-mode (gives you a loop-level handle, streams logs cleanly):
  ```bash
  for mode in full no_st no_env no_st_env; do
      python jsdm_train.py data.csv \
          --ablation $mode \
          --output_dir ./ablation_out/$mode \
          --splits_path ./ablation_out/full/splits.json \
          --num_epochs 30 --hidden_size 256 --num_hidden_layers 3 \
          --temporal_fire_freqs 4 --gate_hidden_size 512 \
          --gate_init_bias -2.0 --gate_l1 1e-4
  done
  ```
  (Run `full` first *without* `--splits_path` so it produces the shared splits, then reuse it for the rest.)
- Usage pattern B — master-driven (auto splits, auto aggregation, single invocation):
  ```bash
  python jsdm_ablation.py data.csv \
      --output_dir ./ablation_out \
      --modes full no_st no_env no_st_env \
      --num_epochs 30 --hidden_size 256 --num_hidden_layers 3 \
      --temporal_fire_freqs 4 --gate_hidden_size 512 \
      --gate_init_bias -2.0 --gate_l1 1e-4
  ```
  Writes `ablation_out/<mode>/{best_model.pt, config.json, ...}` per mode and `ablation_out/ablation_comparison.json` with test AUC, val AUC, and param count per mode.

**Inference**
- `predict` subcommand flags: `--eval_split {val,test}` (default `test`), plus the data/split flags above. Pass `--splits_path <model_dir>/splits.json` to reuse the exact train/val/test partition from training; otherwise supply matching `--fold`/`--resolution`/`--train_frac`/`--test_frac`/`--seed` so the split is reproduced.
- `interactions` subcommand also accepts `--splits_path` to restrict the source pool to the original training rows.
- Species ordering in the CSV must match the trained model's `species_names.json`; the script asserts this and errors on mismatch.

## Outputs
Training writes to `--output_dir`: `best_model.pt`, `config.json`, `species_names.json`,
`splits.json` (unless `--no_save_splits`), `training_log.csv`, `per_species_auc_jsdm.csv`,
`ablation_summary.json` (mode + test AUPRC/AUC by p + param count), `interaction_matrix.npy`, and
periodic checkpoints. Inference writes predictions as a parquet with per-row lat/lon plus one
column per species, and a `per_species_auc_{val,test}.csv`.

The master `jsdm_ablation.py` additionally writes `ablation_comparison.json` in its `--output_dir`,
aggregating each mode's `ablation_summary.json` into a single table for A/B comparison.
