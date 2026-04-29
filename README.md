# STEM-LM

**Input CSV**: `time, latitude, longitude, env_*, species_*` — one row per site–time
observation. Env columns must be prefixed `env_*` or passed via `--env_cols`.
Species values must be 0/1.

## Files
| File | Purpose |
|---|---|
| `STEMLM_model.py` | Model (species self-attn, ST + env cross-attn, FIRE distance bias). Single `JSDMConfig.ablation` field selects `full`, `no_st`, `no_env`, or `no_st_env` — ablated branches are not instantiated. |
| `STEMLM_data.py` | Dataset, collator, H3 / grid block CV splits. Pairwise distances are never materialized; `|Δt|` and haversine are computed on-the-fly from stored site coords. |
| `STEMLM_train.py` | Training; also hosts shared `train_epoch` / `evaluate` helpers and the final fixed-p test eval. Takes `--ablation` directly. |
| `STEMLM_inference.py` | `predict` (per-p fixed-mask AUC) and `interactions` (species × species attention matrix). |

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
- `--temporal_fire_init_periods 365 182 ...` — init periods (days) for `K` learnable sin/cos channels added to FIRE's temporal distance bias on ST attention scores. Periodic features are evaluated on Δt directly (`cos(ω_k·Δt)`, `sin(ω_k·Δt)`) and concatenated with the monotone log-distance feature, then mapped through FIRE's MLP to a scalar bias. Recommended for cyclic phenology (e.g., eButterfly): `365 182` (annual + semi-annual). Omit (default) to disable.
- `--fire_no_zero_init_periodic` — flag (off by default). With zero-init on, FIRE's input-linear columns for the periodic features start at zero so the monotone bias is unaffected at step 0. Disable only for legacy reproducibility.
- `--gate_hidden_size` (default `hidden_size // 8`) — bottleneck width of the ST/Env gate MLP. Old checkpoints use the default; widen to `2 * hidden_size` (or more) for sharper per-species routing.

**Training**
- `--batch_size` (`32`), `--num_epochs` (`50`), `--learning_rate` (`1e-4`), `--weight_decay` (`0.01`).
- `--p` (default `0.15`) — per-row mask rate applied to both presences and absences. Accepts a float in `[0, 1]` or a `rand[:lo,hi]` string (sample `Uniform[lo, hi]` per row; bare `rand` = `rand:0.0,1.0`). Example regimes:
  - `--p 0.15` — fixed 15% random masking (classical JSDM, current default).
  - `--p rand` — sample a single mask rate per row from `Uniform[0, 1]`; covers the full range in training.
  - `--p 1.0` — always 100% mask (matches the `predict` deployment regime).
- `--max_grad_norm` (`1.0`) — gradient clipping.
- `--class_weighting` (`0.999`) — up-weight rare species (effective-number).
- `--gradient_checkpointing` — trade compute for memory.
- `--val_p_list` (default `0.25 0.5 0.75 1.0`) — fixed mask rates evaluated each epoch with deterministic per-batch seeding (`FixedPValCollator`). Per-p val AUC and AUPRC are logged to `training_log.csv` as `val_auc_p... / val_auprc_p...`; their mean AUC (`val_auc_mean`) drives `best_model.pt` selection. AUPRC is reported but not used for selection.
- `--mixed_precision {none,bf16,fp16}` (default `none`) — autocast region around forward/backward. `bf16` is the recommended setting on A40/L40/A100/H100: ~50% activation-memory savings, no loss-scaling needed. `fp16` runs `GradScaler` automatically (older GPUs only). State dict on disk stays fp32.
- `--grad_accum_steps` (default `1`) — accumulate gradients across this many micro-batches before each `optimizer.step()`. Effective batch = `batch_size × grad_accum_steps × world_size`. Use to keep a target effective batch when shrinking `--batch_size` for VRAM.
- **Distributed (multi-GPU per run):** launch with `torchrun` and the same script runs unchanged. `--batch_size` is **per-GPU** under DDP. Auto-detected from `LOCAL_RANK` / `WORLD_SIZE` env vars; without `torchrun` everything is a no-op single-process path. Every epoch rank 0 writes `latest_checkpoint.pt`; resubmitting with the same `--output_dir` resumes from the last completed epoch (preemption-safe; pair with `#SBATCH --requeue`).
  ```bash
  torchrun --nproc_per_node=4 STEMLM_train.py data.csv \
      --output_dir ./out --mixed_precision bf16 --gradient_checkpointing \
      --batch_size 32 --num_epochs 100 [other args]
  ```
  For per-mode ablations on a 4-GPU node (run two 2-GPU jobs in parallel), set distinct `--master_port` values:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
      STEMLM_train.py data.csv --ablation full --output_dir ./run/full \
      --mixed_precision bf16 [args] &
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
      STEMLM_train.py data.csv --ablation no_st --output_dir ./run/no_st \
      --mixed_precision bf16 [args] &
  wait
  ```
  Validation predictions are `all_gather`-ed across ranks so AUC is computed on the full val set; only rank 0 logs and writes outputs. `find_unused_parameters=True` is set so ablation modes (whose disabled branches receive no gradient) work under DDP — costs a small per-step overhead vs. full-mode-only DDP. Before paper-quality runs, sanity-check that (a) `--mixed_precision none` still reproduces the original numbers within seed noise, (b) `--mixed_precision bf16` matches `none` within ~0.005 val AUC, and (c) `torchrun --nproc_per_node=2` matches single-GPU at the same effective batch within seed variance.

**Ablation**
- `STEMLM_train.py --ablation {full,no_st,no_env,no_st_env}` — which cross-attention branches to keep. Param counts differ across modes. Ablated branches are not instantiated, so `no_env` / `no_st_env` have no environmental modules at all (no dead weight). Every training run writes `ablation_summary.json` alongside the checkpoint.
- Per-mode loop (run `full` first *without* `--splits_path` so it produces the shared splits, then reuse it for the rest):
  ```bash
  for mode in full no_st no_env no_st_env; do
      python STEMLM_train.py data.csv \
          --ablation $mode \
          --output_dir ./ablation_out/$mode \
          --splits_path ./ablation_out/full/splits.json \
          --num_epochs 30 --hidden_size 256 --num_hidden_layers 3 \
          --temporal_fire_init_periods 365 182 --gate_hidden_size 512
  done
  ```

**Inference**
- `predict` subcommand flags: `--eval_split {val,test}` (default `test`), plus the data/split flags above. Pass `--splits_path <model_dir>/splits.json` to reuse the exact train/val/test partition from training; otherwise supply matching `--fold`/`--resolution`/`--train_frac`/`--test_frac`/`--seed` so the split is reproduced.
- `interactions` subcommand also accepts `--splits_path` to restrict the source pool to the original training rows.
- Species ordering in the CSV must match the trained model's `species_names.json`; the script asserts this and errors on mismatch.

## Outputs
Training writes to `--output_dir`: `best_model.pt`, `config.json`, `species_names.json`,
`splits.json` (unless `--no_save_splits`), `training_log.csv`, `per_species_auc.csv`,
`ablation_summary.json` (mode + test AUPRC/AUC by p + param count + `world_size` + `mixed_precision`),
`interaction_matrix.npy`, `latest_checkpoint.pt` (rewritten every epoch for preemption resume), and
periodic numbered checkpoints every 10 epochs. Inference writes predictions as a parquet with per-row
lat/lon plus one column per species, and a `per_species_auc_{val,test}.csv`.

