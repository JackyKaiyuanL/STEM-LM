# STEM-LM

Joint species distribution model with masked-species pretraining.

**Input CSV**: `time, latitude, longitude, env_*, species_*` — one row per
site–time observation. Species are 0/1; env columns must be prefixed `env_*`
or passed via `--env_cols`.

## Files

| File | Purpose |
|---|---|
| `STEMLM_model.py` | Model: species self-attn, ST + env cross-attn, FIRE distance bias. `JSDMConfig.ablation` ∈ `{full, no_st, no_env, no_st_env}`. |
| `STEMLM_data.py` | Dataset, collators (uniform-mask + absence-mask), H3 splits. |
| `STEMLM_train.py` | Training, per-epoch val, end-of-training K-pass-bagged test eval. |
| `STEMLM_metric.py` | AUROC / AUPRC / CBI / Brier / ECE primitives + `bagged_evaluate_at_p`. Library only — no CLI. |

## Quickstart

```bash
python STEMLM_train.py data.csv \
    --output_dir ./out \
    --p unif:0.0,1.0 \
    --temporal_fire_init_periods 365 182
```

Defaults to focal loss (α=0.25, γ=2.0). Saves two checkpoints
(`best_model.pt` by val-AUROC, `best_model_by_cbi.pt` by val-CBI), evaluates
both, and reports per-p AUROC / AUPRC / CBI / Brier / ECE on a uniform-mask
block AND a presence-only (absence-mask) block.

To use BCE instead, pass `--loss_type bce`. Focal trades a small AUROC for big
CBI gains; γ=2 maximizes CBI at the cost of slight decrease in AUROC.

## Key options

**Loss** (defaults shown)
- `--loss_type {bce,focal}` `focal`
- `--focal_alpha 0.25 --focal_gamma 2.0` (RetinaNet defaults)
- `--class_weighting [β]` opt-in. Pass alone → β=0.999. BCE only.

**Mask rate**
- `--p` per-row mask rate. Float in `[0,1]`, `unif[:lo,hi]` (Uniform per row),
  or `beta:α,β` (Beta per row). Default `0.15`. Use `unif:0.0,1.0` for variable-p training.
- `--val_p_list 0.25 0.5 0.75 1.0` per-epoch fixed-p val rates.
- `--absence_mask_p_list 0.25 0.5 0.75 1.0` rates for the presence-only test block.
- `--no_absence_mask_eval` to skip the presence-only block.

**Splits** (block CV by default)
- `--fold {h3,grid,random}` `h3`
- `--resolution` block resolution (H3: `0..15`, default `2`; grid: side length, default `20`).
- `--splits_path` reuse a prior `splits.json` (keeps train/val/test identical across runs).
- `--train_frac 0.8 --test_frac 0.1` (val = remainder)
- `--num_source_sites 64`

**Model**
- `--hidden_size 256 --num_attention_heads 8 --num_hidden_layers 4 --intermediate_size 512`
- `--num_env_groups 5 --dropout 0.1`
- `--temporal_fire_init_periods 365 182 ...` periods (days) for sin/cos input added to FIRE temporal bias. Omit to disable.
- `--per_species_env_rank 8` parallel per-species env head (low-rank A·B + bias on raw target_env).
- `--no_time` purely spatial.
- `--euclidean_coords` non-geographic 2D coords.

**Training**
- `--batch_size 32 --num_epochs 50 --learning_rate 1e-4 --weight_decay 0.01`
- `--max_grad_norm 1.0 --gradient_checkpointing`
- `--mixed_precision {none,bf16,fp16}` `none`. `bf16`
- `--grad_accum_steps 1` effective batch = `batch_size × grad_accum_steps × world_size`.
- `--test_bag_K 10` K-pass bagging at end of training.

**Ablation**
- `--ablation {full,no_st,no_env,no_st_env}` `full`.

## Distributed (multi-GPU)

Launch with `torchrun`; `--batch_size` is per-GPU. Auto-detected via
`LOCAL_RANK` / `WORLD_SIZE`; without `torchrun` it's a single-process no-op.
Preemption-safe: rank 0 writes `latest_checkpoint.pt` every epoch and resumes
on resubmission.

```bash
torchrun --nproc_per_node=4 STEMLM_train.py data.csv \
    --output_dir ./out --mixed_precision bf16 --batch_size 32 [args]
```

## Per-mode ablation loop

Run `full` first (without `--splits_path`) to produce the shared splits, then
reuse:
```bash
for mode in full no_st no_env no_st_env; do
    python STEMLM_train.py data.csv --ablation $mode \
        --output_dir ./ablation/$mode \
        --splits_path ./ablation/full/splits.json \
        --temporal_fire_init_periods 365 182
done
```

## Outputs

Each `--output_dir` gets:
- `best_model.pt` (best-by-val-AUROC) and `best_model_by_cbi.pt` (best-by-val-CBI)
- `latest_checkpoint.pt` (preemption resume) + periodic `checkpoint_epoch{N}.pt`
- `config.json`, `species_names.json`, `splits.json`
- `training_log.csv` per-epoch metrics
- `test_results.csv` flat test metrics by mask scheme × p
- `per_species_auc.csv` per-species AUROC / AUPRC / CBI per p
- `ablation_summary.json` test metrics for both checkpoints + both mask schemes
- `cooccurrence_matrix.npy` learned species-species attention

## Inference

`STEMLM_metric.py` is a library; import `bagged_evaluate_at_p`,
`compute_per_species_metrics`, `summarize_per_species_metrics`. Pass
`--splits_path <run_dir>/splits.json` so source-pool matches training rows;
species ordering must match the run's `species_names.json`.
