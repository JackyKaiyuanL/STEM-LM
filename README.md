# STEM-LM

Joint species distribution model with masked-species pretraining.

**Input CSV**: `time, latitude, longitude, env_*, species_*` — one row per
site–time observation. Species are 0/1; env columns must be prefixed `env_*`
or passed via `--env_cols`.

## Data

Both datasets used in the paper are public; preparation is fully scripted under `data_processing/`.

| Dataset | Source | URL | Prep |
|---|---|---|---|
| eButterfly NA 2011–2025 (M=17,077, S=173) | GBIF DwC-A `cf3bdc30-370c-48d3-8fff-b587a39d72d6` | https://www.gbif.org/dataset/cf3bdc30-370c-48d3-8fff-b587a39d72d6 | `data_processing/eButterfly/` |
| sPlotOpen v2.0 (M=95,104, S=1,201) | iDiv Data Repository v76 (Sabatini et al. 2021) | https://idata.idiv.de/ddm/Data/ShowData/3474 | `data_processing/sPlotOpen/` |

Each subfolder's README lists the exact download steps for the raw archive plus environmental rasters (ERA5-Land, MOD13Q1, WorldClim, SoilGrids, Copernicus DEM) and the script order to produce the final files consumed by `STEMLM_train.py`:
- `<dataset>.csv` — wide presence/absence + env covariates
- `<dataset>_splits.json` — H3 resolution-2 spatial-block split, seed 42, 80/10/10

## Files

| File | Purpose |
|---|---|
| `STEMLM_model.py` | Model: species self-attn, ST + env cross-attn, FIRE distance bias. `JSDMConfig.ablation` ∈ `{full, no_st, no_env, no_st_env}`. |
| `STEMLM_data.py` | Dataset, collators (uniform-mask + absence-mask), H3 splits. |
| `STEMLM_train.py` | Training, per-epoch val, end-of-training K-pass-bagged test eval. |
| `STEMLM_metric.py` | AUROC / AUPRC / CBI / Brier / ECE calculations + `bagged_evaluate_at_p`. Library only — no CLI. |

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
- `--temperature_scaling` adds Guo 2017 post-hoc temperature scaling: fit T\* on val logits at p=1.00, apply at every test p. Saves `temperature.json` (T\* + per-p T-cal ECE) and a `tcal_ece` column in `test_results.csv`. Apply at inference with `sigmoid(logits / T*)`.

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

## Reproducing paper results

All STEM-LM runs use H3 resolution-2 spatial splits (seed 42); deep-learning runs use three training seeds (41, 42, 43), and the mean across seeds is reported in the paper. Default training hyperparameters: focal loss (γ=2, α=0.25), `--p unif:0.0,1.0`, batch size 128, bf16, AdamW, cosine LR.

**eButterfly — main runs (Tables 3, 5)**
```bash
for s in 41 42 43; do
  python STEMLM_train.py data/ebutterfly_na_2011_2025.csv \
      --output_dir ./out/ebutterfly_focal_seed${s} \
      --splits_path data/ebutterfly_splits.json \
      --p unif:0.0,1.0 --temporal_fire_init_periods 365 182 122 91 \
      --num_epochs 100 --batch_size 128 --mixed_precision bf16 \
      --temperature_scaling --seed ${s}
done

# BCE variant (Table 3 STEM-LM (B))
for s in 41 42 43; do
  python STEMLM_train.py data/ebutterfly_na_2011_2025.csv \
      --output_dir ./out/ebutterfly_bce_seed${s} \
      --splits_path data/ebutterfly_splits.json \
      --loss_type bce --p unif:0.0,1.0 \
      --temporal_fire_init_periods 365 182 122 91 \
      --num_epochs 100 --batch_size 128 --mixed_precision bf16 \
      --temperature_scaling --seed ${s}
done
```

**eButterfly — cross-attention head ablation (Table 1)**
```bash
for mode in full no_st no_env no_st_env; do
  for s in 41 42 43; do
    python STEMLM_train.py data/ebutterfly_na_2011_2025.csv \
        --ablation ${mode} \
        --output_dir ./out/ablation/${mode}_seed${s} \
        --splits_path data/ebutterfly_splits.json \
        --p unif:0.0,1.0 --temporal_fire_init_periods 365 182 122 91 \
        --num_epochs 100 --batch_size 128 --mixed_precision bf16 \
        --seed ${s}
  done
done
```

**eButterfly — source-site count ablation (Table 2)**
```bash
for N in 32 64 128; do
  for s in 41 42 43; do
    python STEMLM_train.py data/ebutterfly_na_2011_2025.csv \
        --num_source_sites ${N} \
        --output_dir ./out/sites${N}_seed${s} \
        --splits_path data/ebutterfly_splits.json \
        --p unif:0.0,1.0 --temporal_fire_init_periods 365 182 122 91 \
        --num_epochs 100 --batch_size 128 --mixed_precision bf16 \
        --seed ${s}
  done
done
```

**sPlotOpen — main runs (Tables 4, 5)**
```bash
for s in 41 42 43; do
  python STEMLM_train.py data/splotopen_global.csv \
      --output_dir ./out/splotopen_focal_seed${s} \
      --splits_path data/splotopen_global_splits.json \
      --no_time --p unif:0.0,1.0 \
      --num_epochs 50 --batch_size 128 --mixed_precision bf16 \
      --temperature_scaling --seed ${s}
done

# BCE variant
for s in 41 42 43; do
  python STEMLM_train.py data/splotopen_global.csv \
      --output_dir ./out/splotopen_bce_seed${s} \
      --splits_path data/splotopen_global_splits.json \
      --loss_type bce --no_time --p unif:0.0,1.0 \
      --num_epochs 50 --batch_size 128 --mixed_precision bf16 \
      --temperature_scaling --seed ${s}
done
```

**Baselines** — see `benchmarks/{statistical_SDM,MaskSDM,CISO}/<dataset>/`. Each subfolder has a self-contained notebook (or script) plus per-baseline README with upstream-repo clone hooks and the exact data-staging requirements.

## Outputs

Each `--output_dir` gets:
- `best_model.pt` (best-by-val-AUROC) and `best_model_by_cbi.pt` (best-by-val-CBI)
- `latest_checkpoint.pt` (preemption resume) + periodic `checkpoint_epoch{N}.pt`
- `config.json`, `species_names.json`, `splits.json`
- `training_log.csv` per-epoch metrics
- `test_results.csv` flat test metrics by mask scheme × p (includes `tcal_ece` when `--temperature_scaling`)
- `per_species_auc.csv` per-species AUROC / AUPRC / CBI per p
- `ablation_summary.json` test metrics for both checkpoints + both mask schemes
- `cooccurrence_matrix.npy` learned species-species attention
- `temperature.json` (only with `--temperature_scaling`) — `T_star`, `fitted_at_p`, and per-p T-cal ECE for both mask schemes

## Inference

`STEMLM_metric.py` is a library; import `bagged_evaluate_at_p`,
`compute_per_species_metrics`, `summarize_per_species_metrics`. Pass
`--splits_path <run_dir>/splits.json` so source-pool matches training rows;
species ordering must match the run's `species_names.json`.
