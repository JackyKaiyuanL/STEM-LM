#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stemlm.data import JSDMDataset, compute_dist_info  # noqa: E402
from stemlm.metric import gather_logits_at_p  # noqa: E402
from stemlm.model import JSDMConfig, JSDMForMaskedSpeciesPrediction  # noqa: E402

META_COLS = ("time", "latitude", "longitude")


def load_model(run_dir, device):
    cfg = json.loads((run_dir / "config.json").read_text())
    config = JSDMConfig(**{k: v for k, v in cfg.items() if k in JSDMConfig.__dataclass_fields__})
    model = JSDMForMaskedSpeciesPrediction(config).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    temperature = run_dir / "temperature.json"
    return model, config, json.loads(temperature.read_text())["T_star"] if temperature.exists() else 1.0


def predict_grid(model, config, env_mean, obs, grid_csv, species, work_csv, device):
    grid = pd.read_csv(grid_csv)
    species_cols = [c for c in obs.columns if c not in META_COLS and not c.startswith("env_")]
    grid = pd.concat([grid, pd.DataFrame(0, index=grid.index, columns=species_cols, dtype=np.int8)],
                     axis=1)[obs.columns]
    pd.concat([obs, grid], ignore_index=True).to_csv(work_csv, index=False)
    dataset = JSDMDataset(str(work_csv), num_source_sites=config.num_source_sites)
    dataset.source_pool = np.arange(len(obs))
    dataset.fill_env(mean=env_mean)
    logits, _ = gather_logits_at_p(model, dataset, np.arange(len(obs), len(obs) + len(grid)),
                                   compute_dist_info(dataset), p_value=1.0, batch_size=512,
                                   device=device, num_workers=4)
    return grid[["latitude", "longitude"]].reset_index(drop=True), logits[:, dataset.species_cols.index(species)]


def plot_panels(predictions, column, out_png, resolution=0.5):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    from shapely.ops import unary_union

    include = {"United States of America", "Canada", "Mexico", "Guatemala", "Belize",
               "Honduras", "El Salvador", "Nicaragua", "Costa Rica", "Panama"}
    names = lambda r: (r.attributes.get("NAME_LONG", "") or r.attributes.get("NAME", "")).lower()
    countries = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    land = unary_union([r.geometry for r in shpreader.Reader(countries).records()
                        if any(i.lower() in names(r) or names(r) in i.lower() for i in include)])
    lakes_shp = shpreader.natural_earth(resolution="50m", category="physical", name="lakes")
    great = [r.geometry for r in shpreader.Reader(lakes_shp).records()
             if (r.attributes.get("name") or "") in {"Lake Superior", "Lake Michigan", "Lake Huron",
                                                     "Lake Erie", "Lake Ontario"}]
    lakes = unary_union(great)
    lon, lat = np.meshgrid(np.arange(-170.0, -50.0 + resolution, resolution),
                           np.arange(7.0, 72.0 + resolution, resolution))
    lon, lat = lon.ravel(), lat.ravel()
    keep = np.array([land.contains(Point(x, y)) for x, y in zip(lon, lat)])
    keep &= ~((lat >= 18) & (lat <= 23) & (lon >= -161) & (lon <= -154))
    lon, lat = lon[keep], lat[keep]
    keep = ~np.array([lakes.contains(Point(x, y)) for x, y in zip(lon, lat)])
    lon, lat = lon[keep], lat[keep]

    filled = {}
    for date, df in predictions.items():
        _, nearest = cKDTree(df[["longitude", "latitude"]].values).query(np.column_stack([lon, lat]))
        filled[date] = df[column].values[nearest]
    norm = Normalize(vmin=0.0, vmax=max(v.max() for v in filled.values()))
    cmap = plt.get_cmap("inferno").copy()
    proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=40, standard_parallels=(20, 60))
    angles = np.deg2rad(np.arange(0, 360, 60))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": proj},
                             gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    for ax, (date, values) in zip(axes.ravel(), filled.items()):
        verts = [np.column_stack([x + 0.72 * resolution * np.cos(angles), y + 0.72 * resolution * np.sin(angles)])
                 for x, y in zip(lon, lat)]
        pc = PolyCollection(verts, array=values, cmap=cmap, norm=norm, edgecolor="none",
                            transform=ccrs.PlateCarree())
        ax.add_collection(pc)
        ax.add_geometries(great, crs=ccrs.PlateCarree(), facecolor="white", edgecolor="none", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="white", zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="white", alpha=0.6, zorder=3)
        ax.set_extent([-128, -65, 14, 60], crs=ccrs.PlateCarree())
        ax.set_aspect("auto")
        ax.text(0.02, 0.96, date, transform=ax.transAxes, fontsize=11, verticalalignment="top", color="white",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    fig.colorbar(pc, ax=axes.ravel().tolist(), shrink=0.7, aspect=30, label="Predicted habitat suitability", pad=0.01)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Predict one species over the enriched grid with a trained STEM-LM run.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--grid_dir", required=True, type=Path)
    parser.add_argument("--dates", nargs="+", default=["2025-05-15", "2025-07-15", "2025-09-15", "2025-11-15"])
    parser.add_argument("--species", default="Danaus plexippus")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, T = load_model(args.run_dir, device)
    env_mean = json.loads((args.run_dir / "env_stats.json").read_text())["mean"]
    obs = pd.read_csv(args.csv_path)
    species = json.loads((args.run_dir / "species_names.json").read_text())
    obs = obs[[c for c in obs.columns if c in META_COLS or c.startswith("env_")] + species]
    predictions = {}
    for date in args.dates:
        cells, z = predict_grid(model, config, env_mean, obs, args.grid_dir / f"monarch_grid_{date}.csv", args.species,
                                args.output_dir / "_combined.csv", device)
        pred = cells.assign(suitability=1.0 / (1.0 + np.exp(-z / T)), suitability_raw=1.0 / (1.0 + np.exp(-z)))
        pred.to_csv(args.output_dir / f"monarch_pred_{date}.csv", index=False)
        predictions[date] = pred
        print(f"{date}: {len(pred):,} cells, T* {T:.4f}, suitability {pred.suitability.min():.4f}"
              f"-{pred.suitability.max():.4f}", flush=True)
    (args.output_dir / "_combined.csv").unlink()
    if not args.no_plot:
        plot_panels(predictions, "suitability", args.output_dir / "monarch_pred_full.png")
        plot_panels(predictions, "suitability_raw", args.output_dir / "monarch_pred_full_raw.png")


if __name__ == "__main__":
    main()
