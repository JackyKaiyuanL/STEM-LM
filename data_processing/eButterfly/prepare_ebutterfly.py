"""
Prepare eButterfly checklists for ST-JSDM.

Source: eButterfly DwC-A from GBIF (cf3bdc30-370c-48d3-8fff-b587a39d72d6)
        Downloaded 2026-04-13, 150,520 events / 637,514 occurrences

Outputs (two files per run):
  ebutterfly[_us]_2011_2025_jsdm.csv — 2011–2025, time = ISO date (YYYY-MM-DD)
                                       For temporal modeling with ERA5-Land + MODIS dynamic
  ebutterfly[_us]_static_jsdm.csv    — all years, time = 0 (compressed, like SatButterfly)
                                       For static modeling with WorldClim + DEM + MODIS climatology

  Columns: time, latitude, longitude, <species...>
  One row = one structured checklist (presence/absence for each species)

Geographic scope: US + Canada (including Alaska) + Mesoamerica
  - Excludes Hawaii (stateProvince == 'Hawaii')
  - Excludes non-continental islands (lon filter: -170 to -50)
  - Excludes Greenland (no eButterfly data there anyway)
  - Mesoamerica: Mexico, Guatemala, Belize, Honduras, El Salvador,
                 Nicaragua, Costa Rica, Panama (~799 structured checklists)
  - Excludes South America (too sparse)

Checklist types included (presence/absence valid):
  Traveling Survey, Area Survey, Timed count, Point Count,
  Atlas square, Pollard Walk, Pollard transect

Excluded:
  Incidental Observation(s) — presence-only, no absence inference
  Historical — aggregated legacy records, no reliable date

Usage:
  python prepare_ebutterfly.py                    # US+Canada+Mesoamerica, min 100 presences
  python prepare_ebutterfly.py --us_only          # Continental US only (excludes Alaska)
  python prepare_ebutterfly.py --min_presences 50
"""

import argparse
import os
import numpy as np
import pandas as pd

REPO_ROOT  = os.environ.get("REPO_ROOT",
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR    = os.environ.get("EBUTTERFLY_RAW", os.path.join(REPO_ROOT, "Examples", "eButterfly", "raw"))
EVENT_FILE = f"{RAW_DIR}/event.txt"
OCC_FILE   = f"{RAW_DIR}/occurrence.txt"

EXCLUDE_PROTOCOLS = {"Incidental Observation(s)", "Historical"}
EXCLUDE_PROVINCES = {"Hawaii"}

INCLUDE_COUNTRIES = {
    "United States", "Canada",
    # Mesoamerica
    "Mexico", "Guatemala", "Belize", "Honduras", "El Salvador",
    "Nicaragua", "Costa Rica", "Panama",
}


def main(min_presences=100, us_only=False, seed=42):
    # ── load events ───────────────────────────────────────────────────────────
    print("Loading events...")
    ev = pd.read_csv(EVENT_FILE, sep="\t", low_memory=False,
                     usecols=["id", "eventDate", "year", "country", "stateProvince",
                               "decimalLatitude", "decimalLongitude", "samplingProtocol"])

    # structured protocols only
    ev = ev[~ev.samplingProtocol.isin(EXCLUDE_PROTOCOLS)].copy()

    if us_only:
        # Continental US only — exclude Alaska and Hawaii
        ev = ev[ev.country == "United States"].copy()
        ev = ev[~ev.stateProvince.isin(EXCLUDE_PROVINCES | {"Alaska"})].copy()
        ev = ev[(ev.decimalLongitude >= -125) & (ev.decimalLongitude <= -66) &
                (ev.decimalLatitude  >=   24) & (ev.decimalLatitude  <=   50)].copy()
        region_label = "Continental US"
        out_suffix   = "_us"
    else:
        # US + Canada + Mesoamerica
        ev = ev[ev.country.isin(INCLUDE_COUNTRIES)].copy()
        # exclude Hawaii and non-continental islands
        ev = ev[~ev.stateProvince.isin(EXCLUDE_PROVINCES)].copy()
        # lon: -170 (Alaska) to -50; lat: 7 (Panama S) to 84 (Canada N)
        ev = ev[(ev.decimalLongitude >= -170) & (ev.decimalLongitude <= -50) &
                (ev.decimalLatitude  >=    7) & (ev.decimalLatitude  <=   84)].copy()
        region_label = "North America (US+Canada+Mesoamerica)"
        out_suffix   = "_na"

    # valid coordinates
    ev = ev.dropna(subset=["decimalLatitude", "decimalLongitude"]).copy()

    # parse date
    ev["date"] = pd.to_datetime(ev["eventDate"], errors="coerce", utc=True)
    ev = ev.dropna(subset=["date"]).copy()
    # sanity filter: plausible years only
    ev = ev[(ev.date.dt.year >= 1950) & (ev.date.dt.year <= 2026)].copy()

    print(f"{len(ev):,} checklists after filtering ({region_label})")
    print(f"Date range: {ev.date.dt.year.min()} – {ev.date.dt.year.max()}")
    print(f"Lat: {ev.decimalLatitude.min():.1f}–{ev.decimalLatitude.max():.1f}  "
          f"Lon: {ev.decimalLongitude.min():.1f}–{ev.decimalLongitude.max():.1f}")

    by_decade = ev.copy()
    by_decade["decade"] = (ev.date.dt.year // 10) * 10
    print("Checklists by decade:")
    for dec, n in by_decade.groupby("decade").size().items():
        print(f"  {dec}s: {n:,}")

    print("\nChecklists by country:")
    for ctry, n in ev.groupby("country").size().sort_values(ascending=False).items():
        print(f"  {ctry}: {n:,}")

    # ── load occurrences ──────────────────────────────────────────────────────
    print("\nLoading occurrences...")
    oc = pd.read_csv(OCC_FILE, sep="\t", low_memory=False,
                     usecols=["eventID", "scientificName", "taxonRank"])
    oc = oc[(oc.taxonRank == "SPECIES") & oc.eventID.isin(ev.id)].copy()
    oc["present"] = 1

    # ── build presence/absence matrix ────────────────────────────────────────
    print("Building species matrix...")
    sp_matrix = (
        oc.groupby(["eventID", "scientificName"])["present"]
        .max().unstack(fill_value=0).astype(np.float32)
    )

    # filter species by min presences
    keep = sp_matrix.columns[sp_matrix.sum() >= min_presences].tolist()
    print(f"Keeping {len(keep)} species with >= {min_presences} presences "
          f"(dropped {len(sp_matrix.columns) - len(keep)})")
    sp_matrix = sp_matrix[keep]

    # ── assemble meta ─────────────────────────────────────────────────────────
    meta_base = ev.set_index("id")[["date", "decimalLatitude", "decimalLongitude"]].copy()
    meta_base = meta_base.rename(columns={"decimalLatitude": "latitude",
                                          "decimalLongitude": "longitude"})

    # ── output 1: 2011–2025, time as ISO date string (YYYY-MM-DD) ────────────
    ev_post2010 = ev[(ev["date"].dt.year >= 2011) & (ev["date"].dt.year <= 2025)]
    meta_abs = meta_base.loc[meta_base.index.isin(ev_post2010.id)].copy()
    meta_abs["time"] = meta_abs["date"].dt.strftime("%Y-%m-%d")
    meta_abs = meta_abs[["time", "latitude", "longitude"]]

    # recompute species matrix on post-2010 rows only, re-apply min_presences
    sp_post = sp_matrix.loc[sp_matrix.index.isin(ev_post2010.id)]
    keep_post = sp_post.columns[sp_post.sum() >= min_presences].tolist()

    df_abs = pd.concat([meta_abs, sp_post[keep_post]], axis=1).dropna().reset_index(drop=True)
    assert not df_abs.isna().any().any()

    prev = df_abs[keep_post].mean()
    out_abs = f"ebutterfly{out_suffix}_2011_2025_jsdm.csv"
    print(f"\n[{out_abs}] 2011–2025, time = ISO date (YYYY-MM-DD)")
    print(f"  date range: {df_abs.time.min()} → {df_abs.time.max()}")
    print(f"  shape: {df_abs.shape}  ({len(keep_post)} species ≥{min_presences} presences in 2011–2025)")
    print(f"  prevalence: mean={prev.mean():.4f}, range=[{prev.min():.4f}, {prev.max():.4f}]")

    df_abs.to_csv(out_abs, index=False)
    print(f"  Saved → {out_abs}")

    # ── output 2: static version, all years, time = 0 ────────────────────────
    meta_static = meta_base.copy()
    meta_static["time"] = 0.0
    meta_static = meta_static[["time", "latitude", "longitude"]]

    df_static = pd.concat([meta_static, sp_matrix], axis=1).dropna().reset_index(drop=True)
    assert not df_static.isna().any().any()

    prev_s = df_static[keep].mean()
    out_static = f"ebutterfly{out_suffix}_static_jsdm.csv"
    print(f"\n[{out_static}] all years, time = 0 (static/compressed)")
    print(f"  shape: {df_static.shape}  ({len(keep)} species ≥{min_presences} presences)")
    print(f"  prevalence: mean={prev_s.mean():.4f}, range=[{prev_s.min():.4f}, {prev_s.max():.4f}]")

    df_static.to_csv(out_static, index=False)
    print(f"  Saved → {out_static}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min_presences", type=int, default=100)
    p.add_argument("--us_only", action="store_true",
                   help="Continental US only (excludes Alaska and Hawaii)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.min_presences, a.us_only, a.seed)
