"""
Prepare NEUS bottom-trawl hauls for ST-JSDM.

Source: FISHGLOB v1 (Maureaud et al. 2024), survey NEUS = NOAA NEFSC Northeast
        US bottom-trawl survey. 36,481 hauls / 416,850 occurrences / 560 taxa,
        1963-2020, Cape Hatteras to the Gulf of Maine.
        Ships as RData only; stage 1 shells out to Rscript.

Output: neus_nefsc.csv
  Columns: time, latitude, longitude, env_depth, env_sbt, env_sst, <species...>
  One row = one haul (presence/absence for each species)

  time      ISO date (YYYY-MM-DD) from year/month/day
  env_depth bottom depth, m          (99.9% of hauls)
  env_sbt   bottom temperature, C    (87.3%)
  env_sst   surface temperature, C   (89.1%)

Zero-fill is against the NEUS taxon list only. Species are kept at
--min_presences hauls; presence is row existence, not num or wgt.

Seasons: Spring (Feb-Jun) and Fall (Sep-Dec). No January, July or August.
Gear changed in 2009 (Albatross IV -> Henry B. Bigelow, 30 -> 20 min tows);
mean richness/haul steps 10.56 -> 14.54. Use --year_max 2008 for one gear era.

Usage:
  python prepare_neus.py \\
      --fishglob ${REPO_ROOT}/data/FishGlob_data/outputs/Compiled_data/FishGlob_public_clean.RData \\
      --workdir  ${REPO_ROOT}/data/_neus_tmp \\
      --out      ${REPO_ROOT}/data/neus_nefsc.csv \\
      --year_min 1970 --year_max 2019 --min_presences 50 --zero_temp_is_missing
"""

import argparse
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd

ENV = ["depth", "sbt", "sst"]

EXTRACT_R = textwrap.dedent("""
    args <- commandArgs(trailingOnly = TRUE)
    load(args[1])
    s <- as.data.frame(data)
    d <- s[s$survey == "NEUS", ]
    num <- function(x) as.numeric(as.character(x))
    for (v in c("year", "month", "day", "depth", "sbt", "sst")) d[[v]] <- num(d[[v]])
    hauls <- unique(d[, c("haul_id", "year", "month", "day", "latitude", "longitude",
                          "depth", "sbt", "sst", "survey_unit", "station", "stratum")])
    occ <- unique(d[, c("haul_id", "accepted_name")])
    write.csv(hauls, file.path(args[2], "hauls.csv"), row.names = FALSE)
    write.csv(occ,   file.path(args[2], "occ.csv"),   row.names = FALSE)
    cat(sprintf("hauls %d  occurrences %d  taxa %d\\n",
                nrow(hauls), nrow(occ), length(unique(occ$accepted_name))))
""")


def extract(fishglob, workdir):
    os.makedirs(workdir, exist_ok=True)
    script = os.path.join(workdir, "_extract.R")
    with open(script, "w") as f:
        f.write(EXTRACT_R)
    r = subprocess.run(["Rscript", script, fishglob, workdir],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"Rscript failed:\n{r.stderr}")
    print(r.stdout.strip())


def main(a):
    if not os.path.exists(os.path.join(a.workdir, "hauls.csv")) or a.force_extract:
        extract(a.fishglob, a.workdir)

    h = pd.read_csv(os.path.join(a.workdir, "hauls.csv"))
    o = pd.read_csv(os.path.join(a.workdir, "occ.csv"))

    h = h[(h.year >= a.year_min) & (h.year <= a.year_max)]
    n0 = len(h)

    if a.zero_temp_is_missing:
        for v in ("sbt", "sst"):
            n_zero = int((h[v] == 0).sum())
            h.loc[h[v] == 0, v] = np.nan
            print(f"{v}: {n_zero} hauls at exactly 0.00 set to missing")

    h = h.dropna(subset=["latitude", "longitude", "year", "month", "day"] + ENV)
    print(f"hauls {n0} -> {len(h)} after dropping incomplete env/date "
          f"({100 * (n0 - len(h)) / n0:.1f}% dropped)")

    t = pd.to_datetime(dict(year=h.year.astype(int), month=h.month.astype(int),
                            day=h.day.astype(int)), errors="coerce")
    h = h[t.notna()].copy()
    h["time"] = t[t.notna()].dt.strftime("%Y-%m-%d")

    o = o[o.haul_id.isin(h.haul_id)]
    counts = o.accepted_name.value_counts()
    keep = sorted(counts[counts >= a.min_presences].index)
    print(f"taxa {len(keep)} at >= {a.min_presences} hauls")

    pa = (pd.crosstab(o[o.accepted_name.isin(keep)].haul_id,
                      o[o.accepted_name.isin(keep)].accepted_name) > 0).astype(np.float32)
    pa = pa.reindex(index=h.haul_id, columns=keep, fill_value=0.0)

    df = pd.concat(
        [h.set_index("haul_id")[["time", "latitude", "longitude"]],
         h.set_index("haul_id")[ENV].add_prefix("env_").astype(np.float32),
         pa],
        axis=1,
    ).reset_index(drop=True)

    assert not df.isna().any().any(), "NaNs in output"
    print(f"shape {df.shape} | species {len(keep)} | "
          f"prevalence {df[keep].mean().min():.4f}-{df[keep].mean().max():.4f} | "
          f"richness/haul mean {df[keep].to_numpy().sum(1).mean():.2f}")
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fishglob", required=True)
    p.add_argument("--workdir", required=True, help="scratch dir for the R extraction")
    p.add_argument("--out", required=True)
    p.add_argument("--min_presences", type=int, default=50)
    p.add_argument("--year_min", type=int, default=1963)
    p.add_argument("--year_max", type=int, default=2019)
    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--zero_temp_is_missing", action="store_true",
                   help="treat sbt/sst == 0.00 as missing (104 hauls, all 1967-1990)")
    main(p.parse_args())
