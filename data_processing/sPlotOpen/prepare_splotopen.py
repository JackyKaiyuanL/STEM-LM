"""
Prepare sPlotOpen for ST-JSDM.

Output: splotopen_jsdm.csv
  Columns: time, latitude, longitude, <species...>

Data source: iDiv sPlotOpen v76 (Sabatini et al. 2021)
  - Uses Resample_1_consensus (climate-balanced subsample, ~43K plots globally)
  - Species: binary presence/absence (1 if recorded, 0 otherwise)
  - Time: 0.0 for all plots (static env dataset; Date_of_recording is unreliable)
  - No env features in this version — add later via a separate raster extract

No observation dates in SatButterfly (time=0 everywhere).
sPlotOpen has Date_of_recording for all consensus plots:
  - Global range: 1888–2015
  - US subset range: 1972–2013

Usage:
  python prepare_splotopen.py              # global, min 100 presences
  python prepare_splotopen.py --all_records  # global using all sPlotOpen records
  python prepare_splotopen.py --us_only    # US only (~8780 plots)
  python prepare_splotopen.py --europe     # Europe excl. Iceland + Svalbard (~17,208 plots)
  python prepare_splotopen.py --australia  # Australia only (~7304 plots)
  python prepare_splotopen.py --us_only --min_presences 50
  python prepare_splotopen.py --max_plots 20000  # subsample global
"""

import argparse
import os
import time
import numpy as np
import pandas as pd

REPO_ROOT   = os.environ.get("REPO_ROOT",
                              os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR     = os.environ.get("SPLOTOPEN_RAW",
                              os.path.join(REPO_ROOT, "Examples", "sPlotOpen", "raw",
                                           "iDiv Data Repository_3474_v76__20260408"))
HEADER_FILE = os.path.join(RAW_DIR, "sPlotOpen_header(3).txt")
DT_FILE     = os.path.join(RAW_DIR, "sPlotOpen_DT(2).txt")

# Europe: all countries in sPlotOpen Continent=="Europe" except islands excluded by user
# Lat cap at 72°N: North Cape (mainland Norway) is 71.1°N; anything above is Arctic islands
# (Bjørnøya/Bear Island labeled "Norway" at ~74°N, Jan Mayen, etc.)
EUROPE_EXCLUDE_COUNTRIES = {"Iceland", "Svalbard and Jan Mayen Is", "Faroe Islands"}
EUROPE_MAX_LAT = 72.0


def _log(msg, *, t0, quiet=False):
    if quiet:
        return
    dt = time.time() - t0
    print(f"[{dt:8.1f}s] {msg}", flush=True)


def _read_dt(plot_ids, *, dt_chunksize=None, quiet=False, t0=None):
    read_kwargs = dict(
        sep="\t",
        low_memory=False,
        usecols=["PlotObservationID", "Species"],
        encoding="latin-1",
        dtype={"PlotObservationID": "int64", "Species": "object"},
        memory_map=True,
    )

    if dt_chunksize:
        chunks = []
        total_rows = 0
        kept_rows = 0
        for k, chunk in enumerate(pd.read_csv(DT_FILE, chunksize=dt_chunksize, **read_kwargs), start=1):
            total_rows += len(chunk)
            chunk = chunk[chunk["PlotObservationID"].isin(plot_ids)]
            kept_rows += len(chunk)
            chunks.append(chunk)
            _log(f"DT chunk {k}: read {total_rows:,} rows, kept {kept_rows:,}", t0=t0, quiet=quiet)
        dt = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["PlotObservationID", "Species"])
    else:
        dt = pd.read_csv(DT_FILE, **read_kwargs)
        dt = dt[dt["PlotObservationID"].isin(plot_ids)].copy()

    dt = dt.dropna(subset=["Species"])
    # Ensure one row per (plot, species) so counts == presence.
    dt = dt.drop_duplicates(["PlotObservationID", "Species"])
    return dt


def _build_species_matrix(dt, *, plot_index, keep_species, quiet=False, t0=None):
    # Prefer a sparse build if SciPy is available (faster/lower memory), but
    # output CSV still ends up dense on disk.
    try:
        from scipy.sparse import coo_matrix

        _log("Building sparse plot×species matrix (SciPy)...", t0=t0, quiet=quiet)
        plot_codes = pd.Categorical(dt["PlotObservationID"], categories=plot_index).codes
        sp_codes = pd.Categorical(dt["Species"], categories=keep_species).codes
        mask = (plot_codes >= 0) & (sp_codes >= 0)
        plot_codes = plot_codes[mask]
        sp_codes = sp_codes[mask]
        data = np.ones(len(plot_codes), dtype=np.uint8)
        mat = coo_matrix(
            (data, (plot_codes, sp_codes)),
            shape=(len(plot_index), len(keep_species)),
            dtype=np.uint8,
        ).tocsr()
        # If duplicates exist, force binary.
        mat.data[:] = 1
        species_matrix = pd.DataFrame.sparse.from_spmatrix(mat, index=plot_index, columns=keep_species)
        _log(f"Matrix: shape={mat.shape}, nnz={mat.nnz:,}", t0=t0, quiet=quiet)
        return species_matrix
    except Exception as e:
        _log(f"SciPy sparse build unavailable/failed ({e}); falling back to pandas crosstab...", t0=t0, quiet=quiet)

    species_matrix = pd.crosstab(dt["PlotObservationID"], dt["Species"]).reindex(plot_index, fill_value=0)
    species_matrix = species_matrix.reindex(columns=keep_species, fill_value=0).astype(np.uint8)
    return species_matrix


def main(min_presences=100, us_only=False, europe=False, australia=False,
         all_records=False, max_plots=None, seed=42,
         dt_chunksize=None, write_chunk_rows=None, quiet=False, progress=False):
    t0 = time.time()
    # ── load header ──────────────────────────────────────────────────────────
    _log("Loading header...", t0=t0, quiet=quiet)
    header = pd.read_csv(HEADER_FILE, sep="\t", low_memory=False,
                         usecols=["PlotObservationID", "Date_of_recording",
                                  "Continent", "Country",
                                  "Latitude", "Longitude", "Resample_1_consensus"],
                         dtype={"PlotObservationID": "int64"},
                         memory_map=True)
    if all_records:
        _log(f"{len(header):,} plots loaded (no Resample_1_consensus filter)", t0=t0, quiet=quiet)
        header = header.dropna(subset=["Latitude", "Longitude"]).copy()
        _log(f"{len(header):,} plots with valid coordinates", t0=t0, quiet=quiet)
    else:
        header = header[header["Resample_1_consensus"] == True].copy()
        _log(f"{len(header):,} plots after Resample_1_consensus filter", t0=t0, quiet=quiet)

        header["date"] = pd.to_datetime(header["Date_of_recording"], errors="coerce")
        header = header.dropna(subset=["date", "Latitude", "Longitude"])
        _log(f"{len(header):,} plots with valid date and coordinates", t0=t0, quiet=quiet)

    if us_only:
        header = header[
            (header["Latitude"] >= 24) & (header["Latitude"] <= 50) &
            (header["Longitude"] >= -125) & (header["Longitude"] <= -66)
        ].copy()
        _log(f"{len(header):,} plots after US filter", t0=t0, quiet=quiet)

    elif europe:
        header = header[header["Continent"] == "Europe"].copy()
        header = header[~header["Country"].isin(EUROPE_EXCLUDE_COUNTRIES)].copy()
        header = header[header["Latitude"] <= EUROPE_MAX_LAT].copy()
        _log(f"{len(header):,} plots after Europe filter "
             f"(excl. Iceland, Svalbard, Faroe Is., lat > {EUROPE_MAX_LAT}°N)", t0=t0, quiet=quiet)
        _log("Top countries:", t0=t0, quiet=quiet)
        for ctry, n in header["Country"].value_counts().head(10).items():
            _log(f"  {ctry}: {n:,}", t0=t0, quiet=quiet)

    elif australia:
        header = header[header["Country"] == "Australia"].copy()
        _log(f"{len(header):,} plots after Australia filter", t0=t0, quiet=quiet)

    else:
        pass  # global — no region filter

    if max_plots and len(header) > max_plots:
        header = header.sample(n=max_plots, random_state=seed)
        _log(f"Subsampled to {max_plots:,} plots", t0=t0, quiet=quiet)

    header["time"] = 0.0
    _log("time set to 0.0 for all plots (static env dataset)", t0=t0, quiet=quiet)

    # ── load species ─────────────────────────────────────────────────────────
    plot_index = pd.Index(header["PlotObservationID"].values, name="PlotObservationID")
    plot_ids = set(plot_index.tolist())
    if progress and dt_chunksize is None:
        dt_chunksize = 2_000_000
    _log(f"Loading species for {len(plot_ids):,} plots (dt_chunksize={dt_chunksize})...", t0=t0, quiet=quiet)
    dt = _read_dt(plot_ids, dt_chunksize=dt_chunksize, quiet=quiet, t0=t0)
    _log(f"DT filtered unique pairs: {len(dt):,} (plot,species)", t0=t0, quiet=quiet)

    sp_counts = dt["Species"].value_counts()
    keep = sp_counts.index[sp_counts >= min_presences].tolist()
    dropped = len(sp_counts) - len(keep)
    _log(f"Keeping {len(keep):,} species with >= {min_presences} presences (dropped {dropped:,})", t0=t0, quiet=quiet)
    dt = dt[dt["Species"].isin(keep)].copy()

    n_cells = len(plot_index) * max(1, len(keep))
    if not quiet:
        _log(f"Dense output size: {len(plot_index):,} plots × {len(keep):,} species = {n_cells:,} cells", t0=t0, quiet=quiet)
        if n_cells >= 250_000_000:
            _log("WARNING: this will be very large/slow to write as CSV; consider --min_presences, --max_plots.", t0=t0, quiet=quiet)

    # binary presence/absence matrix: plots × species
    species_matrix = _build_species_matrix(dt, plot_index=plot_index, keep_species=keep, quiet=quiet, t0=t0)

    # ── assemble output ───────────────────────────────────────────────────────
    meta = (header.set_index("PlotObservationID")
            [["time", "Latitude", "Longitude"]]
            .rename(columns={"Latitude": "latitude", "Longitude": "longitude"}))

    # Ensure all plots in meta exist in the matrix (plots with zero kept species get all-zeros).
    species_matrix = species_matrix.reindex(meta.index, fill_value=0)

    df = pd.concat([meta, species_matrix], axis=1).dropna().reset_index(drop=True)
    assert not df.isna().any().any(), "NaNs in output — check join"

    print(f"Final shape: {df.shape}")
    prev = (sp_counts.loc[keep] / len(plot_index)).astype(float)
    print(f"Prevalence: mean={prev.mean():.4f}, "
          f"range=[{prev.min():.4f}, {prev.max():.4f}]")
    print(f"Lat: {df.latitude.min():.1f}–{df.latitude.max():.1f}, "
          f"Lon: {df.longitude.min():.1f}–{df.longitude.max():.1f}")

    if us_only:
        suffix = "_us"
    elif europe:
        suffix = "_europe"
    elif australia:
        suffix = "_australia"
    else:
        suffix = ""
    if all_records and suffix == "":
        out = "splotopen_global_all_jsdm.csv"
    elif all_records:
        out = f"splotopen{suffix}_all_jsdm.csv"
    else:
        out = f"splotopen{suffix}_jsdm.csv"

    if progress and write_chunk_rows is None:
        write_chunk_rows = 2000

    _log(f"Writing CSV → {out} (write_chunk_rows={write_chunk_rows})...", t0=t0, quiet=quiet)
    if write_chunk_rows:
        # Chunked write gives progress visibility for huge outputs.
        n = len(df)
        for start in range(0, n, write_chunk_rows):
            end = min(n, start + write_chunk_rows)
            mode = "w" if start == 0 else "a"
            header_flag = (start == 0)
            df.iloc[start:end].to_csv(out, index=False, mode=mode, header=header_flag)
            _log(f"Wrote rows {end:,}/{n:,}", t0=t0, quiet=quiet)
    else:
        df.to_csv(out, index=False)
    _log(f"Saved → {out}", t0=t0, quiet=quiet)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min_presences", type=int, default=100,
                   help="Min number of presences to keep a species (default: 100)")
    p.add_argument("--us_only", action="store_true",
                   help="Filter to continental US (lat 24–50, lon -125 to -66)")
    p.add_argument("--europe", action="store_true",
                   help="Filter to Europe, excluding Iceland and Svalbard (UK+Ireland included)")
    p.add_argument("--australia", action="store_true",
                   help="Filter to Australia only")
    p.add_argument("--all_records", action="store_true",
                   help="Use all sPlotOpen records (skip Resample_1_consensus and allow missing dates)")
    p.add_argument("--max_plots", type=int, default=None,
                   help="Randomly subsample to this many plots (for memory control)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dt_chunksize", type=int, default=None,
                   help="Read sPlotOpen_DT in chunks of this many rows (enables progress prints; may be slower)")
    p.add_argument("--write_chunk_rows", type=int, default=None,
                   help="Write output CSV in row chunks of this size (enables progress prints; may be slower)")
    p.add_argument("--progress", action="store_true",
                   help="Enable extra progress logging (implies dt_chunksize=2e6 and write_chunk_rows=2000 if unset)")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce logging output")
    a = p.parse_args()
    main(
        a.min_presences,
        a.us_only,
        a.europe,
        a.australia,
        a.all_records,
        a.max_plots,
        a.seed,
        dt_chunksize=a.dt_chunksize,
        write_chunk_rows=a.write_chunk_rows,
        quiet=a.quiet,
        progress=a.progress,
    )
