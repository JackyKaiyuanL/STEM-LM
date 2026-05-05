"""
Download MODIS MOD13Q1 v061 (16-day 250m NDVI/EVI) over NA+Mesoamerica via earthaccess.

Requires:
  pip install earthaccess
  Earthdata Login credentials: either a ~/.netrc entry
    machine urs.earthdata.nasa.gov login USER password PASS
  or earthaccess.login() will prompt on first run and write ~/.netrc for you.

Usage (from ${REPO_ROOT}/Examples/env_vars/modis_phenology):
  python download_mod13q1_na.py --start 2011-01-01 --end 2025-12-31 --outdir raw
"""
import argparse, os, sys

import earthaccess

# NA + Mesoamerica bbox (W, S, E, N) — matches ERA5-Land script
BBOX = (-170.0, 7.0, -50.0, 72.0)
SHORT_NAME = "MOD13Q1"
VERSION = "061"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end",   default="2025-12-31")
    ap.add_argument("--outdir", default="raw")
    ap.add_argument("--threads", type=int, default=8,
                    help="parallel download threads (earthaccess default is 8)")
    ap.add_argument("--dry_run", action="store_true",
                    help="only list granule count, do not download")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Logging in to Earthdata...")
    auth = earthaccess.login(strategy="netrc", persist=True)
    if not auth.authenticated:
        print("Auth failed; ensure ~/.netrc has a urs.earthdata.nasa.gov entry.", file=sys.stderr)
        sys.exit(1)

    print(f"Searching CMR: {SHORT_NAME} v{VERSION}  bbox={BBOX}  {args.start} → {args.end}")
    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        bounding_box=BBOX,
        temporal=(args.start, args.end),
    )
    print(f"Found {len(results)} granules")

    # earthaccess default pulls every URL in a granule — including BROWSE .jpg thumbnails.
    # Keep only the .hdf data file per granule.
    hdf_urls = []
    for g in results:
        for url in g.data_links():
            if url.lower().endswith(".hdf"):
                hdf_urls.append(url)
                break
    print(f"Filtered to {len(hdf_urls)} HDF URLs (dropped browse/metadata)")

    if args.dry_run:
        for url in hdf_urls[:5]:
            print(" ", url)
        return

    fs = earthaccess.get_fsspec_https_session()
    os.makedirs(args.outdir, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def fetch(url):
        name = url.rsplit("/", 1)[-1]
        out = os.path.join(args.outdir, name)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return f"[skip] {name}"
        tmp = out + ".part"
        with fs.open(url, "rb") as src, open(tmp, "wb") as dst:
            for chunk in iter(lambda: src.read(4 * 1024 * 1024), b""):
                dst.write(chunk)
        os.replace(tmp, out)
        return f"[done] {name}"
    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for fut in as_completed([ex.submit(fetch, u) for u in hdf_urls]):
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"[err] {e}"
            done += 1
            if done % 100 == 0 or msg.startswith("[err]"):
                print(f"({done}/{len(hdf_urls)}) {msg}", flush=True)
    print(f"Downloaded / resumed into {args.outdir}/")


if __name__ == "__main__":
    main()
