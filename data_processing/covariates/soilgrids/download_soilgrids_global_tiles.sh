#!/bin/bash
# Download SoilGrids v2.0 (0-5cm, mean) for (near-)global coverage using tiles.
#
# This avoids trying to fetch a single huge GeoTIFF, and instead builds a VRT
# mosaic per variable:
#   raw/{var}_0-5cm_global.vrt
#
# Notes:
# - SoilGrids coverage is roughly lat [-60, 90]. We use [-60, 89.999] here.
# - SCALEFACTOR controls server-side downsampling; 0.25 roughly matches the
#   existing regional downloads in this repo.
#
# Usage:
#   bash download_soilgrids_global_tiles.sh
#   SCALEFACTOR=0.25 bash download_soilgrids_global_tiles.sh
#   bash download_soilgrids_global_tiles.sh --jobs 16
#   JOBS=16 bash download_soilgrids_global_tiles.sh
#
# Output:
#   ${REPO_ROOT}/Examples/env_vars/soilgrids/raw/tiles/global/*.tif
#   ${REPO_ROOT}/Examples/env_vars/soilgrids/raw/{var}_0-5cm_global.vrt

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/Examples/env_vars/soilgrids/raw}"
TILEDIR="${OUTDIR}/tiles/global"
VARS="bdod cec cfvo clay nitrogen phh2o sand silt"
BASE="https://maps.isric.org/mapserv?map=/map"

SCALEFACTOR="${SCALEFACTOR:-0.25}"
CURL_OPTS=( -f -L --retry 5 --retry-delay 3 --connect-timeout 30 --progress-bar )
JOBS="${JOBS:-4}"

usage() {
  cat <<EOF
Usage:
  bash download_soilgrids_global_tiles.sh [--jobs N]

Env vars:
  SCALEFACTOR=0.25   Server-side downsampling factor (default: 0.25)
  JOBS=16            Parallel tile downloads (default: 4)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "Invalid --jobs/JOBS value: $JOBS" >&2
  exit 2
fi

# Tile grid: 4 lon bins × 3 lat bins = 12 tiles.
LON_BINS=(-180 -90 0 90 180)
LAT_BINS=(-60 -10 40 89.999)

mkdir -p "$TILEDIR"

echo "SoilGrids global tiling download"
echo "  OUTDIR=$OUTDIR"
echo "  TILEDIR=$TILEDIR"
echo "  SCALEFACTOR=$SCALEFACTOR"
echo "  JOBS=$JOBS"
echo "  vars: $VARS"
echo

download_tile() {
  local var="$1" lon_min="$2" lon_max="$3" lat_min="$4" lat_max="$5"
  local out="${TILEDIR}/${var}_0-5cm_global_lon${lon_min}_${lon_max}_lat${lat_min}_${lat_max}.tif"

  if [ -f "$out" ]; then
    echo "  SKIP tile lon[$lon_min,$lon_max] lat[$lat_min,$lat_max] (exists)"
    return 0
  fi

  echo "  Downloading tile lon[$lon_min,$lon_max] lat[$lat_min,$lat_max] ..."
  local url="${BASE}/${var}.map&SERVICE=WCS&VERSION=2.0.1\
&REQUEST=GetCoverage\
&COVERAGEID=${var}_0-5cm_mean\
&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326\
&SUBSET=X(${lon_min},${lon_max})\
&SUBSET=Y(${lat_min},${lat_max})\
&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326\
&SCALEFACTOR=${SCALEFACTOR}\
&FORMAT=image/tiff"

  local tmp
  tmp="$(mktemp "${out}.part.XXXXXX")"
  if curl "${CURL_OPTS[@]}" -o "$tmp" "$url"; then
    mv -f "$tmp" "$out"
    echo "    -> $(du -sh "$out" | cut -f1)  $out"
  else
    rm -f "$tmp"
    echo "    ERROR: failed tile lon[$lon_min,$lon_max] lat[$lat_min,$lat_max]" >&2
    return 1
  fi
}

for VAR in $VARS; do
  echo "=== $VAR ==="

  # Download tiles
  active=0
  failed=0
  for ((i=0; i<${#LON_BINS[@]}-1; i++)); do
    LON_MIN="${LON_BINS[$i]}"
    LON_MAX="${LON_BINS[$((i+1))]}"
    for ((j=0; j<${#LAT_BINS[@]}-1; j++)); do
      LAT_MIN="${LAT_BINS[$j]}"
      LAT_MAX="${LAT_BINS[$((j+1))]}"
      download_tile "$VAR" "$LON_MIN" "$LON_MAX" "$LAT_MIN" "$LAT_MAX" &
      active=$((active + 1))
      if [ "$active" -ge "$JOBS" ]; then
        if ! wait -n; then
          failed=1
        fi
        active=$((active - 1))
      fi
    done
  done
  while [ "$active" -gt 0 ]; do
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  done
  if [ "$failed" -ne 0 ]; then
    echo "ERROR: one or more downloads failed for $VAR" >&2
    exit 1
  fi

  # Build VRT mosaic
  VRT="${OUTDIR}/${VAR}_0-5cm_global.vrt"
  echo "  Building VRT: $VRT"
  gdalbuildvrt -overwrite "$VRT" "${TILEDIR}/${VAR}_0-5cm_global_lon"*"_lat"*.tif
done

echo
echo "Done. Global VRTs:"
ls -lh "$OUTDIR"/*_0-5cm_global.vrt
