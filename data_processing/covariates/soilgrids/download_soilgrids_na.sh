#!/bin/bash
# Download SoilGrids v2.0 (0-5cm) for North America + Mesoamerica region
# Bounding box: lon -170 to -50, lat 7 to 72
# Covers eButterfly NA dataset (US, Canada, Mexico, Mesoamerica)
# Output: raw/{var}_0-5cm_na.tif

set -e
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/Examples/env_vars/soilgrids/raw}"
VARS="bdod cec cfvo clay nitrogen phh2o sand silt"
BASE="https://maps.isric.org/mapserv?map=/map"

# NA bounding box
LON_MIN=-170
LON_MAX=-50
LAT_MIN=7
LAT_MAX=72

for VAR in $VARS; do
    OUT="$OUTDIR/${VAR}_0-5cm_na.tif"
    if [ -f "$OUT" ]; then
        echo "SKIP $VAR (already exists)"
        continue
    fi
    echo "Downloading $VAR ..."
    URL="${BASE}/${VAR}.map&SERVICE=WCS&VERSION=2.0.1\
&REQUEST=GetCoverage\
&COVERAGEID=${VAR}_0-5cm_mean\
&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326\
&SUBSET=X(${LON_MIN},${LON_MAX})\
&SUBSET=Y(${LAT_MIN},${LAT_MAX})\
&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326\
&SCALEFACTOR=0.25\
&FORMAT=image/tiff"
    curl -f -L -o "$OUT" "$URL"
    SIZE=$(du -sh "$OUT" | cut -f1)
    echo "  -> $OUT ($SIZE)"
done

echo "Done. Files in $OUTDIR:"
ls -lh "$OUTDIR"/*_na.tif
