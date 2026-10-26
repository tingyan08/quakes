#!/bin/bash
# DSM + DTM for each method from its point cloud, with ONE PDAL pipeline, so the
# methods differ only in their points:
#   ODM   : odm_georeferencing/odm_georeferenced_model.laz (OpenSfM + OpenMVS)
#   3DGS  : back-projected expected depth (render_dsm.py --dump_points)
#   2DGS  : back-projected median depth   (render_dsm.py --dump_points)
#   ...   : any other <name>_points.csv in the output directory
#
# Pipeline: statistical outlier removal -> DSM = mean Z of all points;
#           SMRF ground filter (ODM's --dtm defaults) -> DTM = mean Z of ground points.
# Grid: the ODM DSM grid decimated 4x (~2.4 m), so everything is cell-aligned.
#
# Usage: bash gs/make_dsm_dtm.sh
# Needs PDAL 2.9 from the photogrammetry image; re-runs itself inside it.
set -euo pipefail
. "$(dirname "$0")/common.sh"
PDAL=/opt/opendronemap/SuperBuild/install/bin/pdal        # ODM's PDAL in the JPL image
if [[ ! -x $PDAL ]]; then
    exec apptainer exec -B /scratch,/depot "$SIF" bash "$0" "$@"
fi
export LD_LIBRARY_PATH=/opt/opendronemap/SuperBuild/build/pdal/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

PROJ=$PHOTOGRAMMETRY
OUT=${OUT:-$RUNS/dsm-dtm-compare}
FACTOR=4
mkdir -p $OUT

# Grid from the ODM DSM (writers.gdal wants the LOWER-left origin)
read RES OX OY W H < <(/usr/bin/python3 - <<EOF   # the image's python (GDAL), not envs/gs on PATH
from osgeo import gdal
d = gdal.Open("$PROJ/odm_dem/dsm.tif")
gt = d.GetGeoTransform()
W, H = d.RasterXSize // $FACTOR, d.RasterYSize // $FACTOR
res = gt[1] * d.RasterXSize / W
print(res, gt[0], gt[3] - res * H, W, H)
EOF
)
echo "grid: ${W}x${H} @ ${RES} m, origin ($OX, $OY)"

# ODM's --dtm defaults (opendm/config.py)
SMRF='"scalar": 1.25, "slope": 0.15, "threshold": 0.5, "window": 18.0'

run() {
    local name=$1 reader=$2
    echo "=== $name"
    cat > $OUT/$name.pipeline.json <<EOF
[
  $reader,
  { "type": "filters.assign", "value": "Classification = 0" },
  { "type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 3.0 },
  { "type": "filters.range", "limits": "Classification![7:7]" },
  { "type": "writers.gdal", "filename": "$OUT/${name}_dsm.tif", "output_type": "mean",
    "resolution": $RES, "origin_x": $OX, "origin_y": $OY, "width": $W, "height": $H,
    "window_size": 3, "nodata": -9999, "data_type": "float32", "gdaldriver": "GTiff", "override_srs": "EPSG:32611" },
  { "type": "filters.smrf", $SMRF, "cell": $RES },
  { "type": "filters.range", "limits": "Classification[2:2]" },
  { "type": "writers.gdal", "filename": "$OUT/${name}_dtm.tif", "output_type": "mean",
    "resolution": $RES, "origin_x": $OX, "origin_y": $OY, "width": $W, "height": $H,
    "window_size": 10, "nodata": -9999, "data_type": "float32", "gdaldriver": "GTiff", "override_srs": "EPSG:32611" }
]
EOF
    $PDAL pipeline $OUT/$name.pipeline.json
}

# ODM's dense cloud, then every <name>_points.csv from render_dsm.py --dump_points.
# METHODS="3dgs-f2" (space-separated) limits the run to those names.
want() { [[ -z "${METHODS:-}" || " $METHODS " == *" $1 "* ]]; }
if want odm; then
    run odm "{ \"type\": \"readers.las\", \"filename\": \"$PROJ/odm_georeferencing/odm_georeferenced_model.laz\" }"
fi
for csv in "$OUT"/*_points.csv; do
    name=$(basename "$csv" _points.csv)
    if want "$name"; then
        run "$name" "{ \"type\": \"readers.text\", \"filename\": \"$csv\" }"
    fi
done
ls -la $OUT/*.tif
