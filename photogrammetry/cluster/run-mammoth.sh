#!/bin/bash
#SBATCH --job-name=quakes-mammoth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --output=runs/mammoth/logs/%x-%j.out

# OpenSfM + ODM on the 2021-06-21 Mammoth Mtn / Mammoth Lakes pass (batch11), inside
# Dima Kogan's JPL QUAKES image (envs/quakes-jpl-trixie.sif) through quakes-deploy/scripts/quakes-jpl.sh.
# Default block: the wings-level stretch, frames 28875-28995, all 8 cameras (968 images).
# Submit from the project root (pass -A/-p yourself):
#   sbatch -A jahansha -p a100-40gb photogrammetry/cluster/run-mammoth.sh
#
# With GCPS=path/to/gcps.json (the format of run-whitney.sh: "lat lon elevation_ft" ->
# [[regex, pixel_x, pixel_y], ...]) this runs photogrammetry/run-opensfm-opendronemap-handoff.sh
# unchanged. Without GCPs it runs the same commands in the same order, minus --gcps:
# preprocess-opensfm.py always writes gcp_list.txt when given --gcps, and ODM refuses an
# empty one. Without GCPs the bundle adjustment uses the INS positions instead
# (bundle_use_gps: true); the config is otherwise the one of run-whitney.sh (the wiki's).
#
# Environment overrides:
#   SIF         the apptainer image. Default: envs/quakes-jpl-trixie.sif
#   DATA        directory with filename_latlonalt_opkdeg--cam*.vnl and images/.
#               Default: data/mammoth-2021-06-21/batch11 (JPL's 2021 conversion)
#   PROJECT     output project directory. Removed and recreated if FRESH=1.
#               Default: runs/mammoth/photogrammetry
#   FIRST,LAST  the frame range to process. Default: 28875, 28995
#   CAMERA_IDS  cameras to use. Default: all 8
#   GCPS        a gcps.json (see above). Default: none
#   RIG=1       pass --rig to preprocess-opensfm.py: the 8 cameras become one OpenSfM rig,
#               relative rotations from the INS-derived poses of one instant, one pose per shutter time.
#               ODM's bundled OpenSfM cannot move rig shots (export_geocoords: "Cannot set an
#               immutable pose"), so before ODM cluster/flatten-rig.py rewrites reconstruction.json
#               with each shot on its own (same poses; the rig one is kept as reconstruction.rig.json).
#               Not possible with GCPS (the handoff script runs OpenSfM and ODM in one go)
#   OPTIMIZE_RIG=1  with RIG=1, let bundle adjustment refine the rig (optimize_rig_parameters: true);
#               otherwise the camera-to-camera rotations stay fixed
#   MAX_ALTITUDE  OpenSfM clamps GPS altitudes to 10 km (opensfm/exif.py maximum_altitude);
#               QUAKES flies at ~13 km. The opensfm calls go through cluster/opensfm-altitude/opensfm,
#               which raises the cap to this value [m] at run time. Default: 100000.
#               MAX_ALTITUDE=off runs the image's /usr/bin/opensfm unchanged (10 km cap)

set -e

PHOTO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # photogrammetry/
# sbatch runs a copy of this script from the spool directory: use the submit directory
[[ -n "$SLURM_SUBMIT_DIR" && -f "$SLURM_SUBMIT_DIR/photogrammetry/cluster/run-mammoth.sh" ]] && PHOTO=$SLURM_SUBMIT_DIR/photogrammetry
ROOT=$(dirname "$PHOTO")                                 # project root
SIF=${SIF:-$ROOT/envs/quakes-jpl-trixie.sif}
DATA=${DATA:-$ROOT/data/mammoth-2021-06-21/batch11}
PROJECT=${PROJECT:-$ROOT/runs/mammoth/photogrammetry}
FIRST=${FIRST:-28875}
LAST=${LAST:-28995}
CAMERA_IDS=${CAMERA_IDS:-SN330068 SN330069 SN330070 SN330071 SN330074 SN330075 SN330076 SN330077}
NPROC=${SLURM_CPUS_PER_TASK:-$(nproc)}
rigopt=(); [[ "$RIG" == 1 ]] && rigopt=(--rig)
MAX_ALTITUDE=${MAX_ALTITUDE:-100000}
optimize_rig=false; [[ "$RIG" == 1 && "$OPTIMIZE_RIG" == 1 ]] && optimize_rig=true

[[ "$RIG" == 1 && -n "$GCPS" ]] && { echo "RIG=1 with GCPS is not supported: ODM fails on rig reconstructions" >&2; exit 1; }
[[ "$FRESH" == 1 ]] && rm -rf "$PROJECT"
mkdir -p "$PROJECT/images-block"

for f in "$DATA"/images/*.jpg; do
    b=$(basename "$f")
    frame=${b%%_*}
    sn=${b##*_}; sn=${sn%.jpg}
    if (( frame >= FIRST && frame <= LAST )) && [[ " $CAMERA_IDS " == *" $sn "* ]]; then
        ln -sf "$f" "$PROJECT/images-block/$b"
    fi
done
echo "Using $(ls "$PROJECT/images-block" | wc -l) images, frames $FIRST-$LAST, cameras $CAMERA_IDS${rigopt:+, rig (optimize: $optimize_rig)}, OpenSfM altitude cap: $MAX_ALTITUDE"

if [[ -n "$GCPS" ]]; then
    cp "$GCPS" "$PROJECT/gcps.json"
    bundle_use_gps=false
else
    rm -f "$PROJECT/gcps.json"
    bundle_use_gps=true
fi

cat > "$PROJECT/config.yaml" <<EOF
align_method: auto
align_orientation_prior: no_roll
bundle_use_gcp: true
bundle_use_gps: $bundle_use_gps
flann_algorithm: KDTREE
matcher_type: FLANN
matching_gps_distance: 0
matching_gps_neighbors: 0
matching_graph_rounds: 3
optimize_camera_parameters: true
optimize_rig_parameters: $optimize_rig
processes: $NPROC
reconstruction_algorithm: triangulation
triangulation_type: ROBUST
undistorted_image_format: tif
undistorted_image_max_size: 5120
use_altitude_tag: true
EOF

altpath=(); [[ "$MAX_ALTITUDE" != off ]] && altpath=(APPTAINERENV_PREPEND_PATH="$PHOTO/cluster/opensfm-altitude" APPTAINERENV_OPENSFM_MAX_ALTITUDE="$MAX_ALTITUDE")
Q() { env "${altpath[@]}" APPTAINERENV_ODM_MAX_CONCURRENCY=$NPROC QUAKES_SIF="$SIF" "$ROOT/quakes-deploy/scripts/quakes-jpl.sh" "$@"; }

if [[ -n "$GCPS" ]]; then
    Q "$PHOTO/run-opensfm-opendronemap-handoff.sh" \
        --metadata   "$DATA/*.vnl" \
        --images     "$PROJECT/images-block" \
        --camera-ids "$CAMERA_IDS" \
        --project    "$PROJECT" \
        "${rigopt[@]}"
else
    # = run-opensfm-opendronemap-handoff.sh without --gcps
    Q "$PHOTO/preprocess-opensfm.py" \
        --metadata   "$DATA/*.vnl" \
        --images     "$PROJECT/images-block" \
        --camera-ids "$CAMERA_IDS" \
        --project    "$PROJECT" \
        "${rigopt[@]}"
    for stage in extract_metadata detect_features match_features create_tracks "reconstruct --algorithm triangulation"; do
        echo "============ $stage"
        Q opensfm $stage "$PROJECT/opensfm"
    done
    [[ "$RIG" == 1 ]] && Q /usr/bin/python3 "$PHOTO/cluster/flatten-rig.py" "$PROJECT/opensfm"
    Q opendronemap \
        --project $PROJECT \
        --max-concurrency $NPROC \
        --rerun-from odm_meshing \
        --dsm \
        "$PROJECT"
fi
