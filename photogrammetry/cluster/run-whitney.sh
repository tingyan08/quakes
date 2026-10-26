#!/bin/bash
#SBATCH --job-name=quakes-whitney
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00

# Runs the Mt. Whitney example from
#   https://github.com/dkogan/quakes/wiki/2026-05-27--packaging-opensfm-opendronemap
# inside Dima Kogan's JPL QUAKES image (envs/quakes-jpl-trixie.sif, see quakes-deploy/), through
# quakes-deploy/scripts/quakes-jpl.sh (GPU driver binds). Works inside an
# interactive job (bash photogrammetry/cluster/run-whitney.sh) or via sbatch (pass -A/-p yourself)
#
# Environment overrides:
#   SIF       the apptainer image. Default: envs/quakes-jpl-trixie.sif
#   DATA      directory with filename_latlonalt_opkdeg--cam*.vnl and images/.
#             Default: runs/whitney/raw-converted (from raw-to-images-whitney.sh);
#             data/whitney-batch1 (the JPL-converted images) also works
#   PROJECT   output project directory. Removed and recreated if FRESH=1.
#             Default: runs/whitney/photogrammetry
#   FIRST,LAST  the frame range to process
#   CAMERA_IDS  cameras to use. Default: 'SN330071 SN330075' (the wiki example);
#               all 8: 'SN330068 SN330069 SN330070 SN330071 SN330074 SN330075 SN330076 SN330077'

set -e

PHOTO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # photogrammetry/
ROOT=$(dirname "$PHOTO")                                 # project root
SIF=${SIF:-$ROOT/envs/quakes-jpl-trixie.sif}
DATA=${DATA:-$ROOT/runs/whitney/raw-converted}
PROJECT=${PROJECT:-$ROOT/runs/whitney/photogrammetry}
FIRST=${FIRST:-25480}
LAST=${LAST:-25535}
CAMERA_IDS=${CAMERA_IDS:-SN330071 SN330075}
NPROC=${SLURM_CPUS_PER_TASK:-$(nproc)}

[[ "$FRESH" == 1 ]] && rm -rf "$PROJECT"
mkdir -p "$PROJECT/small-images"

# A small subset of images: the frames that see the GCPs below
for f in "$DATA"/images/*.jpg; do
    b=$(basename "$f")
    frame=${b%%_*}
    sn=${b##*_}; sn=${sn%.jpg}
    if (( frame >= FIRST && frame <= LAST )) && [[ " $CAMERA_IDS " == *" $sn "* ]]; then
        ln -sf "$f" "$PROJECT/small-images/$b"
    fi
done
echo "Using $(ls "$PROJECT/small-images" | wc -l) images"

cat > "$PROJECT/gcps.json" <<'EOF'
{
  "36.64609 -118.25383 7396":
  [ ["25507.*75\\.jpg", 2989, 391],
    ["25517.*75\\.jpg", 2772, 1955],
    ["25527.*75\\.jpg", 2713, 3483],
    ["25490.*71\\.jpg", 2002, 620]
  ],
  "36.63275 -118.26595 9424":
  [ ["25516.*75\\.jpg", 747, 389],
    ["25520.*75\\.jpg", 687, 1072],
    ["25525.*75\\.jpg", 700, 1916],
    ["25502.*71\\.jpg", 4226, 215],
    ["25497.*71\\.jpg", 4196, 1065],
    ["25488.*71\\.jpg", 4022, 2543]
  ],
  "36.64871 -118.26404 8721":
  [ ["25527.*75\\.jpg", 3199, 2360],
    ["25519.*75\\.jpg", 3231, 1069],
    ["25515.*75\\.jpg", 3322, 408],
    ["25488.*71\\.jpg", 1572, 2290],
    ["25495.*71\\.jpg", 1644, 1187],
    ["25501.*71\\.jpg", 1699, 215]
  ]
}
EOF

cat > "$PROJECT/config.yaml" <<EOF
align_method: auto
align_orientation_prior: no_roll
bundle_use_gcp: true
bundle_use_gps: false
flann_algorithm: KDTREE
matcher_type: FLANN
matching_gps_distance: 0
matching_gps_neighbors: 0
matching_graph_rounds: 3
optimize_camera_parameters: true
processes: $NPROC
reconstruction_algorithm: triangulation
triangulation_type: ROBUST
undistorted_image_format: tif
undistorted_image_max_size: 5120
use_altitude_tag: true
EOF


APPTAINERENV_ODM_MAX_CONCURRENCY=$NPROC QUAKES_SIF="$SIF" "$ROOT/quakes-deploy/scripts/quakes-jpl.sh" \
  "$PHOTO/run-opensfm-opendronemap-handoff.sh" \
    --metadata   "$DATA/*.vnl" \
    --images     "$PROJECT/small-images" \
    --camera-ids "$CAMERA_IDS" \
    --project    "$PROJECT"
