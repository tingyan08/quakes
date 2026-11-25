#!/bin/zsh

read -r -d '' usage <<EOF || true
Usage: $0 [--rig] --metadata GLOBS_VNL --images IMAGES_DIR --camera-ids IDS --project PROJECT_DIR
EOF

# comma-separated list of long options. Trailing : means "has required
# argument".
longopts='rig,metadata:,images:,camera-ids:,project:'
Ntrailing_options_required=0

# The parsing code to use getopt comes from
#   /usr/share/doc/util-linux/examples/getopt-example.bash
# I'm not an expert, but this sample makes things work
TEMP=$(getopt -o '' --long $longopts -n "$0" -- "$@")
[ $? -ne 0 ] && {
    echo '' > /dev/stderr
    echo $usage > /dev/stderr
    exit 1;
}
eval set -- "$TEMP"
unset TEMP
# The arguments now appear in a nice canonical order, and they are terminated by
# --

while {true} {
	case "$1" in
	        '--metadata')
                        shift
                        metadata=$1
                        shift
                        continue
                ;;

	        '--images')
                        shift
                        images=$1
                        shift
                        continue
                ;;

	        '--camera-ids')
                        shift
                        camera_ids=$1
                        shift
                        continue
                ;;

	        '--project')
                        shift
                        project=$1
                        shift
                        continue
                ;;

	        '--rig')
                        shift
                        rigopt="--rig"
                        continue
                ;;

                '--')
                        shift
                        break
                ;;

                *)
                        echo 'Internal error in argument parsing' >&2
                        exit 1
                ;;
        esac
}

if (( $#* != Ntrailing_options_required )) {
       echo "Exactly $Ntrailing_options_required non-option arguments are required. Got $#* instead" > /dev/stderr
       echo '' > /dev/stderr
       echo $usage > /dev/stderr
       exit 1
}

if [[ -z "$metadata" ]] {
  echo "ERROR: --metadata is required" > /dev/stderr
  echo $usage > /dev/stderr
  exit 1
}
if [[ -z "$images" ]] {
  echo "ERROR: --images is required" > /dev/stderr
  echo $usage > /dev/stderr
  exit 1
}
if [[ -z "$camera_ids" ]] {
  echo "ERROR: --camera-ids is required" > /dev/stderr
  echo $usage > /dev/stderr
  exit 1
}
if [[ -z "$project" ]] {
  echo "ERROR: --project is required" > /dev/stderr
  echo $usage > /dev/stderr
  exit 1
}




set -e

${0:h:A}/preprocess-opensfm.py \
  --metadata   $metadata \
  --images     $images \
  --camera-ids "$camera_ids" \
  --project    $project \
  --gcps       $project/gcps.json \
  $rigopt

##### OpenSfM
stages=(
  extract_metadata
  detect_features
  match_features
  create_tracks
  "reconstruct --algorithm triangulation")
for stage ($stages) {
  echo "============ $stage"
  opensfm ${=stage} $project/opensfm
}

##### ODM handoff
opendronemap \
  --project $project \
  --max-concurrency ${ODM_MAX_CONCURRENCY:-64} \
  --rerun-from odm_meshing \
  --dsm \
  $project(:A)
