#!/bin/zsh

# Converts the 2021-06 Mt. Whitney .raw frames into the inputs of
# run-opensfm-opendronemap-handoff.sh: debayered .jpg images with EXIF, and the
# per-camera filename_latlonalt_opkdeg--cam*.vnl pose files. This reproduces
# 2021-07-23--processed/batch1 (see the original process.sh~ in the dataset)
#
# Needs pyexiv2 and the _raw_convert C extension, which the JPL QUAKES image lacks;
# envs/quakes-jpl-raw.sif adds them (see photogrammetry/cluster/README.md). Build
# _raw_convert once (make -C photogrammetry, in that image), then:
#   QUAKES_SIF=envs/quakes-jpl-raw.sif quakes-deploy/scripts/quakes-jpl.sh photogrammetry/cluster/raw-to-images-whitney.sh
#
# Environment overrides:
#   CAMS        camera serial numbers to convert. Default: "330071 330075"
#   FIRST,LAST  frame range. Default: all frames
#   OUT         output directory. Default: runs/whitney/raw-converted
#   JOBS        parallel conversions. Default: $SLURM_CPUS_PER_TASK or nproc

set -e

PHOTO=${0:A:h:h}          # photogrammetry/
ROOT=${PHOTO:h}           # project root
W=${W:-/depot/andread/data/JPLdisk0/2021-06-21-23-quakes-mt-whitney}
RAW=${RAW:-/depot/andread/data/JPLdisk1/06-21-2021}
CAMS=(${=CAMS:-330071 330075})
OUT=${OUT:-$ROOT/runs/whitney/raw-converted}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}

# The camera mounting: from process.sh~ and Preproc_quakes.zsh
typeset -A mount
mount[330070]="--camera-pitch-deg $((90. + 11.3)) --camera-yaw-deg $((-(40.5/2.))) --camera-roll-deg 0"
mount[330075]="--camera-pitch-deg $((90. + 11.3)) --camera-yaw-deg $((-(17.1/2.))) --camera-roll-deg 0"
mount[330074]="--camera-pitch-deg $((90. + 11.3)) --camera-yaw-deg $(( (17.1/2.))) --camera-roll-deg 0"
mount[330068]="--camera-pitch-deg $((90. + 11.3)) --camera-yaw-deg $(( (40.5/2.))) --camera-roll-deg 0"
mount[330076]="--camera-pitch-deg $((90. - 11.3)) --camera-yaw-deg $((-(40.5/2.))) --camera-roll-deg 180"
mount[330071]="--camera-pitch-deg $((90. - 11.3)) --camera-yaw-deg $((-(17.1/2.))) --camera-roll-deg 180"
mount[330077]="--camera-pitch-deg $((90. - 11.3)) --camera-yaw-deg $(( (17.1/2.))) --camera-roll-deg 180"
mount[330069]="--camera-pitch-deg $((90. - 11.3)) --camera-yaw-deg $(( (40.5/2.))) --camera-roll-deg 180"

mkdir -p $OUT/images $OUT/metadata

for cam ($CAMS) {
    echo "=== camera $cam"

    # The batch4-raw/ symlinks point to JPL paths. The same frames live in $RAW;
    # I link them here, keeping the original names
    rawdir=$OUT/metadata/raw-cam$cam
    mkdir -p $rawdir
    for l ($W/batch4-raw/cam$cam/*.raw(N)) {
        ln -sf $RAW/$cam/${l:t} $rawdir/${l:t}
    }

    frameopts=()
    [[ -n "$FIRST" ]] && frameopts+=(--first-frame $FIRST)
    [[ -n "$LAST"  ]] && frameopts+=(--last-frame  $LAST)
    $PHOTO/cluster/ins-vnl-plane-to-metadata.py $frameopts \
        $W/gps_data_v2/batch4cam$cam.vnl $rawdir \
        > $OUT/metadata/images-referenced-cam$cam.vnl

    $PHOTO/raw-convert.py \
        -j $JOBS \
        --metadata                $OUT/metadata/images-referenced-cam$cam.vnl \
        --body-manufacturer       QUAKES \
        --body-model              EO-5120x3840 \
        --lens-model              'Schneider Xenon FF 100mm T2.1' \
        --body-serial-number      $cam \
        --lens-serial-number      $cam \
        --focal-length-nominal-mm 100 \
        --pixels-per-mm           $((1./6.4e-3)) \
        ${=mount[$cam]} \
        --camera-pitch-then-yaw \
        --model                   $W/cam68-dimacal-equalfxy.cameramodel \
        --image-dims              5120,3840 \
        --grayworld \
        --outdir                  $OUT/images \
        > $OUT/filename_latlonalt_opkdeg--cam$cam.vnl
}
