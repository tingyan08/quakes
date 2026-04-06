#!/bin/zsh

set -x
set -e

D=$1
shift

if [[ -z "$D" ]] {
  echo "Usage: $0 project_directory" > /dev/stderr
  exit 1
}

${0:h:A}/preprocess-opensfm.py \
  --metadata   '/mnt/aargh/proj/uavsar/2021-06-21-23-quakes-mt-whitney/2021-07-23--processed/batch1/*.vnl' \
  --images     '/mnt/diskA/2026-03--quakes/whitney' \
  --camera-ids 'SN330071 SN330075' \
  --project    $D \
  --gcps       $D/gcps.json

##### OpenSfM
stages=(
  extract_metadata
  detect_features
  match_features
  create_tracks
  "reconstruct --algorithm triangulation")
for stage ($stages) {
  
  ( cd ~/quakes/OpenSfM/;
    echo "============ $stage"
    bin/opensfm ${=stage} $D/opensfm
  )
}

##### ODM handoff

ODM=~/quakes/ODM;

( cd $ODM;
  LD_LIBRARY_PATH=$ODM/SuperBuild/install/lib:$ODM/SuperBuild/build/pdal/lib \
  PYTHONPATH="$ODM/../PyODM:$HOME/debianstuff/pygltflib:$HOME/debianstuff/euclidean-distance-transform-3d/build/lib.linux-x86_64-cpython-313:$HOME/debianstuff/CODEM/src" \
  python3 run.py \
    --project $D \
    --max-concurrency 64 \
    --rerun-from odm_meshing \
    --dsm \
    . )
