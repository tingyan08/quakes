#!/bin/bash
# Turn the photogrammetry (ODM) project into a COLMAP-format dataset for gsplat.
#
#   PHOTOGRAMMETRY/opensfm  --opensfm export_colmap-->  DATASET/sparse/0/{cameras,images,points3D}.bin
#   PHOTOGRAMMETRY/small-images                     -->  DATASET/images, DATASET/images_4 (symlinks)
#
# Same poses and intrinsics as the photogrammetry. Note: ODM rewrites
# opensfm/reconstruction.json into UTM minus the offset in
# odm_georeferencing/coords.txt, so the dataset's world frame is offset UTM
# (render_dsm.py adds the offset back).
#
# gsplat's Parser wants images_<factor>/ to exist for data_factor > 1; when it
# holds .jpg files it writes its own downsampled images_<factor>_png/.
#
# Usage: bash gs/prepare_dataset.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"

apptainer exec -B /scratch,/depot "$SIF" opensfm export_colmap --binary "$PHOTOGRAMMETRY/opensfm"

mkdir -p "$DATASET/sparse/0" "$DATASET/images"
cp "$PHOTOGRAMMETRY"/opensfm/colmap_export/{cameras,images,points3D}.bin "$DATASET/sparse/0/"

# Relative links to the actual image files, so the project can be moved
for f in "$PHOTOGRAMMETRY"/small-images/*.jpg; do
    ln -sfn "$(realpath --relative-to="$DATASET/images" "$(readlink -f "$f")")" "$DATASET/images/$(basename "$f")"
done
for factor in 2 4; do
    mkdir -p "$DATASET/images_$factor"
    for f in "$DATASET"/images/*.jpg; do
        ln -sfn "../images/$(basename "$f")" "$DATASET/images_$factor/$(basename "$f")"
    done
done
echo "dataset: $DATASET ($(ls "$DATASET/images" | wc -l) images)"
