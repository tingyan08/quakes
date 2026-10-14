#!/bin/bash
# Train 2DGS (gsplat simple_trainer_2dgs) on the Whitney dataset with the 2DGS
# paper's regularizers: normal consistency (lambda 0.05, from step 7000) and
# depth distortion (lambda 0.01, from step 3000). Same resolution, steps and
# hold-out as train_3dgs.sh. Extra arguments go to the trainer.
#
# At the default weight the normal loss degraded this scene after step 7000
# (PSNR 29.3 -> 25.6 dB); to try a lower weight:
#   OUT=runs/whitney/gs/2dgs-normal0.005 bash gs/train_2dgs.sh --normal_lambda 0.005
set -euo pipefail
. "$(dirname "$0")/common.sh"
OUT=${OUT:-$RUNS/2dgs}
FACTOR=${FACTOR:-4}
# hold out every TEST_EVERY-th image (name order). With 8 cameras per frame use 9:
# 8 would hold out one whole camera
TEST_EVERY=${TEST_EVERY:-8}
# absolute paths: the trainer runs from gsplat's examples/ directory
OUT=$(realpath -m "$OUT"); DATASET=$(realpath -m "$DATASET")

cd "$GSPLAT/examples"
"$GS_PY" simple_trainer_2dgs.py \
    --data_dir "$DATASET" --data_factor "$FACTOR" --result_dir "$OUT" \
    --test_every "$TEST_EVERY" --max_steps 30000 \
    --eval_steps 7000 30000 --save_steps 7000 30000 \
    --normal_loss --dist_loss \
    --disable_viewer \
    "$@"
