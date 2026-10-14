#!/bin/bash
# Train vanilla 3DGS (gsplat simple_trainer, original Kerbl et al. densification)
# on the Whitney dataset: 1/4 resolution (1280 x 960), 30k steps, every 8th image
# held out for evaluation. Extra arguments go to the trainer.
#
# Usage: bash gs/train_3dgs.sh [--max_steps 7000 ...]
#   OUT=runs/whitney/gs/3dgs-smoke bash gs/train_3dgs.sh --max_steps 7000 --eval_steps 7000 --save_steps 7000
set -euo pipefail
. "$(dirname "$0")/common.sh"
OUT=${OUT:-$RUNS/3dgs}
FACTOR=${FACTOR:-4}
# absolute paths: the trainer runs from gsplat's examples/ directory
OUT=$(realpath -m "$OUT"); DATASET=$(realpath -m "$DATASET")

cd "$GSPLAT/examples"
"$GS_PY" simple_trainer.py default \
    --data_dir "$DATASET" --data_factor "$FACTOR" --result_dir "$OUT" \
    --test_every 8 --max_steps 30000 \
    --eval_steps 7000 30000 --save_steps 7000 30000 \
    --save_ply --ply_steps 30000 \
    --disable_viewer \
    "$@"
