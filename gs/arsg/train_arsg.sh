#!/bin/bash
# Train on the Mammoth ARSGaussian block with gs/arsg/train_arsg.py (gsplat's simple_trainer + lidar).
#   MODE=default : plain 3DGS (the baseline)       MODE=arsg : lidar densification + GeoLoss
# Ablations add flags, e.g.  MODE=default ... --geo-loss    or    MODE=arsg ... --no-geo-loss
# Usage: OUT=runs/mammoth/arsg/3dgs MODE=default bash gs/arsg/train_arsg.sh [extra trainer flags]
set -euo pipefail
. "$(dirname "$0")/../common.sh"
# common.sh defaults DATASET to the Whitney dataset, so this block has its own variable
DATASET=${ARSG_DATASET:-$ROOT/runs/mammoth/arsg/dataset}
OUT=${OUT:?set OUT}
MODE=${MODE:-arsg}
FACTOR=${FACTOR:-4}
TEST_EVERY=${TEST_EVERY:-9}          # 8 cameras per instant: 8 would hold out one whole camera
OUT=$(realpath -m "$OUT"); DATASET=$(realpath -m "$DATASET")
mkdir -p "$OUT"

# densification re-allocates params + Adam state as ever larger tensors: without this, ~14 GB of the
# 40 GB A100 ends up reserved but unusable (fragmented) and the GeoLoss runs ran out of memory at ~18 M GSs
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$GSPLAT/examples"
"$GS_PY" "$ROOT/gs/arsg/train_arsg.py" "$MODE" \
    --data_dir "$DATASET" --data_factor "$FACTOR" --result_dir "$OUT" \
    --test_every "$TEST_EVERY" --max_steps 30000 \
    --eval_steps 7000 30000 --save_steps 7000 30000 \
    --save_ply --ply_steps 30000 \
    --disable_viewer \
    "$@" 2>&1 | tee "$OUT/train.log"
