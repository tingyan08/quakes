#!/bin/bash
#SBATCH -J gs-train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=120G
#SBATCH --time=12:00:00
# One Gaussian-splatting model end to end, as a batch job:
#   train (train_3dgs.sh / train_2dgs.sh) -> render_dsm.py (DSM vs ODM + point cloud)
#   -> eval_renders.py at 1/4 and 1/2 resolution
#
# Submit from the project root (account/partition as your allocation needs):
#   MODEL=3dgs FACTOR=2 NAME=3dgs-f2 OUT=runs/whitney-8cam/gs/3dgs-f2 \
#   DATASET=runs/whitney-8cam/gs/dataset PHOTOGRAMMETRY=runs/whitney-8cam/photogrammetry \
#   CMP=runs/whitney-8cam/gs/dsm-dtm-compare \
#   sbatch -A <account> -p <partition> --export=ALL -o runs/logs/%x-%j.out gs/train_and_evaluate.sbatch
#
#   MODEL  3dgs | 2dgs        FACTOR  training resolution divisor (4 = 1280 x 960)
#   NAME   point-cloud name in CMP (<NAME>_points.csv, for make_dsm_dtm.sh)
set -euo pipefail
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}          # sbatch runs a copy of this file, so not $(dirname $0)
. "$ROOT/gs/common.sh"
MODEL=${MODEL:?MODEL=3dgs or 2dgs}
FACTOR=${FACTOR:-4}
NAME=${NAME:-$MODEL}
TEST_EVERY=${TEST_EVERY:-8}     # 9 for 8-camera data (8 would hold out one whole camera)
OUT=$(realpath -m "${OUT:?OUT=output directory}")
DATASET=$(realpath -m "$DATASET")
PHOTOGRAMMETRY=$(realpath -m "$PHOTOGRAMMETRY")
CMP=$(realpath -m "${CMP:-$RUNS/dsm-dtm-compare}")
export FACTOR OUT DATASET TEST_EVERY

echo "== $(date +%T) train $MODEL at 1/$FACTOR resolution on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# --disable_video exists only in the 3DGS trainer
bash "$ROOT/gs/train_$MODEL.sh" $([[ $MODEL == 3dgs ]] && echo --disable_video)

# 3DGS: expected depth; 2DGS: median depth (more robust to floaters).
# Keep every (16/FACTOR)th pixel, so every resolution gives the same point density.
depth=$([[ $MODEL == 2dgs ]] && echo median || echo expected)
echo "== $(date +%T) render DSM ($depth depth)"
"$GS_PY" "$ROOT/gs/render_dsm.py" --result_dir "$OUT" --model "$MODEL" --depth "$depth" \
    --data_factor "$FACTOR" --test_every "$TEST_EVERY" --data_dir "$DATASET" --photogrammetry "$PHOTOGRAMMETRY" \
    --dump_stride $((16 / FACTOR)) --dump_points "$CMP/${NAME}_points.csv"

for f in 4 2; do
    echo "== $(date +%T) score renders at 1/$f resolution"
    "$GS_PY" "$ROOT/gs/eval_renders.py" --result_dir "$OUT" --model "$MODEL" --eval_factor $f --test_every "$TEST_EVERY" --data_dir "$DATASET"
done
echo "== $(date +%T) done"
