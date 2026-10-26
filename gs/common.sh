# Shared settings for the gs/ scripts. Source it: . "$(dirname "$0")/common.sh"
#
# Everything is relative to the project root, so the project can live anywhere.
# Any of these can be overridden from the environment.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# Python environment with torch + gsplat (see gs/README.md)
GS_ENV=${GS_ENV:-$ROOT/envs/gs}
GS_PY=${GS_PY:-$GS_ENV/bin/python}
# gsplat source, pinned at the installed version; its examples/ hold the trainers
GSPLAT=${GSPLAT:-$ROOT/gs/third_party/gsplat}
# The photogrammetry image: Dima Kogan's JPL QUAKES image (OpenSfM, ODM, PDAL), see quakes-deploy/
SIF=${SIF:-$ROOT/envs/quakes-jpl-trixie.sif}

# Inputs and outputs
PHOTOGRAMMETRY=${PHOTOGRAMMETRY:-$ROOT/runs/whitney/photogrammetry}   # ODM project
DATASET=${DATASET:-$ROOT/runs/whitney/gs/dataset}                     # COLMAP-format dataset
RUNS=${RUNS:-$ROOT/runs/whitney/gs}                                   # training/evaluation outputs

# gsplat compiles its CUDA kernels on first use (cached in ~/.cache/torch_extensions).
# Keep these the same between runs so the cached build is reused.
export CUDA_HOME=${CUDA_HOME:-${RCAC_CUDA_ROOT:-/usr/local/cuda}}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}     # A100; H100 is 9.0
export PATH=$GS_ENV/bin:$PATH
