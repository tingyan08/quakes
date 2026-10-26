#!/bin/bash
# Run a command in Dima Kogan's JPL QUAKES image (Debian trixie, prebuilt OpenSfM
# and OpenDroneMap): the Apptainer equivalent of the wiki's
#   schroot -c quakes -- CMD
# Usage, from the project root:
#   quakes-deploy/scripts/quakes-jpl.sh opendronemap --help
#   QUAKES_SIF=other.sif quakes-deploy/scripts/quakes-jpl.sh CMD ...
#
# GPU: the image ships Debian's NVIDIA user-space driver libraries (libcuda,
# libnvidia-ml, libnvidia-ptxjitcompiler; 550.163). They must match the host's
# kernel driver, which on Gilbreth they do not. --nv alone is not enough: the
# image's /lib/x86_64-linux-gnu comes first on the library path, and the
# opendronemap wrapper replaces LD_LIBRARY_PATH outright. So the host's copies
# are bound over the image's, at run time; the image itself is not changed.
# Set QUAKES_NO_GPU=1 to run without the GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SIF=${QUAKES_SIF:-$ROOT/envs/quakes-jpl-trixie.sif}
IMG_LIBDIR=/usr/lib/x86_64-linux-gnu/nvidia/current
args=(--cleanenv -B /scratch,/depot)

if [[ "${QUAKES_NO_GPU:-0}" != 1 ]] && [[ -r /proc/driver/nvidia/version ]]; then
    host_ver=$(sed -n 's/.*Kernel Module.*  \([0-9][0-9.]*\)  .*/\1/p' /proc/driver/nvidia/version | head -1)
    binds=()
    while read -r f; do                                   # e.g. libcuda.so.550.163.01
        base=${f%.so.*}.so
        host=$(ls /usr/lib64/$base.$host_ver 2>/dev/null || true)
        if [[ -n "$host" ]]; then
            binds+=("$host:$IMG_LIBDIR/$f")
        else
            echo "quakes-jpl: no host $base.$host_ver for the image's $f" >&2
        fi
    done < <(apptainer exec "$SIF" sh -c "ls $IMG_LIBDIR 2>/dev/null | grep -E '\.so\.[0-9]+\.[0-9]+\.[0-9]+$'")
    args+=(--nv)
    (( ${#binds[@]} )) && args+=(-B "$(IFS=,; echo "${binds[*]}")")
else
    args+=(--env ODM_NO_GPU=1)                            # ODM checks this before looking for nvidia-smi
fi
exec apptainer exec "${args[@]}" "$SIF" "$@"
