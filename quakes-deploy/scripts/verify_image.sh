#!/bin/bash
# Repeatable runtime smoke tests of the JPL QUAKES image (SIF or unpacked rootfs).
# Read-only: queries the OS, the package manifest, the entry points and their
# shared libraries, and the tools' own --help. Runs no reconstruction.
#   bash quakes-deploy/scripts/verify_image.sh [IMAGE] [MANIFEST_DIR]
# IMAGE defaults to envs/quakes-jpl-trixie.sif. With MANIFEST_DIR, the package
# list is written to MANIFEST_DIR/dpkg-packages.txt.
set -uo pipefail
IMAGE=${1:-envs/quakes-jpl-trixie.sif}
OUT=${2:-}
X() { apptainer exec --cleanenv "$IMAGE" "$@"; }
fail=0
step() { echo; echo "== $*"; }
check() { "$@"; local s=$?; echo "-- exit $s"; [ $s -eq 0 ] || fail=1; return $s; }

step "image"
echo "$IMAGE"; [ -f "$IMAGE" ] && sha256sum "$IMAGE"
step "OS"
check X cat /etc/os-release
step "entry points"
check X sh -c 'for c in opendronemap opensfm python3 zsh; do printf "%-13s %s\n" "$c" "$(command -v $c || echo MISSING)"; done'
step "JPL packages"
check X dpkg-query -W opendronemap python3-opensfm libxml2-16 python3-mrcal python3-numpysane python3-vnlog zsh
if [ -n "$OUT" ]; then
    X dpkg-query -W -f '${Package}\t${Version}\t${Architecture}\n' > "$OUT/dpkg-packages.txt"
    echo "wrote $OUT/dpkg-packages.txt ($(wc -l < "$OUT/dpkg-packages.txt") packages)"
fi
step "package files intact (dpkg --verify of the JPL packages; no output = no changes)"
check X dpkg --verify opendronemap python3-opensfm
step "python imports"
check X python3 -c 'import mrcal, numpysane, vnlog, opensfm; print("mrcal", mrcal.__version__ if hasattr(mrcal, "__version__") else "?"); print("opensfm", opensfm.__file__)'
step "missing shared libraries of the entry points' ELF files"
check X sh -c '
  for c in opendronemap opensfm; do
    p=$(command -v $c) || continue
    r=$(readlink -f "$p")
    case "$(head -c4 "$r" | od -An -c | tr -d " ")" in
      177ELF) ldd "$r" | grep "not found" && exit 1 ;;
      *) echo "$c -> $r (script: $(head -1 "$r"))" ;;
    esac
  done; exit 0'
step "opensfm --help"
check X sh -c 'opensfm --help 2>&1 | head -20'
step "opendronemap --help"
check X sh -c 'opendronemap --help 2>&1 | head -20'
step "OpenMVS links CUDA, and no library is missing"
check X sh -c 'ldd /opt/opendronemap/SuperBuild/install/bin/OpenMVS/DensifyPointCloud | grep -E "libcuda|not found"; ! ldd /opt/opendronemap/SuperBuild/install/bin/OpenMVS/DensifyPointCloud | grep -q "not found"'
step "GPU through quakes-jpl.sh (host driver bound over the image's), with the opendronemap wrapper's LD_LIBRARY_PATH"
check env QUAKES_SIF="$IMAGE" "$(dirname "$0")/quakes-jpl.sh" sh -c 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && LD_LIBRARY_PATH=/opt/opendronemap/SuperBuild/build/pdal/lib python3 -c "import ctypes, sys; r = ctypes.CDLL(\"libcuda.so.1\").cuInit(0); print(\"cuInit\", r); sys.exit(r)"'
step "GPU with plain --nv (expected to FAIL on Gilbreth: image driver libs 550 vs host driver; informational only)"
apptainer exec --cleanenv --nv "$IMAGE" nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; echo "-- exit $? (not counted)"
echo; [ $fail -eq 0 ] && echo "SMOKE TESTS PASSED" || echo "SMOKE TESTS FAILED (see exit codes above)"
exit $fail
