#!/bin/bash
# Checks that the raw-conversion image (quakes-deploy/defs/quakes-jpl-raw.def)
# only adds to the JPL base image: every base package is still installed at the
# same version, dpkg --verify finds no changed file of any of them, and the
# raw-convert.py dependencies import. Read-only.
#   bash quakes-deploy/scripts/verify_raw_image.sh [RAW_SIF] [BASE_SIF] [OUT_DIR]
set -uo pipefail
RAW=${1:-envs/quakes-jpl-raw.sif}
BASE=${2:-envs/quakes-jpl-trixie.sif}
OUT=${3:-quakes-deploy/manifests}
X() { apptainer exec --cleanenv "$@"; }
fail=0
q='${Package}:${Architecture}\t${Version}\n'

X "$BASE" dpkg-query -W -f "$q" | sort > "$OUT/raw-image-base-packages.tmp"
X "$RAW"  dpkg-query -W -f "$q" | sort > "$OUT/raw-image-packages.txt"

echo "== base packages missing or at another version in $RAW"
changed=$(comm -23 "$OUT/raw-image-base-packages.tmp" "$OUT/raw-image-packages.txt")
if [ -n "$changed" ]; then echo "$changed"; fail=1; else echo "none"; fi

echo "== packages added"
comm -13 "$OUT/raw-image-base-packages.tmp" "$OUT/raw-image-packages.txt" | tee "$OUT/raw-image-added.txt" | wc -l

echo "== dpkg --verify of all base packages (no output = no file changed)"
cut -f1 "$OUT/raw-image-base-packages.tmp" | X "$RAW" xargs dpkg --verify
s=$?; echo "-- exit $s"; [ $s -eq 0 ] || fail=1
rm -f "$OUT/raw-image-base-packages.tmp"

echo "== held packages (expected none)"
X "$RAW" apt-mark showhold | head

echo "== raw-convert.py imports"
X "$RAW" python3 -c 'import pyexiv2, numpy, numpysane, mrcal, vnlog, cv2, pyproj; print("pyexiv2", pyexiv2.__version__)'
s=$?; echo "-- exit $s"; [ $s -eq 0 ] || fail=1

echo; [ $fail -eq 0 ] && echo "RAW IMAGE CHECKS PASSED" || echo "RAW IMAGE CHECKS FAILED"
exit $fail
