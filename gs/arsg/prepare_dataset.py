"""COLMAP-format dataset for the ARSGaussian experiment, from the lidar-aligned reconstruction.

Keeps the shots that see the 4 x 4 km block (at least MIN_FRAC of a grid of block points, at the
lidar surface height, projects inside the image), writes them to a reduced reconstruction, runs
opensfm export_colmap on it and links the images, the layout gs/prepare_dataset.sh makes:
  OUT/sparse/0/{cameras,images,points3D}.bin, OUT/images/, OUT/images_{2,4}/ (symlinks)

  QUAKES_NO_GPU=1 quakes-deploy/scripts/quakes-jpl.sh /usr/bin/python3 gs/arsg/prepare_dataset.py \
      runs/mammoth/arsg/aligned runs/mammoth/arsg/dataset
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(__file__))
from lidar_prep import X0, X1, Y0, Y1   # noqa: E402

MIN_FRAC = 0.15
Z_SURFACE = 2550 - 24.66      # typical ellipsoid height of the block's terrain [m]


def main(aligned, out):
    tr = json.load(open(f"{aligned}/transform.json"))
    off = tr["offset"]
    src = f"{aligned}/opensfm"
    rec_all = json.load(open(f"{src}/reconstruction.json"))
    rec = rec_all[0]
    gx, gy = np.meshgrid(np.linspace(X0, X1, 9) - off[0], np.linspace(Y0, Y1, 9) - off[1])
    G = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, Z_SURFACE)]

    keep = {}
    for k, s in rec["shots"].items():
        cam = rec["cameras"][s["camera"]]
        Rw = Rot.from_rotvec(s["rotation"]).as_matrix()
        x = G @ Rw.T + s["translation"]
        u, v = x[:, 0] / x[:, 2], x[:, 1] / x[:, 2]            # normalised by max(w, h), centred
        w, h = cam["width"], cam["height"]
        f = cam["focal"]
        inside = (x[:, 2] > 0) & (np.abs(u * f) < 0.5 * w / max(w, h)) & (np.abs(v * f) < 0.5 * h / max(w, h))
        if inside.mean() >= MIN_FRAC:
            keep[k] = s
    print(f"{len(keep)} of {len(rec['shots'])} shots see >= {MIN_FRAC:.0%} of the block")
    rec["shots"] = keep

    # a project folder whose reconstruction holds only those shots; everything else links to the aligned one
    proj = f"{out}/opensfm"
    os.makedirs(proj, exist_ok=True)
    for f in os.listdir(src):
        if f not in ("reconstruction.json", "colmap_export") and not os.path.lexists(f"{proj}/{f}"):
            os.symlink(os.path.realpath(f"{src}/{f}"), f"{proj}/{f}")
    json.dump(rec_all, open(f"{proj}/reconstruction.json", "w"))
    subprocess.run(["opensfm", "export_colmap", "--binary", proj], check=True)

    os.makedirs(f"{out}/sparse/0", exist_ok=True)
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy(f"{proj}/colmap_export/{f}", f"{out}/sparse/0/{f}")
    os.makedirs(f"{out}/images", exist_ok=True)
    for k in keep:
        dst = f"{out}/images/{k}"
        if not os.path.lexists(dst):
            os.symlink(os.path.relpath(os.path.realpath(f"{proj}/images/{k}"), f"{out}/images"), dst)
    for factor in (2, 4):
        os.makedirs(f"{out}/images_{factor}", exist_ok=True)
        for k in keep:
            if not os.path.lexists(f"{out}/images_{factor}/{k}"):
                os.symlink(f"../images/{k}", f"{out}/images_{factor}/{k}")
    json.dump(dict(shots=sorted(keep), min_frac=MIN_FRAC, offset=off), open(f"{out}/selection.json", "w"), indent=1)
    print("dataset:", out)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
