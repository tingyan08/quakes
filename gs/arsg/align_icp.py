"""Align an ODM/OpenSfM reconstruction to the training lidar with a trimmed rigid ICP (replaces Colmap-PCD).

Frame: ODM's, i.e. WGS84 UTM 11N minus the offset in odm_georeferencing/coords.txt, ellipsoid heights.
The lidar (NAD83(2011) UTM 11N + NAVD88) enters the same frame with x - offset, y - offset,
z + N_GEOID; the < 1.5 m NAD83/WGS84 difference ends up in the fitted transform.

Moving: OpenSfM sparse points inside the training (west) half. Fixed: train.laz on a 1 m voxel grid.
The initial translation comes from a coarse grid search of SfM point heights against the lidar.
Output in OUT/: transform.json (4x4, SfM -> lidar frame), residuals before / after on the training half
and, as a check only, on the evaluation half (eval.laz, not used in the fit), and
opensfm/reconstruction.json with every shot and point moved by the transform (other files symlinked).

  QUAKES_NO_GPU=1 quakes-deploy/scripts/quakes-jpl.sh /usr/bin/python3 gs/arsg/align_icp.py \
      runs/mammoth/photogrammetry-rig-opt-fixed runs/mammoth/arsg/aligned
"""
import json
import os
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))
from lidar_prep import OUT as LIDAR, X0, X1, XM, Y0, Y1, PDAL, ENV   # noqa: E402

N_GEOID = -24.66          # ellipsoid - NAVD88 here (OPUS reports of CR06 and CN)


def lidar_xyz(laz, cell, offset, cache):
    """Lidar points thinned to one per `cell` m voxel, in the ODM frame."""
    if not os.path.exists(cache):
        txt = cache[:-4] + ".csv"
        subprocess.run([PDAL, "translate", laz, txt, "voxelcenternearestneighbor",
                        f"--filters.voxelcenternearestneighbor.cell={cell}",
                        "--writers.text.order=X,Y,Z", "--writers.text.keep_unspecified=false"], check=True, env=ENV)
        np.save(cache, np.loadtxt(txt, delimiter=",", skiprows=1))
        os.remove(txt)
    p = np.load(cache)
    return p - [offset[0], offset[1], -N_GEOID]


def rigid(a, b, w=None):
    """R, t minimising |R a + t - b| (Kabsch)."""
    w = np.ones(len(a)) if w is None else w
    ca, cb = (w[:, None] * a).sum(0) / w.sum(), (w[:, None] * b).sum(0) / w.sum()
    H = ((a - ca) * w[:, None]).T @ (b - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def stats(d):
    m = np.median(d)
    return dict(n=int(d.size), median=float(m), nmad=float(1.4826 * np.median(np.abs(d - m))))


def main(project, out):
    os.makedirs(out, exist_ok=True)
    off = [float(v) for v in open(f"{project}/odm_georeferencing/coords.txt").read().split("\n")[1].split()]
    rec_all = json.load(open(f"{project}/opensfm/reconstruction.json"))
    rec = rec_all[0]
    P = np.array([p["coordinates"] for p in rec["points"].values()])

    train = lidar_xyz(f"{LIDAR}/train.laz", 1.0, off, f"{LIDAR}/train_1m.npy")
    evalp = lidar_xyz(f"{LIDAR}/eval.laz", 1.0, off, f"{LIDAR}/eval_1m.npy")
    tree_t, tree_e = cKDTree(train), cKDTree(evalp)
    bx = lambda xa, xb: (xa - off[0], xb - off[0], Y0 - off[1], Y1 - off[1])

    # coarse start: horizontal grid search (20 m, then 4 m) of SfM heights vs the lidar surface (top return per 4 m cell)
    def in_box(Q, b, m=0):
        return (Q[:, 0] > b[0] + m) & (Q[:, 0] < b[1] - m) & (Q[:, 1] > b[2] + m) & (Q[:, 1] < b[3] - m)
    tb = bx(X0, XM)
    g = 4.0
    gi = np.floor((train[:, :2] - [tb[0], tb[2]]) / g).astype(int)
    top = {}
    for (i, j), z in zip(map(tuple, gi), train[:, 2]):
        if z > top.get((i, j), -1e9):
            top[(i, j)] = z

    def surf_dz(Q):
        ij = np.floor((Q[:, :2] - [tb[0], tb[2]]) / g).astype(int)
        z = np.array([top.get(tuple(k), np.nan) for k in ij])
        return Q[:, 2] - z
    best = None
    for step, rng, c in [(20, 400, (0, 0)), (4, 24, None)]:
        c = c or best[1:]
        for dx in np.arange(c[0] - rng, c[0] + rng + 1e-6, step):
            for dy in np.arange(c[1] - rng, c[1] + rng + 1e-6, step):
                Q = P + [dx, dy, 0]
                d = surf_dz(Q[in_box(Q, tb, 50)])
                d = d[np.isfinite(d)]
                if d.size < 200:
                    continue
                s = 1.4826 * np.median(np.abs(d - np.median(d)))
                if best is None or s < best[0]:
                    best = (s, dx, dy, -np.median(d))
        print(f"grid {step} m: shift E {best[1]:+.1f} N {best[2]:+.1f} U {best[3]:+.1f}, NMAD {best[0]:.2f} m", flush=True)
    R, t = np.eye(3), np.array([best[1], best[2], best[3]])

    # trimmed rigid ICP on the training half
    sel = in_box(P @ R.T + t, tb, 50)
    Ps = P[sel]
    for it in range(40):
        Q = Ps @ R.T + t
        d, k = tree_t.query(Q)
        thr = max(3 * np.median(d), 1.0)
        keep = d < thr
        dR, dt = rigid(Q[keep], train[k[keep]])
        R, t = dR @ R, dR @ t + dt
        ang = np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec()))
        if it % 5 == 0 or (ang < 1e-5 and np.linalg.norm(dt) < 1e-3):
            print(f"icp {it:2d}: median dist {np.median(d):.2f} m, kept {keep.mean():.0%}, step {np.linalg.norm(dt):.3f} m {ang:.5f}°", flush=True)
        if ang < 1e-5 and np.linalg.norm(dt) < 1e-3:
            break

    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    eb = bx(XM, X1)
    res = dict(transform=T.tolist(), rotation_deg=float(np.degrees(np.linalg.norm(Rot.from_matrix(R).as_rotvec()))),
               translation_m=t.tolist(), offset=off)
    for name, tree, box in [("train", tree_t, tb), ("eval", tree_e, eb)]:
        for when, Q in [("before", P), ("after", P @ R.T + t)]:
            q = Q[in_box(Q, box, 50)]
            d, _ = tree.query(q)
            res[f"{name}_{when}_nn_dist"] = stats(d)
    json.dump(res, open(f"{out}/transform.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "transform"}, indent=1))

    # moved reconstruction; every other OpenSfM file is a symlink to the original
    od = f"{out}/opensfm"
    os.makedirs(od, exist_ok=True)
    for f in os.listdir(f"{project}/opensfm"):
        if f in ("reconstruction.json", "colmap_export"):
            continue
        dst = f"{od}/{f}"
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(f"{project}/opensfm/{f}"), dst)
    for s in rec["shots"].values():
        Rs = Rot.from_rotvec(s["rotation"]).as_matrix()
        c = -Rs.T @ np.array(s["translation"])
        Rn = Rs @ R.T
        s["rotation"] = Rot.from_matrix(Rn).as_rotvec().tolist()
        s["translation"] = (-Rn @ (R @ c + t)).tolist()
    for p in rec["points"].values():
        p["coordinates"] = (R @ np.array(p["coordinates"]) + t).tolist()
    for key in ("rig_instances", "rig_cameras"):
        rec.pop(key, None)
    json.dump(rec_all, open(f"{od}/reconstruction.json", "w"))
    print("wrote", f"{od}/reconstruction.json")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
