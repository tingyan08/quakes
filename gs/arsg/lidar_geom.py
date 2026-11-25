"""Training lidar for ARSGaussian: points, normals, and lookups in both frames.

Two frames:
  metric : ODM's frame (UTM minus the offset, ellipsoid heights), z up; what the lidar and the
           densification thresholds are in.
  world  : gsplat's normalised frame, world = T @ metric (parser.transform: rotation, translation,
           uniform scale), what the Gaussians and cameras are in.

prepare() (run once, in the JPL image, no GPU): train_1m.npy -> train_1m_normals.npz with a PCA
normal per point (k = 16 neighbours, oriented up).
  QUAKES_NO_GPU=1 quakes-deploy/scripts/quakes-jpl.sh /usr/bin/python3 gs/arsg/lidar_geom.py
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
from lidar_prep import OUT as LIDAR, X0, XM, Y0, Y1   # noqa: E402

N_GEOID = -24.66
NPZ = f"{LIDAR}/train_1m_normals.npz"


def prepare(offset):
    p = np.load(f"{LIDAR}/train_1m.npy") - [offset[0], offset[1], -N_GEOID]     # -> metric frame
    tree = cKDTree(p)
    _, idx = tree.query(p, k=16, workers=-1)
    q = p[idx] - p[idx].mean(axis=1, keepdims=True)                                 # [N, 16, 3]
    cov = np.einsum("nki,nkj->nij", q, q)
    w, v = np.linalg.eigh(cov)                                                       # ascending eigenvalues
    n = v[:, :, 0]
    n *= np.sign(n[:, 2:3] + 1e-12)                                                  # up
    np.savez(NPZ, points=p.astype(np.float64), normals=n.astype(np.float32), offset=np.array(offset),
             bounds=np.array([X0 - offset[0], XM - offset[0], Y0 - offset[1], Y1 - offset[1]]))
    print(f"wrote {NPZ}: {len(p):,} points")


class LidarGeom:
    """Lidar for the trainer; `transform` is gsplat's parser.transform (metric -> world)."""

    def __init__(self, transform, device="cuda", cell=1.0):
        import torch
        d = np.load(NPZ)
        self.points_m, self.normals_m, self.bounds = d["points"], d["normals"], d["bounds"]
        T = np.asarray(transform, dtype=np.float64)
        self.scale = float(np.linalg.norm(T[:3, 0]))
        self.R = T[:3, :3] / self.scale                                              # pure rotation
        self.T, self.Tinv = T, np.linalg.inv(T)
        self.tree = cKDTree(self.points_m)
        # per 1 m column: lowest and highest return, for "a lidar point directly above"
        self.cell = cell
        ij = np.floor((self.points_m[:, :2] - [self.bounds[0], self.bounds[2]]) / cell).astype(np.int64)
        self.nx, self.ny = ij[:, 0].max() + 1, ij[:, 1].max() + 1
        flat = ij[:, 0] * self.ny + ij[:, 1]
        self.zmin = np.full(self.nx * self.ny, np.inf); np.minimum.at(self.zmin, flat, self.points_m[:, 2])
        self.zmax = np.full(self.nx * self.ny, -np.inf); np.maximum.at(self.zmax, flat, self.points_m[:, 2])
        # world-frame copies on the GPU for the depth / normal maps
        self.points_w = torch.tensor(self.points_m @ T[:3, :3].T + T[:3, 3], dtype=torch.float32, device=device)
        self.normals_w = torch.tensor(self.normals_m @ self.R.T, dtype=torch.float32, device=device)

    def to_metric(self, means_w):
        return means_w @ self.Tinv[:3, :3].T + self.Tinv[:3, 3]

    def inside(self, pm, margin=20.0):
        b = self.bounds
        return ((pm[:, 0] > b[0] + margin) & (pm[:, 0] < b[1] - margin)
                & (pm[:, 1] > b[2] + margin) & (pm[:, 1] < b[3] - margin))

    def column(self, pm):
        """(lowest, highest) lidar return in the 1 m column of each point; +-inf where empty."""
        ij = np.floor((pm[:, :2] - [self.bounds[0], self.bounds[2]]) / self.cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        lo = np.full(len(pm), np.inf); hi = np.full(len(pm), -np.inf)
        flat = ij[ok, 0] * self.ny + ij[ok, 1]
        lo[ok], hi[ok] = self.zmin[flat], self.zmax[flat]
        return lo, hi

    def depth_normal_maps(self, camtoworld, K, width, height):
        """Z-buffer of the lidar in one view: depth [H, W] (0 = none) and camera-frame normal [H, W, 3]."""
        import torch
        w2c = torch.linalg.inv(camtoworld)
        pc = self.points_w @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).floor().long()
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).floor().long()
        ok = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        idx = v[ok] * width + u[ok]
        zf = z[ok]
        depth = torch.full((height * width,), float("inf"), device=z.device)
        depth.scatter_reduce_(0, idx, zf, reduce="amin")
        win = zf <= depth[idx]                                                       # the nearest point of its pixel
        normal = torch.zeros((height * width, 3), device=z.device)
        n_c = self.normals_w[ok][win] @ w2c[:3, :3].T
        n_c = n_c * -torch.sign((n_c * pc[ok][win]).sum(-1, keepdim=True) + 1e-12)   # face the camera
        normal[idx[win]] = n_c
        depth[~torch.isfinite(depth)] = 0.0
        return depth.view(height, width), normal.view(height, width, 3)


if __name__ == "__main__":
    import json
    off = json.load(open(sys.argv[1]))["offset"] if len(sys.argv) > 1 else None
    if off is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        c = open(f"{root}/runs/mammoth/photogrammetry-rig-opt-fixed/odm_georeferencing/coords.txt").read().split("\n")[1]
        off = [float(x) for x in c.split()]
    prepare(off)
