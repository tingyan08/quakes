"""Turn a trained 3DGS/2DGS scene into a DSM and compare it with the ODM DSM (Whitney).

Renders depth from every training view, back-projects it to UTM, and grids it on
the (decimated) ODM DSM grid. Optionally dumps the points for make_dsm_dtm.sh.

For scenes exported from an ODM project's OpenSfM
reconstruction (opensfm export_colmap). ODM rewrites opensfm/reconstruction.json
into UTM coordinates minus the offset in odm_georeferencing/coords.txt (the
original ENU version is kept in reconstruction.topocentric.json). So the COLMAP
world frame is offset UTM, and the map to UTM is exact: no similarity fit needed.
Nothing here feeds back into training.

Usage (with the gs environment's python):
  python gs/render_dsm.py --result_dir runs/whitney/gs/3dgs
  python gs/render_dsm.py --result_dir runs/whitney/gs/2dgs --model 2dgs --depth median \
      --dump_points runs/whitney/gs/dsm-dtm-compare/2dgs_points.csv
"""
import argparse
import json
import os
import re
import sys

# Same settings as gs/common.sh, so gsplat finds its cached CUDA kernels (and
# ninja, if it has to rebuild them) when this is run directly
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("CUDA_HOME", os.environ.get("RCAC_CUDA_ROOT", "/usr/local/cuda"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
from gsplat.rendering import rasterization, rasterization_2dgs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # project root
sys.path.insert(0, f"{ROOT}/gs/third_party/gsplat/examples")
from datasets.colmap import Parser  # noqa: E402

SURFACE, TEXT, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQUENTIAL = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
              '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
# diverging blue <-> gray <-> red, from the same reference palette
DIVERGING = ['#1c5cab', '#3987e5', '#86b6ef', '#f0efec', '#f0a39a', '#e34948', '#a8302f']


def hillshade(z, res, az=315, alt=45):
    gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmedian(z)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360 - az + 90), np.radians(alt)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs[np.isnan(z)] = np.nan
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--data_dir", default=f"{ROOT}/runs/whitney/gs/dataset",
                    help="the COLMAP-format dataset the model was trained on")
    ap.add_argument("--photogrammetry", default=f"{ROOT}/runs/whitney/photogrammetry",
                    help="the ODM project: reference DSM and the UTM offset")
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--grid_factor", type=int, default=4,
                    help="decimation of the ODM DSM grid used as the comparison grid")
    ap.add_argument("--min_alpha", type=float, default=0.9)
    ap.add_argument("--model", choices=("3dgs", "2dgs"), default="3dgs",
                    help="which rasterizer the checkpoint was trained with")
    ap.add_argument("--depth", choices=("expected", "median"), default="expected",
                    help="rendered depth to back-project. median is 2dgs-only")
    ap.add_argument("--dump_points", default=None,
                    help="also write the back-projected points (UTM X,Y,Z) to this text file, for PDAL")
    ap.add_argument("--dump_stride", type=int, default=4,
                    help="keep every Nth pixel in u and v when dumping points")
    args = ap.parse_args()
    if args.depth == "median" and args.model != "2dgs":
        ap.error("--depth median needs --model 2dgs")
    label = args.model.upper()
    # 3dgs/expected keeps the original output directory name
    out_dir = os.path.join(args.result_dir,
                           "dsm_eval" if (args.model, args.depth) == ("3dgs", "expected")
                           else f"dsm_eval_{args.model}_{args.depth}")
    os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"

    COORDS = f"{args.photogrammetry}/odm_georeferencing/coords.txt"
    ODM_DSM = f"{args.photogrammetry}/odm_dem/dsm.tif"
    parser = Parser(args.data_dir, factor=args.data_factor, normalize=True, test_every=8)
    T_inv = np.linalg.inv(parser.transform)        # normalized -> OpenSfM ENU
    with open(COORDS) as f:
        f.readline()                                # "WGS84 UTM 11N"
        offset = np.array([float(x) for x in f.readline().split()] + [0.0])

    def norm_to_utm(p):
        return p @ T_inv[:3, :3].T + T_inv[:3, 3] + offset

    # ---- comparison grid: decimated ODM DSM ----
    with rasterio.open(ODM_DSM) as d:
        H, W = d.height // args.grid_factor, d.width // args.grid_factor
        ref = d.read(1, out_shape=(H, W), resampling=Resampling.average).astype(np.float64)
        nod = d.nodata
        T = d.transform * d.transform.scale(d.width / W, d.height / H)
        prof = d.profile
    ref[ref == nod] = np.nan
    inv = ~T

    def to_grid(pu, acc, cnt):
        col, row = inv * (pu[:, 0], pu[:, 1])
        col, row = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        idx = row[ok] * W + col[ok]
        np.add.at(acc, idx, pu[ok, 2])
        np.add.at(cnt, idx, 1)

    # ---- sanity check of the frame chain: SfM points vs ODM DSM ----
    pu = norm_to_utm(parser.points)
    col, row = inv * (pu[:, 0], pu[:, 1])
    col, row = np.floor(col).astype(int), np.floor(row).astype(int)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    dz_sfm = pu[ok, 2] - ref[row[ok], col[ok]]
    dz_sfm = dz_sfm[np.isfinite(dz_sfm)]
    print(f"frame check: SfM points - ODM DSM: median {np.median(dz_sfm):.2f} m "
          f"(n={len(dz_sfm)})")

    # ---- splats ----
    ckpt = args.ckpt or sorted(
        (os.path.join(args.result_dir, "ckpts", f)
         for f in os.listdir(os.path.join(args.result_dir, "ckpts"))),
        key=lambda p: int(re.search(r"ckpt_(\d+)", p).group(1)))[-1]
    sp = torch.load(ckpt, map_location=dev)["splats"]
    means, quats = sp["means"], sp["quats"]
    scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    sh_degree = int(np.sqrt(colors.shape[1]) - 1)
    print(f"ckpt {ckpt}: {len(means):,} Gaussians")

    # ---- render depth from every training view, back-project ----
    acc, cnt = np.zeros(H * W), np.zeros(H * W)
    fdump = None
    if args.dump_points:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_points)), exist_ok=True)
        fdump = open(args.dump_points, "w")
        fdump.write("X,Y,Z\n")
    for i in range(len(parser.image_names)):
        cid = parser.camera_ids[i]
        K = parser.Ks_dict[cid].copy()
        w, h = parser.imsize_dict[cid]
        c2w = torch.tensor(parser.camtoworlds[i], dtype=torch.float32, device=dev)
        Kt = torch.tensor(K, dtype=torch.float32, device=dev)
        viewmat = torch.linalg.inv(c2w)[None]
        with torch.no_grad():
            if args.model == "3dgs":
                out, alpha, _ = rasterization(
                    means, quats, scales, opac, colors, viewmat, Kt[None],
                    w, h, sh_degree=sh_degree, render_mode="RGB+ED", packed=False)
                depth = out[0, ..., 3]
            else:
                out, alpha, _, _, _, median, _ = rasterization_2dgs(
                    means, quats, scales, opac, colors, viewmat, Kt[None],
                    w, h, sh_degree=sh_degree, render_mode="RGB+ED", packed=False)
                depth = out[0, ..., 3] if args.depth == "expected" else median[0, ..., 0]
        depth = depth.cpu().numpy()
        a = alpha[0, ..., 0].cpu().numpy()
        v, u = np.nonzero(a > args.min_alpha)
        z = depth[v, u]
        rays = np.linalg.inv(K) @ np.stack([u + 0.5, v + 0.5, np.ones_like(u, dtype=float)])
        pc = (rays * z).T
        pw = (parser.camtoworlds[i, :3, :3] @ pc.T).T + parser.camtoworlds[i, :3, 3]
        pu = norm_to_utm(pw)
        to_grid(pu, acc, cnt)
        if fdump is not None:
            keep = (u % args.dump_stride == 0) & (v % args.dump_stride == 0)
            np.savetxt(fdump, pu[keep], fmt="%.2f", delimiter=",")
    if fdump is not None:
        fdump.close()
        print(f"Wrote points to {args.dump_points}")

    gs = np.full(H * W, np.nan)
    gs[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    gs = gs.reshape(H, W)
    prof.update(dtype="float32", nodata=np.nan, width=W, height=H, transform=T)
    with rasterio.open(os.path.join(out_dir, "gs_dsm.tif"), "w", **prof) as o:
        o.write(gs.astype(np.float32), 1)

    # ---- metrics (no bias removal needed, but reported both ways) ----
    dz = gs - ref
    m = np.isfinite(dz)
    d = dz[m]
    med = float(np.median(d))
    nmad = float(1.4826 * np.median(np.abs(d - med)))
    res = float(abs(T.a))
    stats = dict(model=args.model, depth=args.depth, frame_check_sfm_minus_odm_median_m=float(np.median(dz_sfm)),
                 coverage_frac_of_odm=float(m.sum() / np.isfinite(ref).sum()),
                 median_dz_m=med, nmad_m=nmad,
                 rmse_m=float(np.sqrt((d ** 2).mean())),
                 p95_abs_m=float(np.percentile(np.abs(d), 95)),
                 grid_res_m=res, n_gaussians=int(len(means)))
    print(json.dumps(stats, indent=1))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    # ---- figure: ODM | 3DGS | difference ----
    plt.rcParams.update({'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'text.color': TEXT, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                         'axes.titlelocation': 'left'})
    lo, hi = np.nanpercentile(ref, [1, 99])
    seq = matplotlib.colors.LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
    div = matplotlib.colors.LinearSegmentedColormap.from_list('div', DIVERGING)

    def shaded(z):
        c = seq(matplotlib.colors.Normalize(lo, hi)(np.clip(z, lo, hi)))
        hs = np.nan_to_num(hillshade(z, res), nan=1.0)
        c[..., :3] *= (0.55 + 0.45 * np.clip(hs, 0, 1))[..., None]
        c[..., 3] = np.isfinite(z)
        return c

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), constrained_layout=True)
    ax[0].imshow(shaded(ref)); ax[0].set_title("ODM DSM (OpenSfM + OpenMVS)")
    ax[1].imshow(shaded(gs)); ax[1].set_title(f"{label} DSM ({args.depth} depth, all views)")
    lim = max(3 * nmad, 5)
    im = ax[2].imshow(dz, cmap=div, vmin=-lim, vmax=lim)
    ax[2].set_title(f"{label} − ODM  ·  median {med:+.1f} m, NMAD {nmad:.1f} m")
    cb = fig.colorbar(im, ax=ax[2], shrink=0.8)
    cb.set_label("height difference (m)", color=TEXT2)
    cb.outline.set_visible(False)
    for a_ in ax:
        a_.set_axis_off()
    fig.savefig(os.path.join(out_dir, "dsm_compare.png"), dpi=130)

    # ---- figure: histogram + centre profile ----
    plt.rcParams.update({'axes.edgecolor': '#d6d5cf', 'axes.labelcolor': TEXT2,
                         'xtick.color': TEXT2, 'ytick.color': TEXT2})
    fig, ax = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    # clip at the 95th-percentile |dz| (rounded up to 50 m, at most 500 m) so
    # the bulk of the distribution is visible without piling into end bins
    clip = float(min(500, max(50, np.ceil(np.percentile(np.abs(d), 95) / 50) * 50)))
    ax[0].hist(np.clip(d, -clip, clip), bins=200, color='#2a78d6')
    ax[0].set_xlabel(f"{label} − ODM height (m), clipped to ±{clip:.0f}")
    ax[0].set_ylabel("grid cells")
    ax[0].set_title("Height difference distribution")
    row = H // 2
    xs = np.arange(W) * res / 1000
    ax[1].plot(xs, ref[row], color='#2a78d6', lw=2, label="ODM")
    ax[1].plot(xs, gs[row], color='#eb6834', lw=2, label=label)
    ax[1].set_xlabel("km along centre row (west → east)")
    ax[1].set_ylabel("elevation (m)")
    ax[1].legend(frameon=False)
    ax[1].set_title("Centre profile")
    for a_ in ax:
        a_.spines[['top', 'right']].set_visible(False)
        a_.grid(alpha=0.3, lw=0.5)
    fig.savefig(os.path.join(out_dir, "dz_hist_profile.png"), dpi=130)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
