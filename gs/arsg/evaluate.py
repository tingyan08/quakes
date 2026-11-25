"""DSM / DTM of each method on the ARSGaussian block vs the lidar, per half.

Methods: every NAME=points.csv given (render_dsm.py --dump_points, dataset frame + ODM offset =
EPSG 6340 with ellipsoid heights after the ICP), plus "odm": ODM's dense cloud moved by the same ICP
transform. Each goes through one PDAL pipeline (gs/make_dsm_dtm.sh's: statistical outlier removal,
mean-Z DSM, SMRF with ODM's --dtm defaults, mean-Z DTM of ground points) on a 2 m grid over the block.

References, NAVD88 (method heights - N_GEOID): lidar DTM = USGS 1 m DEM; lidar DSM = highest point per
cell of train_1m / eval_1m; bare ground = cells where lidar DSM - DTM < 0.5 m.
West half = the lidar the model was trained with (not independent); east half = never seen in training.

  QUAKES_NO_GPU=1 quakes-deploy/scripts/quakes-jpl.sh /usr/bin/python3 gs/arsg/evaluate.py \
      3dgs=runs/mammoth/arsg/3dgs/points.csv arsg=runs/mammoth/arsg/arsg/points.csv ...
Writes runs/mammoth/arsg/eval/: <name>_{dsm,dtm}.tif, stats.json, stats.md, dtm_diff.png
"""
import json
import os
import subprocess
import sys

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(__file__))
from lidar_prep import ENV, OUT as LIDAR, PDAL, X0, X1, XM, Y0, Y1   # noqa: E402

gdal.UseExceptions()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT = f"{ROOT}/runs/mammoth/arsg/eval"
ALIGNED = f"{ROOT}/runs/mammoth/arsg/aligned"
ODM = f"{ROOT}/runs/mammoth/photogrammetry-rig-opt-fixed"
N_GEOID = -24.66
RES = 2.0
W, H = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
GCPS = {"CR06": (323798.538, 4168437.999, 2530.416), "CN": (323387.518, 4168377.069, 2539.055)}
SMRF = dict(scalar=1.25, slope=0.15, threshold=0.5, window=18.0)


def grid_writer(fn, window):
    return {"type": "writers.gdal", "filename": fn, "output_type": "mean", "resolution": RES,
            "origin_x": X0, "origin_y": Y0, "width": W, "height": H, "window_size": window,
            "nodata": -9999, "data_type": "float32", "gdaldriver": "GTiff", "override_srs": "EPSG:6340"}


def dsm_dtm(name, reader, extra=()):
    stages = [reader, *extra,
              {"type": "filters.crop", "bounds": f"([{X0 - 100},{X1 + 100}],[{Y0 - 100},{Y1 + 100}])"},   # block + margin
              {"type": "filters.assign", "value": "Classification = 0"},
              {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 3.0},
              {"type": "filters.range", "limits": "Classification![7:7]"},
              grid_writer(f"{OUT}/{name}_dsm.tif", 3),
              {"type": "filters.smrf", **SMRF, "cell": RES},
              {"type": "filters.range", "limits": "Classification[2:2]"},
              grid_writer(f"{OUT}/{name}_dtm.tif", 10)]
    p = f"{OUT}/{name}.pipeline.json"
    json.dump(stages, open(p, "w"), indent=1)
    subprocess.run([PDAL, "pipeline", p], check=True, env=ENV)


def read(fn):
    a = gdal.Open(fn).ReadAsArray().astype(float)
    a[a < -1e4] = np.nan
    return a - N_GEOID                       # ellipsoid -> NAVD88


def lidar_refs():
    dtm = gdal.Warp("", f"{ROOT}/data/lidar-mammoth/usgs-1m-dem-B22/B22_1m.vrt", format="MEM",
                    outputBounds=(X0, Y0, X1, Y1), xRes=RES, yRes=RES, dstSRS="EPSG:6340", resampleAlg="average",
                    dstNodata=np.nan, outputType=gdal.GDT_Float32).ReadAsArray().astype(float)
    dsm = np.full((H, W), -np.inf)
    for f in ("train_1m.npy", "eval_1m.npy"):
        p = np.load(f"{LIDAR}/{f}")                                       # EPSG 6340, NAVD88
        c = np.floor((p[:, 0] - X0) / RES).astype(int); r = np.floor((Y1 - p[:, 1]) / RES).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        np.maximum.at(dsm, (r[ok], c[ok]), p[ok, 2])
    dsm[~np.isfinite(dsm)] = np.nan
    return dsm, dtm


def stats(d):
    d = d[np.isfinite(d)]
    if not d.size:
        return dict(n=0)
    m = np.median(d)
    return dict(n=int(d.size), median=round(float(m), 2), nmad=round(float(1.4826 * np.median(np.abs(d - m))), 2),
                rmse=round(float(np.sqrt(np.mean(d ** 2))), 2), p95_abs=round(float(np.percentile(np.abs(d), 95)), 2))


def main(args):
    os.makedirs(OUT, exist_ok=True)
    methods = dict(a.split("=", 1) for a in args)
    # ODM's dense cloud, moved by the ICP transform (it acts on x - offset, y - offset)
    tr = json.load(open(f"{ALIGNED}/transform.json"))
    T, off = np.array(tr["transform"]), np.array(tr["offset"] + [0.0])
    M = np.eye(4); M[:3, :3] = T[:3, :3]; M[:3, 3] = T[:3, 3] + off - T[:3, :3] @ off
    if not os.path.exists(f"{OUT}/odm_dtm.tif"):
        dsm_dtm("odm", {"type": "readers.las", "filename": f"{ODM}/odm_georeferencing/odm_georeferenced_model.laz"},
                [{"type": "filters.transformation", "matrix": " ".join(f"{v:.10f}" for v in M.ravel())}])
    for name, csv in methods.items():
        dsm_dtm(name, {"type": "readers.text", "filename": os.path.abspath(csv)})

    ldsm, ldtm = lidar_refs()
    bare = (ldsm - ldtm) < 0.5
    half = {"west (train lidar)": np.s_[:, : (XM - X0) // int(RES)], "east (held out)": np.s_[:, (XM - X0) // int(RES):]}
    res = {}
    names = ["odm"] + list(methods)
    for name in names:
        dsm, dtm = read(f"{OUT}/{name}_dsm.tif"), read(f"{OUT}/{name}_dtm.tif")
        r = {}
        for h, s in half.items():
            r[h] = dict(coverage=round(float(np.isfinite(dsm[s]).mean()), 3),
                        dtm_vs_lidar_dtm_bare=stats(np.where(bare[s], dtm[s] - ldtm[s], np.nan)),
                        dsm_vs_lidar_dsm=stats(dsm[s] - ldsm[s]))
        g = {}
        for n, (x, y, z) in GCPS.items():
            c, rr = int((x - X0) / RES), int((Y1 - y) / RES)
            g[n] = dict(dsm=round(float(np.nanmedian(dsm[rr - 1:rr + 2, c - 1:c + 2]) - z), 2),
                        dtm=round(float(np.nanmedian(dtm[rr - 1:rr + 2, c - 1:c + 2]) - z), 2))
        r["gcps (method - GCP, NAVD88)"] = g
        res[name] = r
    json.dump(res, open(f"{OUT}/stats.json", "w"), indent=1)

    lines = ["| method | half | DTM−lidar DTM, bare: median / NMAD / RMSE (m) | DSM−lidar DSM: median / NMAD (m) | coverage |",
             "|---|---|---|---|---|"]
    for name in names:
        for h in half:
            a = res[name][h]; t = a["dtm_vs_lidar_dtm_bare"]; s = a["dsm_vs_lidar_dsm"]
            lines.append(f"| {name} | {h} | {t.get('median')} / {t.get('nmad')} / {t.get('rmse')} | "
                         f"{s.get('median')} / {s.get('nmad')} | {a['coverage']:.0%} |")
    lines += ["", "| method | CR06 DSM / DTM (m) | CN DSM / DTM (m) |", "|---|---|---|"]
    for name in names:
        g = res[name]["gcps (method - GCP, NAVD88)"]
        lines.append(f"| {name} | {g['CR06']['dsm']} / {g['CR06']['dtm']} | {g['CN']['dsm']} / {g['CN']['dtm']} |")
    open(f"{OUT}/stats.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.6), constrained_layout=True, squeeze=False)
    for a, name in zip(ax[0], names):
        im = a.imshow(read(f"{OUT}/{name}_dtm.tif") - ldtm, cmap="RdBu_r", vmin=-10, vmax=10, extent=(0, 4, 0, 4))
        a.axvline(2, color="k", lw=0.8); a.set_title(f"{name}: DTM − lidar DTM", fontsize=9)
        a.text(0.1, 3.7, "train", fontsize=8); a.text(2.1, 3.7, "held out", fontsize=8)
        a.set_xlabel("east [km]")
    fig.colorbar(im, ax=ax[0, -1], label="dz [m]", shrink=0.8)
    fig.savefig(f"{OUT}/dtm_diff.png", dpi=110)
    print(f"{OUT}/dtm_diff.png")


if __name__ == "__main__":
    main(sys.argv[1:])
