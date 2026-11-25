"""Check every DSM/DTM against the ground control points (GCPs).

Samples each raster at the GCP positions (median of a WINDOW x WINDOW cell
neighbourhood, so one empty cell does not drop a GCP) and reports raster - GCP
height. Caveat: the GCPs constrain OpenSfM's bundle adjustment, so they are
control points, not independent checkpoints; ODM should match them closely, and
the GS models inherit their poses from that same adjustment.

Usage (gs environment's python):
  python gs/compare_gcps.py --gcps runs/whitney/photogrammetry/gcp_list.txt \
      --out runs/whitney-8cam/gs/gcp-check \
      "ODM native · 2 cams=runs/whitney/photogrammetry/odm_dem/dsm.tif" \
      "3DGS · 8 cams=runs/whitney-8cam/gs/dsm-dtm-compare/3dgs" ...
  Each argument is LABEL=PATH, PATH either a .tif or a <prefix> with <prefix>_dsm.tif
  and <prefix>_dtm.tif (from make_dsm_dtm.sh).
Writes OUT/gcp_check.json and OUT/gcp_check.md
"""
import argparse
import json
import os

import numpy as np
import rasterio
from rasterio.warp import transform


def read_gcps(path):
    """ODM gcp_list.txt (EPSG:4326 lon lat alt_m px py image) -> unique (lon, lat, alt) with obs counts."""
    rows = [l.split() for l in open(path).read().splitlines()[1:] if l.strip()]
    gcps = {}
    for r in rows:
        k = tuple(float(v) for v in r[:3])
        gcps[k] = gcps.get(k, 0) + 1
    return [dict(lon=k[0], lat=k[1], alt_m=k[2], n_obs=n) for k, n in gcps.items()]


def sample(tif, gcps, window):
    with rasterio.open(tif) as d:
        xs, ys = transform("EPSG:4326", d.crs, [g["lon"] for g in gcps], [g["lat"] for g in gcps])
        z = d.read(1).astype(float)
        if d.nodata is not None:
            z[z == d.nodata] = np.nan
        out = []
        for x, y in zip(xs, ys):
            r, c = d.index(x, y)
            h = window // 2
            if not (0 <= r < d.height and 0 <= c < d.width):
                out.append(None)
                continue
            v = z[max(r - h, 0):r + h + 1, max(c - h, 0):c + h + 1]
            v = v[np.isfinite(v)]
            out.append(float(np.median(v)) if v.size else None)
        return out, abs(d.res[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcps", required=True, help="ODM gcp_list.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=3, help="cells; odd")
    ap.add_argument("rasters", nargs="+", help="LABEL=PATH")
    args = ap.parse_args()
    gcps = read_gcps(args.gcps)
    os.makedirs(args.out, exist_ok=True)

    res = {"gcps": gcps, "window_cells": args.window, "rows": []}
    for spec in args.rasters:
        label, path = spec.split("=", 1)
        products = [("DSM", path)] if path.endswith(".tif") else [("DSM", f"{path}_dsm.tif"), ("DTM", f"{path}_dtm.tif")]
        for prod, tif in products:
            z, cell = sample(tif, gcps, args.window)
            dz = [None if v is None else v - g["alt_m"] for v, g in zip(z, gcps)]
            ok = [d for d in dz if d is not None]
            res["rows"].append(dict(label=label, product=prod, raster=os.path.relpath(tif), cell_m=cell, dz_m=dz,
                                    mean_m=float(np.mean(ok)) if ok else None,
                                    rmse_m=float(np.sqrt(np.mean(np.square(ok)))) if ok else None,
                                    max_abs_m=float(np.max(np.abs(ok))) if ok else None))
    json.dump(res, open(f"{args.out}/gcp_check.json", "w"), indent=1)

    f = lambda v: "—" if v is None else f"{v:+.1f}"
    hdr = [f"GCP {i + 1} ({g['alt_m']:.0f} m)" for i, g in enumerate(gcps)]
    md = ["| Method | Product | " + " | ".join(hdr) + " | Mean | RMSE |",
          "|---|---|" + "---|" * (len(hdr) + 2)]
    for r in res["rows"]:
        rmse = "—" if r["rmse_m"] is None else f"{r['rmse_m']:.1f}"
        md.append(f"| {r['label']} | {r['product']} | " + " | ".join(f(d) for d in r["dz_m"]) +
                  f" | {f(r['mean_m'])} | {rmse} |")
    open(f"{args.out}/gcp_check.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
