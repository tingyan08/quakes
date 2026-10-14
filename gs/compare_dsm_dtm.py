"""Compare the DSMs and DTMs of ODM, 3DGS and 2DGS made by make_dsm_dtm.sh.

All statistics are over the cells valid in ALL six rasters, so every method is
scored on the same footprint. ODM is the reference here, not ground truth: it is
the conventional photogrammetry result from the same images and poses.

Writes into runs/whitney/gs/dsm-dtm-compare/ (or the directory given as argument):
  dsm_dtm_maps.png      DSM and DTM of each method (shared color scale per row)
  dsm_dtm_diff.png      3DGS - ODM and 2DGS - ODM, for DSM and DTM
  dsm_dtm_profile.png   centre-row profiles
  stats.json, stats.md  the numbers
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # project root
D = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/runs/whitney/gs/dsm-dtm-compare"
# ODM first, then 3DGS, 2DGS, then any other <name>_{dsm,dtm}.tif from make_dsm_dtm.sh
LABELS = {"odm": "ODM (OpenSfM + OpenMVS)", "3dgs": "3DGS", "2dgs": "2DGS",
          "3dgs-f2": "3DGS · 1/2 res", "2dgs-f2": "2DGS · 1/2 res",
          # models from another run (e.g. the 2-camera block) compared on this run's grid
          "2cam-3dgs": "3DGS · 2 cams", "2cam-2dgs": "2DGS · 2 cams",
          "2cam-3dgs-f2": "3DGS · 1/2 res · 2 cams", "2cam-2dgs-f2": "2DGS · 1/2 res · 2 cams"}
found = sorted(f[:-len("_dsm.tif")] for f in os.listdir(D)
               if f.endswith("_dsm.tif") and os.path.exists(f"{D}/{f[:-len('_dsm.tif')]}_dtm.tif"))
order = ["odm", "3dgs", "2dgs"]
METHODS = [(m, LABELS.get(m, m.upper())) for m in order if m in found] + \
          [(m, LABELS.get(m, m.upper())) for m in found if m not in order]

SURFACE, TEXT, TEXT2 = "#fcfcfb", "#0b0b0b", "#52514e"
# categorical palette in fixed order: a method keeps its colour whichever others are present
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES = {m: PALETTE[i % len(PALETTE)] for i, m in
          enumerate(["odm", "3dgs", "2dgs"] + [m for m, _ in METHODS if m not in ("odm", "3dgs", "2dgs")])}
SEQUENTIAL = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
              '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
DIVERGING = ['#1c5cab', '#3987e5', '#86b6ef', '#f0efec', '#f0a39a', '#e34948', '#a8302f']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': TEXT, 'axes.labelcolor': TEXT2, 'xtick.color': TEXT2, 'ytick.color': TEXT2,
    'axes.edgecolor': '#d6d5cf', 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})
seq = matplotlib.colors.LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
div = matplotlib.colors.LinearSegmentedColormap.from_list('div', DIVERGING)


def load(path):
    with rasterio.open(path) as d:
        z = d.read(1).astype(np.float64)
        res = abs(d.transform.a)
        nod = d.nodata
    z[(z == nod) | ~np.isfinite(z)] = np.nan
    return z, res


z = {}
for m, _ in METHODS:
    for kind in ("dsm", "dtm"):
        z[m, kind], res = load(f"{D}/{m}_{kind}.tif")

common = np.ones_like(z["odm", "dsm"], dtype=bool)
for v in z.values():
    common &= np.isfinite(v)
print(f"common footprint: {common.sum():,} cells ({common.sum() * res * res / 1e6:.1f} km²)")
for k in z:
    z[k] = np.where(common, z[k], np.nan)


def robust(d):
    d = d[np.isfinite(d)]
    med = float(np.median(d))
    return dict(median_m=med,
                nmad_m=float(1.4826 * np.median(np.abs(d - med))),
                rmse_m=float(np.sqrt((d ** 2).mean())),
                p95_abs_m=float(np.percentile(np.abs(d), 95)),
                within_10m_frac=float((np.abs(d) <= 10).mean()))


stats = dict(grid_res_m=res, common_cells=int(common.sum()),
             common_area_km2=float(common.sum() * res * res / 1e6), methods={})
for m, _ in METHODS:
    s = dict(ndsm_median_m=float(np.nanmedian(z[m, "dsm"] - z[m, "dtm"])))
    if m != "odm":
        s["dsm_vs_odm"] = robust(z[m, "dsm"] - z["odm", "dsm"])
        s["dtm_vs_odm"] = robust(z[m, "dtm"] - z["odm", "dtm"])
    stats["methods"][m] = s
with open(f"{D}/stats.json", "w") as f:
    json.dump(stats, f, indent=1)

# markdown table
rows = ["| | DSM median Δ | DSM NMAD | DSM within ±10 m | DTM median Δ | DTM NMAD | DTM within ±10 m | DSM − DTM (median) |",
        "|---|---|---|---|---|---|---|---|"]
for m, name in METHODS:
    s = stats["methods"][m]
    if m == "odm":
        rows.append(f"| {name} | reference | | | reference | | | {s['ndsm_median_m']:.1f} m |")
    else:
        a, b = s["dsm_vs_odm"], s["dtm_vs_odm"]
        rows.append(f"| {name} | {a['median_m']:+.1f} m | {a['nmad_m']:.1f} m | {100*a['within_10m_frac']:.0f}% "
                    f"| {b['median_m']:+.1f} m | {b['nmad_m']:.1f} m | {100*b['within_10m_frac']:.0f}% "
                    f"| {s['ndsm_median_m']:.1f} m |")
with open(f"{D}/stats.md", "w") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(rows))


def hillshade(a):
    gy, gx = np.gradient(np.nan_to_num(a, nan=np.nanmedian(a)), res)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    return np.clip(np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1)


def shaded(a, lo, hi):
    c = seq(matplotlib.colors.Normalize(lo, hi)(np.clip(a, lo, hi)))
    c[..., :3] *= (0.55 + 0.45 * hillshade(a))[..., None]
    c[..., 3] = np.isfinite(a)
    return c


# ---- maps: DSM row, DTM row; one color scale per row, set by ODM ----
fig, ax = plt.subplots(2, len(METHODS), figsize=(6 * len(METHODS), 7.6), constrained_layout=True, squeeze=False)
for r, kind in enumerate(("dsm", "dtm")):
    lo, hi = np.nanpercentile(z["odm", kind], [1, 99])
    for c, (m, name) in enumerate(METHODS):
        ax[r, c].imshow(shaded(z[m, kind], lo, hi), interpolation='antialiased', interpolation_stage='rgba')
        ax[r, c].set_title(f"{name} · {kind.upper()}")
        ax[r, c].set_axis_off()
    sm = plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(lo, hi), cmap=seq)
    cb = fig.colorbar(sm, ax=ax[r, :], fraction=0.015, pad=0.01)
    cb.set_label(f"{kind.upper()} elevation (m)", color=TEXT2)
    cb.outline.set_visible(False)
fig.suptitle(f"Whitney block, {stats['common_area_km2']:.1f} km² common footprint, {res:.1f} m grid · "
             "same PDAL pipeline for every method (outlier removal, mean Z; DTM = SMRF ground points)",
             x=0.01, ha='left', color=TEXT2, fontsize=10)
fig.savefig(f"{D}/dsm_dtm_maps.png", dpi=130)
plt.close(fig)

# ---- differences vs ODM ----
lim = 60
others = [m for m, _ in METHODS if m != "odm"]
fig, ax = plt.subplots(2, len(others), figsize=(6.5 * len(others), 7.6), constrained_layout=True, squeeze=False)
for r, kind in enumerate(("dsm", "dtm")):
    for c, m in enumerate(others):
        s = stats["methods"][m][f"{kind}_vs_odm"]
        im = ax[r, c].imshow(z[m, kind] - z["odm", kind], cmap=div, vmin=-lim, vmax=lim)
        ax[r, c].set_title(f"{LABELS.get(m, m.upper())} − ODM · {kind.upper()} · median {s['median_m']:+.1f} m, "
                           f"NMAD {s['nmad_m']:.1f} m")
        ax[r, c].set_axis_off()
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cb.set_label(f"height difference (m), color clipped at ±{lim}", color=TEXT2)
cb.outline.set_visible(False)
fig.savefig(f"{D}/dsm_dtm_diff.png", dpi=130)
plt.close(fig)

# ---- centre profiles ----
row = int(np.argmax(common.sum(axis=1)))      # the row with the most common cells
fig, ax = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True, constrained_layout=True)
xs = np.arange(common.shape[1]) * res / 1000
for r, kind in enumerate(("dsm", "dtm")):
    for m, name in METHODS:
        ax[r].plot(xs, z[m, kind][row], color=SERIES[m], lw=2 if m == "odm" else 1.5, label=name)
    ax[r].set_ylabel(f"{kind.upper()} elevation (m)")
    ax[r].set_title(f"{kind.upper()} along grid row {row} (west → east)")
    ax[r].grid(alpha=0.3, lw=0.5)
    ax[r].spines[['top', 'right']].set_visible(False)
ax[0].legend(frameon=False, loc='upper right')
ax[1].set_xlabel("km along row")
fig.savefig(f"{D}/dsm_dtm_profile.png", dpi=130)
plt.close(fig)
print(f"Wrote figures to {D}")
