#!/usr/bin/python3

r'''Export overview figures from a finished run-opensfm-opendronemap-handoff.sh project

Writes PNGs into PROJECT/exports:

- orthophoto-cameras.png: the orthophoto, with the camera centers and the GCPs
- dsm-hillshade.png:      the DSM, colored by elevation over a hillshade
- raw-vs-converted.png:   one .raw Bayer frame next to its converted .jpg, if
                          --raw and --jpg are given

Run in the JPL QUAKES image (its python3 has GDAL and matplotlib):

  quakes-deploy/scripts/quakes-jpl.sh photogrammetry/cluster/export-figures.py runs/whitney/photogrammetry

'''

import sys
import argparse
import os
import json
import re

parser = argparse.ArgumentParser(description = __doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--raw', help='''A .raw frame for the raw-vs-converted figure''')
parser.add_argument('--jpg', help='''The .jpg converted from --raw''')
parser.add_argument('--width', type=int, default=2400,
                    help='''Width of the downsampled rasters, in pixels''')
parser.add_argument('project')
args = parser.parse_args()

import numpy as np
from osgeo import gdal, osr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

gdal.UseExceptions()

# Colors: reference palette (dataviz skill). Light mode
SURFACE    = '#fcfcfb'
TEXT       = '#0b0b0b'
TEXT2      = '#52514e'
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a']   # fixed order: cam 1, cam 2, GCPs
SEQUENTIAL  = ['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5',
               '#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': TEXT, 'axes.labelcolor': TEXT2, 'xtick.color': TEXT2, 'ytick.color': TEXT2,
    'axes.edgecolor': '#d6d5cf', 'axes.linewidth': 0.8, 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})

outdir = os.path.join(args.project, 'exports')
os.makedirs(outdir, exist_ok=True)


def read_downsampled(path, width, bands=None):
    d  = gdal.Open(path)
    h  = int(round(d.RasterYSize * width / d.RasterXSize))
    gt = d.GetGeoTransform()
    bands = bands or list(range(1, d.RasterCount+1))
    arr = np.stack([d.GetRasterBand(b).ReadAsArray(buf_xsize=width, buf_ysize=h) for b in bands], axis=-1)
    nodata = d.GetRasterBand(bands[0]).GetNoDataValue()
    # extent in map coords, for imshow: left,right,bottom,top
    extent = (gt[0], gt[0] + gt[1]*d.RasterXSize,
              gt[3] + gt[5]*d.RasterYSize, gt[3])
    pixel_size = gt[1] * d.RasterXSize / width
    return arr, nodata, extent, pixel_size, d.GetProjection()


def utm_from_lonlat(wkt, lonlat):
    src = osr.SpatialReference(); src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference(wkt=wkt)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    t = osr.CoordinateTransformation(src, dst)
    return np.array([t.TransformPoint(lon, lat)[:2] for lon,lat in lonlat])


def km_axes(ax, extent):
    # Map coords relative to the lower-left corner, in km: easier to read than raw UTM
    x0, y0 = extent[0], extent[2]
    ax.set_aspect('equal')
    # Ticks every whole km relative to the lower-left corner. Call this after
    # everything is plotted, so the axis limits are final
    for axis, lim, v0 in ((ax.xaxis, ax.get_xlim(), x0),
                          (ax.yaxis, ax.get_ylim(), y0)):
        k = np.arange(np.ceil((lim[0]-v0)/1000), np.floor((lim[1]-v0)/1000) + 1)
        axis.set_major_locator(matplotlib.ticker.FixedLocator(v0 + 1000*k))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{int(i)}" for i in k]))
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.tick_params(length=3, width=0.6)


def scalebar(ax, extent, length_m=1000):
    x = extent[0] + 0.04*(extent[1]-extent[0])
    y = extent[2] + 0.07*(extent[3]-extent[2])
    ax.plot([x, x+length_m], [y, y], color=TEXT, lw=2, solid_capstyle='butt')
    ax.text(x + length_m/2, y + 0.02*(extent[3]-extent[2]), f"{length_m/1000:g} km",
            ha='center', va='bottom', fontsize=9, color=TEXT)


######### Camera centers and GCPs
cams = dict()
shots_geojson = os.path.join(args.project, 'odm_report', 'shots.geojson')
with open(shots_geojson) as f:
    for feat in json.load(f)['features']:
        name = feat['properties'].get('filename', '')
        m = re.search(r'(SN[0-9]+)', name)
        sn = m.group(1) if m else 'camera'
        cams.setdefault(sn, []).append(feat['geometry']['coordinates'])

gcps = []
gcps_json = os.path.join(args.project, 'gcps.json')
if os.path.exists(gcps_json):
    with open(gcps_json) as f:
        for key in json.load(f).keys():
            lat, lon, alt_ft = (float(x) for x in key.split())
            gcps.append((lon, lat, alt_ft * 0.3048))


######### Orthophoto + cameras + GCPs
ortho, _, extent, px, wkt = read_downsampled(
    os.path.join(args.project, 'odm_orthophoto', 'odm_orthophoto.tif'), args.width)
rgb   = ortho[..., :3].astype(float) / 255.
alpha = ortho[..., 3] > 0 if ortho.shape[-1] > 3 else np.ones(rgb.shape[:2], bool)
rgba  = np.dstack([rgb, alpha.astype(float)])

fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
ax.imshow(rgba, extent=extent, interpolation='antialiased', interpolation_stage='rgba')
for i, (sn, pts) in enumerate(sorted(cams.items())):
    pts = np.array(pts)
    xy  = utm_from_lonlat(wkt, pts[:, :2])
    ax.scatter(xy[:,0], xy[:,1], s=22, color=CATEGORICAL[i % 2], edgecolor='white',
               linewidth=1.2, zorder=3, label=f"Camera {sn} centers ({len(pts)})")
if gcps:
    xy = utm_from_lonlat(wkt, [(g[0], g[1]) for g in gcps])
    ax.scatter(xy[:,0], xy[:,1], s=90, marker='^', color=CATEGORICAL[2], edgecolor='white',
               linewidth=1.5, zorder=4, label=f"Ground control points ({len(gcps)})")
    for (x, y), g in zip(xy, gcps):
        ax.annotate(f"GCP {g[2]:.0f} m", (x, y), xytext=(8, -4), textcoords='offset points',
                    fontsize=9, color=TEXT,
                    bbox=dict(boxstyle='round,pad=0.25', fc=SURFACE, ec='none', alpha=0.85))
km_axes(ax, extent)
scalebar(ax, extent)
ax.set_title(f"Orthophoto from raw QUAKES frames, 2021-06-21 Mt. Whitney flight  ·  {px:.1f} m/px shown (native {gdal.Open(os.path.join(args.project,'odm_orthophoto','odm_orthophoto.tif')).GetGeoTransform()[1]:.2f} m/px)")
leg = ax.legend(loc='upper right', frameon=True, fontsize=9, facecolor=SURFACE, edgecolor='none')
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'orthophoto-cameras.png'))
plt.close(fig)
print(f"Wrote {outdir}/orthophoto-cameras.png")


######### DSM, hillshaded
dsm, nodata, extent, px, wkt = read_downsampled(
    os.path.join(args.project, 'odm_dem', 'dsm.tif'), args.width, bands=[1])
dsm   = dsm[..., 0].astype(float)
valid = np.isfinite(dsm) & (dsm != nodata)
z     = np.where(valid, dsm, np.nan)

# The edges of a small block have spikes. Clip the color scale to the 1-99th
# percentiles so they don't wash out the terrain
lo, hi = np.nanpercentile(z, [1, 99])

# Hillshade: sun from the NW at 45 deg elevation
gy, gx = np.gradient(np.where(valid, z, np.nanmedian(z)), px)
slope  = np.pi/2 - np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)
az, alt = np.radians(315), np.radians(45)
shade = np.sin(alt)*np.sin(slope) + np.cos(alt)*np.cos(slope)*np.cos(az - aspect)
shade = np.clip(shade, 0, 1)

cmap   = mcolors.LinearSegmentedColormap.from_list('seq_blue', SEQUENTIAL)
norm   = mcolors.Normalize(lo, hi)
color  = cmap(norm(np.clip(z, lo, hi)))
# multiply-blend the hillshade in, softened so the color still reads
color[..., :3] *= (0.55 + 0.45*shade)[..., None]
color[..., 3]   = valid

fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
ax.imshow(color, extent=extent, interpolation='antialiased', interpolation_stage='rgba')
km_axes(ax, extent)
scalebar(ax, extent)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
cb.set_label('Elevation (m, WGS84 ellipsoid)', color=TEXT2)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)
ax.set_title(f"Digital surface model  ·  {np.nanmin(z):.0f}–{np.nanmax(z):.0f} m, "
             f"median {np.nanmedian(z):.0f} m  ·  color scale clipped to 1–99th percentile")
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'dsm-hillshade.png'))
plt.close(fig)
print(f"Wrote {outdir}/dsm-hillshade.png")


######### Raw Bayer frame vs converted jpg
if args.raw and args.jpg:
    W, H, header = 5120, 3840, 196
    raw = np.fromfile(args.raw, dtype=np.uint8)
    if raw.size != W*H + header:
        raise Exception(f"Unexpected .raw size {raw.size}; expected {W*H+header}")
    bayer = raw[header:].reshape(H, W)
    jpg   = plt.imread(args.jpg)

    # A 1:1 crop from the middle of the frame, so the Bayer mosaic is visible
    cy, cx, half = H//2, W//2, 160
    sl = (slice(cy-half, cy+half), slice(cx-half, cx+half))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150,
                             gridspec_kw=dict(width_ratios=[1.333, 1, 1]))
    axes[0].imshow(jpg[::8, ::8])
    axes[0].add_patch(plt.Rectangle((sl[1].start/8, sl[0].start/8), 2*half/8, 2*half/8,
                                    fill=False, ec=CATEGORICAL[1], lw=1.5))
    axes[0].set_title('Converted frame (full, downsampled 8×)')
    axes[1].imshow(bayer[sl], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[1].set_title('Raw Bayer data (1:1 crop)')
    axes[2].imshow(jpg[sl], interpolation='nearest')
    axes[2].set_title('After raw-convert.py (1:1 crop)')
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
    fig.suptitle(os.path.basename(args.raw), x=0.01, ha='left', color=TEXT2, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'raw-vs-converted.png'))
    plt.close(fig)
    print(f"Wrote {outdir}/raw-vs-converted.png")
