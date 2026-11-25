"""Lidar for the ARSGaussian experiment: a 4 x 4 km block around the Mammoth GCPs, split in two.

  west half -> train.laz : noise removed, every 6th point kept (~8 pts/m², the paper's density);
                           used by the model (depth / normal maps, densification)
  east half -> eval.laz  : noise removed, full density (~46 pts/m²); never shown to the model,
                           only used to evaluate DSM / DTM

Source: USGS 3DEP CA_SierraNevada_13_B22 (EPT on AWS), reprojected to NAD83(2011) UTM 11N + NAVD88
(EPSG 6340 + 5703), as data/lidar-mammoth/ept-B22-gcp-test. Run in the JPL image (ODM's PDAL):
  QUAKES_NO_GPU=1 quakes-deploy/scripts/quakes-jpl.sh /usr/bin/python3 gs/arsg/lidar_prep.py [west|east ...]
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT = f"{ROOT}/data/lidar-mammoth/ept-B22-arsg-4km"
EPT = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/CA_SierraNevada_13_B22/ept.json"
PDAL = "/opt/opendronemap/SuperBuild/install/bin/pdal"
ENV = dict(os.environ, LD_LIBRARY_PATH="/opt/opendronemap/SuperBuild/build/pdal/lib")

# block: 4 km centred on the midpoint of GCPs CR06 and CN (EPSG 6340), split at its centre
X0, X1, Y0, Y1 = 321593, 325593, 4166408, 4170408
XM = (X0 + X1) // 2
HALVES = {"west": dict(x=(X0, XM), file="train.laz", decimate=6),
          "east": dict(x=(XM, X1), file="eval.laz", decimate=1)}


def pipeline(half):
    import pyproj   # only here: the trainer imports this module's constants in an env without pyproj
    h = HALVES[half]
    t = pyproj.Transformer.from_crs("EPSG:6340", "EPSG:3857", always_xy=True)
    xs, ys = t.transform([h["x"][0], h["x"][1]] * 2, [Y0, Y0, Y1, Y1])
    b3857 = f"([{min(xs) - 50},{max(xs) + 50}],[{min(ys) - 50},{max(ys) + 50}])"   # EPT is in EPSG 3857
    stages = [{"type": "readers.ept", "filename": EPT, "bounds": b3857},
              {"type": "filters.reprojection", "out_srs": "EPSG:6340+5703"},
              {"type": "filters.crop", "bounds": f"([{h['x'][0]},{h['x'][1]}],[{Y0},{Y1}])"},
              {"type": "filters.range", "limits": "Classification![7:7],Classification![18:18]"}]   # drop noise
    if h["decimate"] > 1:
        stages.append({"type": "filters.decimation", "step": h["decimate"]})
    stages.append({"type": "writers.las", "filename": f"{OUT}/{h['file']}", "compression": "laszip",
                   "a_srs": "EPSG:6340+5703", "extra_dims": "all"})
    return stages


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for half in sys.argv[1:] or list(HALVES):
        p = f"{OUT}/pipeline_{half}.json"
        json.dump(pipeline(half), open(p, "w"), indent=1)
        print(f"{half}: pdal pipeline {p}", flush=True)
        subprocess.run([PDAL, "pipeline", "--stream", p], check=True, env=ENV)
        print(f"{half}: wrote {HALVES[half]['file']}", flush=True)
