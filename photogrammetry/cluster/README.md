# Running the QUAKES OpenSfM/ODM tools on a cluster (Apptainer)

The [packaging wiki page](https://github.com/dkogan/quakes/wiki/2026-05-27--packaging-opensfm-opendronemap)
ships a Debian/trixie rootfs for `schroot`. That needs root, and the tarball is
only reachable from inside JPL. `quakes.def` rebuilds an equivalent toolset from
public sources as an Apptainer image, with no root needed:

| wiki image                    | here                                                                 |
|-------------------------------|----------------------------------------------------------------------|
| `opendronemap_3.6.0` .deb     | `opendronemap/odm:3.6.2-gpu` docker image (`/code`)                  |
| `python3-opensfm` (new fork)  | github.com/OpenSfM/OpenSfM, pinned commit, conda env in `/opt`       |
| `python3-mrcal` etc           | Ubuntu 24.04 packages for `/usr/bin/python3`; `vnlog.py` 1.40 (the trixie version) |

The `opensfm` and `opendronemap` commands are wrappers in `/usr/local/bin`, so
`../run-opensfm-opendronemap-handoff.sh` runs unmodified.

All paths below are relative to the project root.

## Build (once, ~30 min, inside a job with several cores)

```sh
export APPTAINER_CACHEDIR=$RCAC_SCRATCH/.apptainer/cache APPTAINER_TMPDIR=$RCAC_SCRATCH/.apptainer/tmp
apptainer build --fakeroot envs/quakes-odm.sif photogrammetry/cluster/quakes.def
apptainer exec -B /scratch envs/quakes-odm.sif make -C photogrammetry   # builds _raw_convert
```

`schroot -c quakes -- CMD` becomes `apptainer exec --nv -B /scratch,/depot envs/quakes-odm.sif CMD`.

## Whitney example, from the .raw frames

```sh
S=envs/quakes-odm.sif

# 1. .raw + INS -> .jpg + filename_latlonalt_opkdeg--cam*.vnl  ->  runs/whitney/raw-converted
#    (CAMS, FIRST, LAST, OUT are optional)
FIRST=25480 LAST=25535 apptainer exec -B /scratch,/depot $S photogrammetry/cluster/raw-to-images-whitney.sh

# 2. OpenSfM + ODM  ->  runs/whitney/photogrammetry   (FRESH=1 starts over)
bash photogrammetry/cluster/run-whitney.sh

# 3. overview figures  ->  runs/whitney/photogrammetry/exports
apptainer exec -B /scratch,/depot $S photogrammetry/cluster/export-figures.py runs/whitney/photogrammetry
```

`run-whitney.sh` also accepts the JPL-converted images: `DATA=data/whitney-batch1 bash photogrammetry/cluster/run-whitney.sh`.

`raw-convert.py` imports `generic_functions` from the private `sfmviz` repo.
`../generic_functions.py` replaces it, with code recovered from git history: it
was deleted from `raw-convert.py` in 3d5cab7 when it moved to sfmviz. It reads
the `reference-images-ins` metadata format, as used by the 2022 flights. For
the 2021 flights, `ins-vnl-plane-to-metadata.py` converts the per-camera
`gps_data_v2/*.vnl` INS files into that format.

On the Whitney frames 25480-25535 (SN330071, SN330075), this reproduces the
original `2021-07-23--processed/batch1` products. The poses match to 1e-13 deg.
The images match except for 0.015% of the pixels, all in the bottom 7 rows.
