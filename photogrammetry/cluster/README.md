# Running the QUAKES OpenSfM/ODM tools on a cluster (Apptainer)

The [packaging wiki page](https://github.com/dkogan/quakes/wiki/2026-05-27--packaging-opensfm-opendronemap)
ships a Debian/trixie rootfs tarball for `schroot`, which needs root. Here the same
tarball is deployed unchanged as an Apptainer image, `envs/quakes-jpl-trixie.sif`;
how, and the checks, are in [../../quakes-deploy/README.md](../../quakes-deploy/README.md).

`schroot -c quakes -- CMD` becomes `quakes-deploy/scripts/quakes-jpl.sh CMD`
(`apptainer exec --nv -B /scratch,/depot` plus the host GPU driver libraries; see
quakes-deploy/README.md for why plain `--nv` is not enough with this image).

All paths below are relative to the project root.

## Whitney example

```sh
Q=quakes-deploy/scripts/quakes-jpl.sh

# 1. .raw + INS -> .jpg + filename_latlonalt_opkdeg--cam*.vnl  ->  runs/whitney/raw-converted
#    Runs in the raw-conversion image envs/quakes-jpl-raw.sif, not in the JPL image itself
#    (see below). data/whitney-batch1 holds JPL's own conversion.
QUAKES_SIF=envs/quakes-jpl-raw.sif FIRST=25480 LAST=25535 $Q photogrammetry/cluster/raw-to-images-whitney.sh

# 2. OpenSfM + ODM  ->  runs/whitney/photogrammetry   (FRESH=1 starts over)
bash photogrammetry/cluster/run-whitney.sh

# 3. overview figures  ->  runs/whitney/photogrammetry/exports
$Q photogrammetry/cluster/export-figures.py runs/whitney/photogrammetry
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

### raw-convert in the JPL image

`raw-convert.py` needs `pyexiv2` (it writes the EXIF/XMP tags) and `_raw_convert`, built
by `make -C photogrammetry` with mrbuild and the OpenCV headers. The JPL image has
neither pyexiv2 nor any build tool. `envs/quakes-jpl-raw.sif`
([quakes-deploy/defs/quakes-jpl-raw.def](../../quakes-deploy/defs/quakes-jpl-raw.def))
is the JPL image plus those, from the Debian snapshot of the tarball's date
(2026-05-27), so nothing of the base is upgraded:
`quakes-deploy/scripts/verify_raw_image.sh` checks that all 780 base packages are
unchanged (versions and `dpkg --verify`) and 99 were added. It is for raw conversion
only; OpenSfM and ODM run in the JPL image itself. Build `_raw_convert` once with
`apptainer exec --cleanenv envs/quakes-jpl-raw.sif make -C photogrammetry`.
Frames 25480-25481 of SN330071 converted this way are byte-identical to
`runs/whitney/raw-converted` (made earlier in the image built from public sources).
(`raw-convert.py --prototype-raw-converter` avoids `_raw_convert`, not `pyexiv2`.)
