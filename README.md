# QUAKES: photogrammetry and Gaussian splatting

Fork of [dkogan/quakes](https://github.com/dkogan/quakes), reorganized to run on
Purdue's RCAC clusters (Gilbreth; Gautschi uses the same OS) and extended with
Gaussian splatting (3DGS, 2DGS) for DSM/DTM from QUAKES airborne imagery.

```
quakes/
  photogrammetry/   the upstream QUAKES tools (raw conversion, OpenSfM/ODM handoff, …)
    cluster/        run scripts for the cluster                             → cluster/README.md
  gs/               Gaussian splatting: dataset, training, DSM/DTM, evaluation → gs/README.md
    third_party/gsplat   (git submodule, v1.5.3)
  data/             input data (git-ignored), e.g. data/whitney-batch1
  runs/             all outputs (git-ignored), e.g. runs/whitney/{raw-converted,photogrammetry,gs}
  quakes-deploy/    Dima Kogan's JPL image deployed with Apptainer: scripts, report → quakes-deploy/README.md
  envs/             container images and Python environments (git-ignored), e.g. envs/quakes-jpl-trixie.sif
```

## Whitney example, end to end

Inside an interactive GPU job, from the project root:

```sh
# photogrammetry: images + poses -> OpenSfM + ODM (orthophoto, DSM), in the JPL image
# (the .raw -> images step is not runnable in that image yet: photogrammetry/cluster/README.md)
bash photogrammetry/cluster/run-whitney.sh

# Gaussian splatting on the same poses, then DSM/DTM against ODM
bash gs/prepare_dataset.sh
bash gs/train_3dgs.sh && bash gs/train_2dgs.sh
# ... render_dsm.py, make_dsm_dtm.sh, compare_dsm_dtm.py: see gs/README.md
```

Raw frames and INS are read from the QUAKES archive on `/depot/andread/data`.

## Changes from upstream

- Upstream files moved into `photogrammetry/` unchanged, except one line in
  `run-opensfm-opendronemap-handoff.sh` (ODM concurrency from `$ODM_MAX_CONCURRENCY`).
- `photogrammetry/generic_functions.py` recovered from git history (commit 3d5cab7)
  to replace the private `sfmviz` module that `raw-convert.py` imports.
- `photogrammetry/cluster/`: run scripts for the cluster.
- `quakes-deploy/`: the wiki's prebuilt schroot tarball, deployed unchanged as an Apptainer image.
- `gs/`: new.
