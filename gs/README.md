# Gaussian splatting on QUAKES imagery (Whitney)

Train 3DGS and 2DGS on the same images and poses as the photogrammetry, then
turn each model into a DSM/DTM and compare with ODM. Paths are relative to the
project root; every script finds the root from its own location. Shared
settings (environment, inputs, outputs) are in `common.sh` and can be
overridden from the environment.

## Pipeline

Run after `photogrammetry/` has produced `runs/whitney/photogrammetry`.

| Step | Script | Output (in `runs/whitney/gs/`) |
|---|---|---|
| 1 | `bash gs/prepare_dataset.sh` | `dataset/`: COLMAP export of the OpenSfM poses + image links |
| 2 | `bash gs/train_3dgs.sh` | `3dgs/`: checkpoints, held-out renders, metrics (~22 min on an A100) |
| 3 | `bash gs/train_2dgs.sh` | `2dgs/` (~54 min) |
| 4 | `envs/gs/bin/python gs/render_dsm.py --result_dir runs/whitney/gs/3dgs --dump_points runs/whitney/gs/dsm-dtm-compare/3dgs_points.csv` | `3dgs/dsm_eval/`: DSM vs ODM, plus the point cloud |
|   | `envs/gs/bin/python gs/render_dsm.py --result_dir runs/whitney/gs/2dgs --model 2dgs --depth median --dump_points runs/whitney/gs/dsm-dtm-compare/2dgs_points.csv` | `2dgs/dsm_eval_2dgs_median/` |
| 5 | `bash gs/make_dsm_dtm.sh` | `dsm-dtm-compare/{odm,3dgs,2dgs}_{dsm,dtm}.tif`: one PDAL pipeline for all three |
| 6 | `envs/gs/bin/python gs/compare_dsm_dtm.py` | `dsm-dtm-compare/`: stats.md, maps, differences, profiles |
| – | `bash gs/gpu_capacity_test.sh` | `gpu-capacity/capacity.csv`: Gaussians vs peak VRAM |

Steps 2–4 and the capacity test need a GPU. Steps 1 and 5 use the photogrammetry
image (`envs/quakes-odm.sif`) for OpenSfM and PDAL.

## Environment (`envs/gs`)

Python 3.11, torch 2.4.1 + CUDA 12.4, gsplat 1.5.3. `requirements-lock.txt` is
the exact package list. To rebuild from scratch:

```sh
conda create -p envs/gs python=3.11
envs/gs/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
envs/gs/bin/pip install -r gs/requirements-lock.txt
```

Two things that were needed to make it work:

- **numpy < 2 and plyfile 1.0.x.** The pycolmap fork gsplat's examples use breaks on
  numpy 2 (`np.uint64(-1)`), and plyfile 1.1 requires numpy 2.
- **gsplat's CUDA kernels compile on first use** (about 2.5 min), cached in
  `~/.cache/torch_extensions`. `common.sh` sets `CUDA_HOME` and
  `TORCH_CUDA_ARCH_LIST` (8.0 for the A100); keep them the same between runs or
  the kernels are rebuilt. On an H100 (Gautschi) use `TORCH_CUDA_ARCH_LIST=9.0`.

`third_party/gsplat` is a git submodule pinned at v1.5.3, the installed version:
the training scripts run its `examples/simple_trainer*.py`. After cloning the
project: `git submodule update --init gs/third_party/gsplat`.
