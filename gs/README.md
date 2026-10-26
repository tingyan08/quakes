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
| 5 | `bash gs/make_dsm_dtm.sh` | `dsm-dtm-compare/<method>_{dsm,dtm}.tif`: one PDAL pipeline for ODM and every `<method>_points.csv` (`METHODS="3dgs-f2"` limits the run) |
| 6 | `envs/gs/bin/python gs/compare_dsm_dtm.py` | `dsm-dtm-compare/`: stats.md, maps, differences, profiles |
| 7 | `envs/gs/bin/python gs/compare_gcps.py --gcps runs/whitney/photogrammetry/gcp_list.txt --out runs/whitney/gs/gcp-check "3DGS=runs/whitney/gs/dsm-dtm-compare/3dgs" ...` | `gcp-check/gcp_check.md`: each DSM/DTM minus the GCP heights (the GCPs are bundle-adjustment control, not independent checkpoints) |
| – | `envs/gs/bin/python gs/eval_renders.py --result_dir runs/whitney/gs/3dgs --eval_factor 4` | `3dgs/eval_renders_f4.json`: held-out PSNR/SSIM/LPIPS at a chosen resolution, so models trained at different resolutions compare on the same images |
| – | `bash gs/gpu_capacity_test.sh` | `gpu-capacity/capacity.csv`: Gaussians vs peak VRAM |

Training resolution: `FACTOR` (default 4, i.e. 1280 × 960). For example
`FACTOR=2 OUT=runs/whitney/gs/3dgs-f2 bash gs/train_3dgs.sh`; then render its DSM with
`--data_factor 2 --dump_stride 8`, which keeps the point density of the 1/4-resolution runs.

Steps 2–4, `eval_renders.py` and the capacity test need a GPU. Steps 1 and 5 use the photogrammetry
image (`envs/quakes-jpl-trixie.sif`, see `quakes-deploy/`) for OpenSfM and PDAL.

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
