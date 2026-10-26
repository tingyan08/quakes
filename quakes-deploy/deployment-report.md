# Deployment report: JPL QUAKES image on Gilbreth (2026-10-26)

| Level | Status | Evidence |
|---|---|---|
| `ARCHIVE_AVAILABLE` | **Reached, with one caveat** | Archive at `quakes-opensfm-opendronemap-trixie.tar.gz` (project root), supplied by the user; fully readable; SHA-256 recorded. **Not authenticated:** no checksum from Dima to compare against |
| `IMAGE_BUILT` | **Reached** | `envs/quakes-jpl-trixie.sif` converted from the unpacked archive; no files added, removed or replaced |
| `RUNTIME_VERIFIED` | **Reached** | `verify_image.sh`: 10/10 checks pass, interactively (`gilbreth-n006`) and as batch job 12024441 (`gilbreth-n010`) |
| `WORKFLOW_VERIFIED` | **Not attempted** | Needs the `uavsar` workflow (or confirmation that `dkogan/quakes` matches it), the `whitney-small-images` list and Dima's reference outputs. `--help` passing is not workflow verification |

## Environment (`manifests/environment.txt`)

Gilbreth compute node `gilbreth-n006` (interactive job 12021489, `a100-40gb`, 8 cores),
Rocky Linux 9.8, x86_64, Apptainer 1.5.3, NVIDIA A100-SXM4-40GB with driver 595.91.07.
No `/etc/subuid` entry. Storage: scratch 200 TB quota (1.0 TB used), 2.0 M file limit
(573 K used after extraction); home 22.5 of 25 GB used, so all Apptainer temp/cache
goes to `$RCAC_SCRATCH/.apptainer`.

## Archive (`manifests/archive.sha256`, `manifests/archive-listing.txt`)

- `bd385bd386376f5eea950efbf02f0ce169c5f2d10e6c83dcd0403c19e58abd22`, 5,771,626,145 bytes, gzip; 3.5 GB tar (size mod 2^32) / 12.0 GB of file contents.
  This is a local fingerprint; it proves authenticity only once Dima publishes the same hash.
- `tar -tvzf` over the whole archive: exit 0, 123,632 entries: 107,746 files, 12,061 directories, 3,816 symlinks, 1 hard link, 8 character devices.
- Audit: no absolute names; no `..` components; no entry written through a symlinked directory;
  the one hard link (`./usr/bin/perl5.40.1` to `./usr/bin/perl`) is internal; device nodes only in `./dev`
  (console, full, null, ptmx, random, tty, urandom, zero); setuid bits on the usual Debian tools
  (`su`, `passwd`, `mount`, ...), irrelevant because Apptainer runs `nosuid`. Three matplotlib
  symlinks have `../` chains that resolve to `/share/javascript/...` inside the root (broken, harmless).
- Contents: Debian GNU/Linux 13.5 (trixie), 780 packages (`manifests/dpkg-packages.txt`). Entry dates: 2026-05-27 (matches the wiki page date).

## Packages the wiki names (`dpkg-query -W`, inside the SIF)

| package | version |
|---|---|
| `opendronemap` | `3.6.0~1binaryblob1` |
| `python3-opensfm` | `0.0.2` |
| `libxml2-16` | `2.15.2+dfsg-0.1` |
| `python3-mrcal` | `2.4.1-4+b2` |
| `python3-numpysane` | `0.42-1` |
| `python3-vnlog` | `1.40-1` |
| `zsh` | `5.9-8+b23` |

`dpkg --verify opendronemap python3-opensfm`: no differences from the packages' checksums.
These are JPL builds; the version strings say nothing about which upstream commits they contain.

## Extraction (`logs/extract.stderr.log`)

`tar --no-same-owner --no-same-permissions -xzf ... -C envs/jpl-trixie-rootfs`, 10 min 22 s,
exit 2. All 8 errors are `./dev/<node>: Cannot mknod: Operation not permitted` (the 8 device
nodes above; the wiki notes the same). 123,624 entries on disk = 123,632 minus those 8. 12 GB.

## SIF (`manifests/image.sha256`, `logs/apptainer-build.log`, `logs/apptainer-inspect.txt`)

`apptainer build envs/quakes-jpl-trixie.sif envs/jpl-trixie-rootfs`, exit 0, 27 min,
no `--fakeroot` needed. 5,462,904,832 bytes,
`7e24112d29802cd9b09bc4c522919e6f0bb25112984942530b8acb9d75f72c04`. The SIF hash differs from the archive's by design.
One warning: "'nodev' mount option set on /scratch/gilbreth" (harmless: there are no device nodes left to store).

## Runtime checks (`logs/smoke-test.log`, `logs/quakes-smoke-12024441.out`)

All with `apptainer exec --cleanenv`:

1. OS is Debian 13 (trixie). 2. `opendronemap`, `opensfm`, `python3`, `zsh` on PATH
(`/usr/bin/opendronemap` and `/usr/bin/opensfm` are wrapper scripts; ODM is in `/opt/opendronemap`).
3. The 7 packages above installed. 4. `dpkg --verify` clean. 5. `import mrcal, numpysane, vnlog, opensfm`.
6. Entry points resolve. 7. `opensfm --help` lists its commands (including `reconstruct`,
`create_rig`, `export_openmvs`). 8. `opendronemap --help` prints ODM's usage.
9. `DensifyPointCloud` (OpenMVS) has no missing libraries and links `libcuda`.
10. GPU through `quakes-jpl.sh`: `nvidia-smi` sees the A100 (driver 595.91.07) and `cuInit`
returns 0 with the ODM wrapper's `LD_LIBRARY_PATH`.

## Blocker found and resolved: GPU driver mismatch

With plain `--nv`, `nvidia-smi` in the image fails ("Driver/library version mismatch", NVML 550.163)
and `cuInit` returns 803 (`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`). Cause: the image ships Debian's
NVIDIA user-space libraries 550.163 (`libcuda1`, `libnvidia-ml1`, `libnvidia-ptxjitcompiler1`),
and they come before Apptainer's host-driver libraries on the library path. Moving
`/.singularity.d/libs` first fixes `nvidia-smi` only: `/usr/bin/opendronemap` replaces
`LD_LIBRARY_PATH`. Fix (run time only, the SIF is unchanged): `quakes-jpl.sh` binds
`/usr/lib64/<lib>.so.595.91.07` over each `/usr/lib/x86_64-linux-gnu/nvidia/current/<lib>.so.550.163.01`.
It reads the host driver version from `/proc/driver/nvidia/version`, so it follows driver updates.
Without it, ODM would still enable GPU features (it only checks that `nvidia-smi` exists) and fail.

## Project scripts switched to this image (2026-10-26)

The earlier image built from public sources (`envs/quakes-odm.sif`, recipe
`photogrammetry/cluster/quakes.def`) was removed, with the `_raw_convert` build products
compiled against it. `gs/common.sh`, `gs/make_dsm_dtm.sh`, `gs/prepare_dataset.sh`,
`photogrammetry/cluster/run-whitney.sh` and `export-figures.py` now default to
`envs/quakes-jpl-trixie.sif`. Checks against outputs made with the old image (copies in a
scratch folder; nothing in `runs/` was overwritten):

- `gs/make_dsm_dtm.sh` (PDAL 2.9.0, git 831631, in both images) on the 2-camera ODM cloud:
  DSM and DTM **identical** to `runs/whitney/gs/dsm-dtm-compare/odm_{dsm,dtm}.tif`
  (same grid and valid cells, max difference 0.0 m).
- `opensfm export_colmap --binary` on a copy of `runs/whitney/photogrammetry/opensfm`:
  `cameras.bin` and `images.bin` byte-identical; `points3D.bin` has the same 46,739 points with
  identical positions, colours, errors and track membership; only the order of the
  observations inside 44,270 tracks differs (no effect on gsplat, which reads positions and colours).
- **Not runnable in this image:** `raw-to-images-whitney.sh`. It needs the `_raw_convert` C extension,
  and the image has no compiler, `make`, `pkg-config` or mrbuild. Earlier conversions remain in
  `runs/*/raw-converted`; `data/whitney-batch1` holds JPL's own.
- No OpenSfM reconstruction or ODM run has been done in this image yet (see `WORKFLOW_VERIFIED`).

## Not done / next actions

- **Authenticity:** ask Dima for the archive's SHA-256 and compare with `bd385bd3...abd22`.
- **`WORKFLOW_VERIFIED`:** needs (1) `github.jpl.nasa.gov/kogan/uavsar`, or confirmation that
  `run-opensfm-opendronemap-handoff.sh` in `github.com/dkogan/quakes` is the same script;
  (2) the image list of `whitney-small-images`; (3) Dima's `opensfm/reconstruction.json`,
  `odm_dem/dsm.tif` and logs, to compare against. Then one Whitney run in this image, sized
  to the allocation (not `processes: 64`).
- **No reconstruction was run.** OpenMVS on the GPU has been checked only as far as library
  loading and `cuInit`, not with real data.
- **Clean-up, for the user to decide:** `envs/jpl-trixie-rootfs/` (12 GB, 123,624 files) is no
  longer needed to run the SIF; deleting it frees ~6% of the scratch file quota. Keep the
  original tarball (and its hash); scratch may be purged, so a copy of the tarball and SIF on
  depot is advisable.
