# JPL QUAKES image (OpenSfM + OpenDroneMap) on Gilbreth

Deployment of Dima Kogan's prebuilt Debian/trixie filesystem
(`quakes-opensfm-opendronemap-trixie.tar.gz`, from the
[packaging wiki page](https://github.com/dkogan/quakes/wiki/2026-05-27--packaging-opensfm-opendronemap))
as an Apptainer image. Nothing is rebuilt: the SIF is the same files as the tarball.
This replaces the earlier image built from public sources (`envs/quakes-odm.sif`,
from `photogrammetry/cluster/quakes.def`), which was removed on 2026-10-26.

Paths are relative to the project root. Status and evidence: [deployment-report.md](deployment-report.md).

## Use

```sh
quakes-deploy/scripts/quakes-jpl.sh opendronemap --help      # = schroot -c quakes -- opendronemap --help
quakes-deploy/scripts/quakes-jpl.sh opensfm --help
quakes-deploy/scripts/quakes-jpl.sh zsh                      # a shell inside the image
```

`quakes-jpl.sh` runs `apptainer exec --cleanenv --nv -B /scratch,/depot envs/quakes-jpl-trixie.sif CMD`
and binds the host's NVIDIA driver libraries over the image's (see GPU below).
`QUAKES_NO_GPU=1` runs without the GPU (and sets `ODM_NO_GPU=1` for ODM);
`QUAKES_SIF=...` picks another image.

| wiki (schroot) | here |
|---|---|
| `/etc/schroot/schroot.conf` entry (root) | none: the SIF is self-contained |
| extra mounts in `desktop/fstab` | `-B /scratch,/depot` (in `quakes-jpl.sh`) |
| NVIDIA "extra dance" | host driver libraries bound over the image's (in `quakes-jpl.sh`) |
| `schroot -c quakes -- CMD` | `quakes-deploy/scripts/quakes-jpl.sh CMD` |

## GPU

The image contains Debian's NVIDIA user-space driver libraries 550.163 (`libcuda1`,
`libnvidia-ml1`, `libnvidia-ptxjitcompiler1`). These must match the host's kernel driver
(Gilbreth: 595.91.07 when deployed). With plain `--nv` they do not: the image's
`/lib/x86_64-linux-gnu` precedes `/.singularity.d/libs` on the library path, so
`nvidia-smi` reports "Driver/library version mismatch" and `cuInit` returns 803. Putting
`/.singularity.d/libs` first is not enough, because `/usr/bin/opendronemap` replaces
`LD_LIBRARY_PATH` with its PDAL path. So `quakes-jpl.sh` binds
`/usr/lib64/<lib>.so.<host version>` onto each `/usr/lib/x86_64-linux-gnu/nvidia/current/<lib>.so.550.163.01`.
This is a run-time mount; the SIF is unchanged. OpenMVS (`DensifyPointCloud`) links
`libcuda`, so ODM needs this to use the GPU. ODM turns GPU features on whenever
`nvidia-smi` exists, which it does in this image, so without the bind it would try the GPU
and fail.

## How the image was made (on a compute node, no root)

```sh
A=quakes-opensfm-opendronemap-trixie.tar.gz
sha256sum $A                                          # manifests/archive.sha256
tar -tvzf $A > quakes-deploy/manifests/archive-listing.txt        # whole archive readable: exit 0
# audit (see deployment-report.md): no absolute or .. paths, nothing written through a symlink,
# 1 internal hard link, 8 device nodes in /dev
mkdir envs/jpl-trixie-rootfs
tar --no-same-owner --no-same-permissions -xzf $A -C envs/jpl-trixie-rootfs   # ~10 min
#   exit 2: only "./dev/<node>: Cannot mknod" x 8, as the wiki says; Apptainer supplies /dev
export APPTAINER_TMPDIR=$RCAC_SCRATCH/.apptainer/tmp APPTAINER_CACHEDIR=$RCAC_SCRATCH/.apptainer/cache
apptainer build envs/quakes-jpl-trixie.sif envs/jpl-trixie-rootfs      # ~27 min, no --fakeroot needed
sha256sum envs/quakes-jpl-trixie.sif                  # manifests/image.sha256
bash quakes-deploy/scripts/verify_image.sh envs/quakes-jpl-trixie.sif quakes-deploy/manifests
```

Files are owned by the building user inside the SIF (extraction without root drops the
archive's root ownership); this does not matter to Apptainer, which runs as the user.

## Scripts

| script | what |
|---|---|
| `scripts/quakes-jpl.sh` | run a command in the image (GPU binds, `/scratch` and `/depot`) |
| `scripts/check_environment.sh` | host facts, read-only (`manifests/environment.txt`) |
| `scripts/verify_image.sh` | smoke tests: OS, packages, `dpkg --verify`, imports, libraries, `--help`, GPU |
| `scripts/quakes_smoke.slurm` | the smoke tests as a batch job: `sbatch -A jahansha -p a100-40gb quakes-deploy/scripts/quakes_smoke.slurm` |
