#!/bin/bash
# Smoke test of the JPL QUAKES image as a batch job (no reconstruction).
# Submit from the project root; pass the account/partition your allocation uses:
#   sbatch -A jahansha -p a100-40gb quakes-deploy/scripts/quakes_smoke.slurm
#SBATCH --job-name=quakes-smoke
#SBATCH --time=00:15:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --gres=gpu:1
#SBATCH --output=quakes-deploy/logs/quakes-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
hostname
bash quakes-deploy/scripts/verify_image.sh envs/quakes-jpl-trixie.sif
