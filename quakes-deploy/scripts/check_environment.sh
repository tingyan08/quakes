#!/bin/bash
# Non-destructive checks of the host before deploying the JPL QUAKES image.
# Prints host, Apptainer, Slurm and storage facts; changes nothing.
#   bash quakes-deploy/scripts/check_environment.sh > quakes-deploy/manifests/environment.txt
set -u
echo "date:      $(date -Is)"
echo "host:      $(hostname)"
echo "user:      $(id)"
echo "arch:      $(uname -m)  kernel $(uname -r)"
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
echo "apptainer: $(command -v apptainer) $(apptainer --version 2>&1)"
echo "slurm job: ${SLURM_JOB_ID:-none} partition ${SLURM_JOB_PARTITION:-none} cpus ${SLURM_CPUS_ON_NODE:-?}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-}  APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-}"
echo "user namespaces: max_user_namespaces=$(cat /proc/sys/user/max_user_namespaces 2>/dev/null)"
grep -q "^$(id -un):" /etc/subuid 2>/dev/null && echo "subuid: present" || echo "subuid: none (apptainer --fakeroot uses its limited mode)"
command -v myquota >/dev/null && myquota
df -h "$HOME" "${RCAC_SCRATCH:-/scratch}" 2>/dev/null
