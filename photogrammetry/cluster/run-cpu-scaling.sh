#!/bin/bash
# Time the fixed-rig Mammoth run (cluster/run-mammoth.sh) with different numbers of CPUs, one after the other.
# Each run is pinned with taskset to the first N CPUs of the allocation, because the CPU count that
# run-mammoth.sh passes on only reaches OpenSfM's "processes" and ODM's --max-concurrency.
#   photogrammetry/cluster/run-cpu-scaling.sh 32 16
# Writes, per run, to runs/mammoth/cpu-scaling/cpus-N/: the project, and time.txt (GNU time; its maximum
# resident set is that of the largest single process). Stage times: opensfm/profile.log, benchmark.txt.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$ROOT/runs/mammoth/cpu-scaling
LOGS=$ROOT/runs/mammoth/logs

# the CPUs this job may use, e.g. 8-39 -> 8 9 ... 39
allowed=($(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status | tr , '\n' |
            awk -F- '{for (i = $1; i <= ($2 == "" ? $1 : $2); i++) printf "%d ", i}'))

for n in "$@"; do
    (( n <= ${#allowed[@]} )) || { echo "only ${#allowed[@]} CPUs allocated, asked for $n" >&2; exit 1; }
    project=$OUT/cpus-$n
    [[ -e "$project" ]] && { echo "$project exists: not overwriting it" >&2; exit 1; }
    mkdir -p "$project"
    cpus=$(IFS=,; echo "${allowed[*]:0:$n}")
    echo "$(date -Is) start: $n CPUs ($cpus) on $(hostname), job ${SLURM_JOB_ID:-none}"
    status=0
    SLURM_CPUS_PER_TASK=$n PROJECT=$project RIG=1 OPTIMIZE_RIG=1 \
        /usr/bin/time -v -o "$project/time.txt" \
        taskset -c "$cpus" "$ROOT/photogrammetry/cluster/run-mammoth.sh" \
        > "$LOGS/cpu-scaling-$n-$(date +%Y%m%d-%H%M).log" 2>&1 || status=$?
    echo "$(date -Is) end: $n CPUs, exit $status"
    (( status == 0 )) || exit "$status"
done
