#!/bin/bash
# GPU-memory capacity test: how many Gaussians fit on the 40 GB A100?
# Short vanilla-3DGS runs on Whitney, raising image resolution (data_factor) and
# loosening densification (grow_grad2d) to push the Gaussian count up. Records
# peak VRAM (torch max_memory_allocated), Gaussian count, PSNR and time per run.
# An out-of-memory run is recorded as OOM: that is the capacity limit.
#
# Usage: bash gs/gpu_capacity_test.sh   (inside a GPU job)
set -uo pipefail
. "$(dirname "$0")/common.sh"
DATA=$DATASET
OUT=${OUT:-$RUNS/gpu-capacity}
STEPS=${STEPS:-7000}
# absolute paths: the trainer runs from gsplat's examples/ directory
OUT=$(realpath -m "$OUT"); DATA=$(realpath -m "$DATA")
CSV=$OUT/capacity.csv
mkdir -p $OUT

[[ -f $CSV ]] || echo "factor,width,height,grow_grad2d,steps,status,num_gs,peak_vram_gb,psnr,ssim,lpips,train_s" > $CSV

# factor  grow_grad2d
RUNS=("2 0.0002" "2 0.0001" "1 0.0001" "1 0.00005")

cd $GSPLAT/examples
for r in "${RUNS[@]}"; do
    read F G <<< "$r"
    TAG=f${F}_g${G}
    RD=$OUT/$TAG
    echo "=== $TAG ($(date +%T))"
    $GS_PY simple_trainer.py default \
        --data_dir $DATA --data_factor $F --result_dir $RD \
        --test_every 8 --max_steps $STEPS --eval_steps $STEPS --save_steps 999999 \
        --strategy.grow-grad2d $G --disable_viewer --disable_video \
        > $OUT/$TAG.log 2>&1
    rc=$?
    W=$((5120 / F)); H=$((3840 / F))
    if [[ $rc -eq 0 && -f $RD/stats/train_step$((STEPS-1))_rank0.json ]]; then
        $GS_PY - $RD $STEPS <<'EOF' | sed "s/^/$F,$W,$H,$G,$STEPS,ok,/" >> $CSV
import json, sys
rd, steps = sys.argv[1], int(sys.argv[2])
t = json.load(open(f"{rd}/stats/train_step{steps-1}_rank0.json"))
v = json.load(open(f"{rd}/stats/val_step{steps-1}.json"))
print(f"{t['num_GS']},{t['mem']:.2f},{v['psnr']:.2f},{v['ssim']:.4f},{v['lpips']:.4f},{t['ellipse_time']:.0f}")
EOF
    else
        status=fail; grep -q "out of memory" $OUT/$TAG.log && status=OOM
        # last Gaussian count the trainer reported before stopping, if any
        n=$(tr '\r' '\n' < $OUT/$TAG.log | grep -oE "Now having [0-9]+ GSs" | tail -1 | grep -oE "[0-9]+")
        echo "$F,$W,$H,$G,$STEPS,$status,${n:-},,,,," >> $CSV
    fi
    tail -1 $CSV
    rm -rf $RD/ckpts
done
echo "done $(date +%T)"
cat $CSV
