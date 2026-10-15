#!/usr/bin/env bash
# UPDATE136: E1 of the onvlm2 Ray+IR Top-5 vs 2000 / 26123 on onvlm2 GPU0+GPU1 (never GPU3).
#   bash e1_136.sh jobs.txt <slots_per_gpu>     jobs.txt lines: name family arm teacher_pt seed ir_mode [sensor lidar|cam]
# Arms: G (GT map), GV (GT masked to LiDAR-visible cells), v25or (deployed LiDAR processor: L0S + DUAL2 DDEEP.7).
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$R"
jobs=$1; per=${2:-3}; read -ra gpus <<<"${E1_GPUS:-0 1}"
export GAST_IMAGE=${GAST_IMAGE:-/data/users/bsseo/gd_lab_isaaclab.sif} GAST_CONTAINER_PYTHON=${GAST_CONTAINER_PYTHON:-/data/users/bsseo/venv/bin/python}
mkdir -p outputs logs; : > logs/queue.lock; echo 0 > logs/queue.next
next(){ ( flock 9; i=$(cat logs/queue.next); echo $((i+1)) > logs/queue.next; echo $i ) 9>logs/queue.lock; }
worker(){ local gpu=$1 slot=$2
  while :; do i=$(next); line=$(sed -n "$((i+1))p" "$jobs"); [ -z "$line" ] && break
    read -r name fam arm teacher seed ir sensor camoff <<<"$line"; O=$R/outputs/$name
    [ -s $O/result.json ] && continue
    rm -rf $O; mkdir -p $O; s=$(date +%s); extra=(); a=$arm
    [ "$arm" = v25or ] && { a=L0S; extra=(U130_DDEEP=0.7 U130_DUAL=2); }
    setsid env GAST_GPU=$gpu GD_LAB_REPLAY_LOCK=$R/logs/replay_g${gpu}s${slot}.lock GAST_USD_TMP=$R/logs/usd_tmp_g${gpu}s${slot} U129_FAMILY=$fam UPDATE116_GPU_LOCK_HELD=1 \
      U127_SENSOR_COUNT=1 OMP_NUM_THREADS=2 U136_IR=$ir U130_SENSOR=${sensor:-lidar} U136_CAM_OFF=${camoff#-} U130_TEACHER=$teacher "${extra[@]}" \
      timeout -k 60 2400 bash experiments/e1/run_trial128.sh $a $seed 32 $O/result.json 30 > $O/run.log 2>&1 &
    pid=$!; wait $pid; e=$?; kill -9 -- -$pid 2>/dev/null
    echo "$(date +%T) gpu$gpu.$slot $name exit=$e result=$( [ -s $O/result.json ] && echo 1 || echo 0) wall=$(( $(date +%s)-s ))s"
  done; }
for g in "${gpus[@]}"; do for s in $(seq 1 $per); do worker $g $s & sleep 20; done; done
wait; echo ALLDONE
