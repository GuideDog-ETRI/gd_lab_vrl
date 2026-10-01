#!/usr/bin/env bash
# usage: bs_run.sh <gpu> <log> <args...>
gpu=$1; log=$2; shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for attempt in 1 2 3; do
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=$gpu TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 apptainer exec --nv --writable-tmpfs --bind $PWD/logs/usd_tmp/blindstart:/tmp/IsaacLab ${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif} ${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python} scripts/train_bivt.py --headless --device cuda:0 --seed 42 --logger tensorboard "$@" >> $log 2>&1
  code=$?
  # carb shm PID-name collision aborts at startup only; retry that case.
  if [ $code -eq 134 ] && grep -q "Failed to create shared memory" $log; then echo "retry $attempt" >> $log; continue; fi
  break
done
echo "EXIT=$code" >> $log
exit $code
