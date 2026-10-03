#!/usr/bin/env bash
# usage: run_bivt_onnx.sh <gpu> <log> <blind_policy.onnx> [train_bivt.py options...]
set -u
gpu=${1:?GPU id required}
log=${2:?log path required}
onnx=${3:?deployed blind ONNX path required}
shift 3
onnx=$(realpath "$onnx")
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$(dirname "$log")"
mkdir -p "$PWD/logs/usd_tmp/blindstart"
[[ -f "$onnx" ]] || { echo "ONNX policy not found: $onnx" >&2; exit 2; }
num_envs=${BIVT_NUM_ENVS:-256}
iterations=${BIVT_MAX_ITERATIONS:-20000}
sif=${GD_LAB_SIF:-/home/user/workspace/gd_lab_isaaclab.sif}
python=${GD_LAB_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}
[[ -f "$sif" ]] || { echo "Isaac Lab SIF not found: $sif" >&2; exit 2; }
apptainer exec "$sif" test -x "$python" || { echo "Python executable not found inside SIF: $python" >&2; exit 2; }

for attempt in 1 2 3; do
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    apptainer exec --nv --writable-tmpfs --bind "$PWD/logs/usd_tmp/blindstart:/tmp/IsaacLab" \
    "$sif" \
    "$python" \
    scripts/train_bivt_onnx.py --headless --device cuda:0 --seed 42 --logger tensorboard \
    --task Gd-VrlBlindStartRaycast-Rbq10-Dreamwaq-v0 --num_envs "$num_envs" --max_iterations "$iterations" --blind_onnx_init "$onnx" "$@" >> "$log" 2>&1
  code=$?
  if [ "$code" -eq 134 ] && grep -q "Failed to create shared memory" "$log"; then
    echo "retry $attempt" >> "$log"
    continue
  fi
  break
done
echo "EXIT=$code" >> "$log"
exit "$code"
