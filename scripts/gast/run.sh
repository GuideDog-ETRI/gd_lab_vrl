#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mode=${1:?Usage: run.sh teacher|student [arguments]}
shift
case "$mode" in teacher|student) ;; *) exit 2 ;; esac
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="${GAST_GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$root/logs/usd_tmp"
if [[ -n "${GAST_PYTHON:-}" ]]; then
    exec "$GAST_PYTHON" "scripts/gast/train_${mode}.py" --headless --device cuda:0 --seed 42 \
      --logger tensorboard --experiment_name gast/arm4 "$@"
fi
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
 "${GAST_IMAGE:-/home/user/workspace/gd_lab_isaaclab.sif}" \
 "${GAST_CONTAINER_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}" \
 "scripts/gast/train_${mode}.py" --headless --device cuda:0 --seed 42 --logger tensorboard \
 --experiment_name gast/arm4 "$@"
