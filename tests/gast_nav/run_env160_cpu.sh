#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mkdir -p tests/gast_nav/.runtime160/tmp
export TMPDIR="$root/tests/gast_nav/.runtime160/tmp" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$root/tests/gast_nav:$root/src:/home/user/workspace/IsaacLab/source/isaaclab"
exec apptainer exec /home/user/workspace/gd_lab_isaaclab.sif /home/user/workspace/venv_apptainer/bin/python -B \
  -m pytest tests/gast_nav/test_env160.py -q -p no:cacheprovider --basetemp "$root/tests/gast_nav/.runtime160/tmp/pytest" "$@"
