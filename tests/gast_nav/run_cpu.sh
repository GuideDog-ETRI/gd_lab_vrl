#!/usr/bin/env bash
# GAST-Nav v1 map-side CPU fixtures (pytest). No simulator, GPU or learning. Runs map and policy fixtures together.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mkdir -p tests/gast_nav/.runtime/tmp
export TMPDIR="$root/tests/gast_nav/.runtime/tmp" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$root/tests/gast_nav:$root/src:${ISAACLAB_SRC:-$HOME/workspace/IsaacLab/source/isaaclab}"
exec apptainer exec "${GD_LAB_SIF:-$HOME/workspace/gd_lab_isaaclab.sif}" "${GD_LAB_PYTHON:-$HOME/workspace/venv_apptainer/bin/python}" -B \
  -m pytest tests/gast_nav -q -p no:cacheprovider --basetemp "$root/tests/gast_nav/.runtime/tmp/pytest" "$@"
