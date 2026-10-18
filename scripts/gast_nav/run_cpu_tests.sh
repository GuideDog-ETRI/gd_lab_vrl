#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/tests/gast_nav/.runtime" APPTAINER_CACHEDIR="$PWD/tests/gast_nav/.runtime"
mkdir -p "$TMPDIR"
export PYTHONPATH="$PWD/tests/gast_nav:$PWD/src:${ISAACLAB_SRC:-$HOME/workspace/IsaacLab/source/isaaclab}"
exec apptainer exec "${GD_LAB_SIF:-$HOME/workspace/gd_lab_isaaclab.sif}" "${GD_LAB_PYTHON:-$HOME/workspace/venv_apptainer/bin/python}" -B -m unittest discover -s tests/gast_nav -p 'test_policy*.py' -v
