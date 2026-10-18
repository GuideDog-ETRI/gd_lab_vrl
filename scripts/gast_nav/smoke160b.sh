#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
out="$root/tests/gast_nav/update160b_smoke"
mkdir -p "$out/runtime/tmp" "$out/runtime/cache" "$out/runtime/mpl"
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$out/runtime/tmp" APPTAINER_TMPDIR="$out/runtime/tmp" XDG_CACHE_HOME="$out/runtime/cache" MPLCONFIGDIR="$out/runtime/mpl"
export NUMBA_CACHE_DIR="$out/runtime/cache/numba" TORCH_HOME="$out/runtime/cache/torch" PYTHONPATH="$root/src:/home/user/workspace/IsaacLab/source/isaaclab"
exec apptainer exec --nv /home/user/workspace/gd_lab_isaaclab.sif /home/user/workspace/venv_apptainer/bin/python -B \
  scripts/gast_nav/train.py --smoke --stage B --headless --output "$out"
