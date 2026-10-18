#!/usr/bin/env bash
# UPDATE162: pilot is the only currently authorized real Nav training scope.
set -euo pipefail
mode="${1:?usage: train_gast_nav_3gpu.sh pilot|train [train.py options]}"
shift
case "$mode" in
  pilot) export GAST_NAV_PILOT=1; mode_flag=--pilot ;;
  train) test "${GAST_NAV_TRAINING_AUTHORIZED:-0}" = 1 || { echo 'training authorization required' >&2; exit 2; }; mode_flag=--train ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export CUDA_VISIBLE_DEVICES=0,1,2 APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2
export TRAIN_ARM=4 GAST_MATH_SDPA=1 OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y%m%d_%H%M%S)_nav_${mode}}"
[[ "$GAST_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid run id' >&2; exit 2; }
launch="$root/logs/launches/$GAST_RUN_ID"
mkdir "$launch" -p
# A run id cannot silently overwrite a previous launch.
(set -o noclobber; printf '%s\n' "$mode" > "$launch/LAUNCH_STARTED")
cp "${BASH_SOURCE[0]}" "$launch/launcher.sh"
cp -r configs/gast_nav "$launch/configs"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" src scripts configs
sha256sum "$launch/source.tar.gz" > "$launch/source.sha256"
sha256sum checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt > "$launch/source_teacher.sha256"
printf '%q ' "$mode" "$@" > "$launch/arguments.txt"
mkdir -p "$launch/runtime/tmp" "$launch/runtime/usd" "$launch/runtime/cache" "$launch/runtime/mpl"
export TMPDIR="$launch/runtime/tmp" APPTAINER_TMPDIR="$launch/runtime/tmp" XDG_CACHE_HOME="$launch/runtime/cache" MPLCONFIGDIR="$launch/runtime/mpl"
export TRITON_CACHE_DIR="$launch/runtime/cache/triton" CUDA_CACHE_PATH="$launch/runtime/cache/cuda"
export TORCH_HOME="$launch/runtime/cache/torch" NUMBA_CACHE_DIR="$launch/runtime/cache/numba"
exec apptainer exec --nv --writable-tmpfs --bind "$launch/runtime/usd:/tmp/IsaacLab" \
  "${GD_LAB_SIF:?set GD_LAB_SIF}" "${GD_LAB_PYTHON:?set GD_LAB_PYTHON}" -B \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
  scripts/gast_nav/train.py "$mode_flag" --distributed --total_envs 4096 --headless "$@"
