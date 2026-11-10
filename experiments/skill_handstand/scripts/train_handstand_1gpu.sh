#!/usr/bin/env bash
# Handstand skill on one GPU.  bash scripts/train_handstand_1gpu.sh <run_name> [envs] [iterations]
#   HANDSTAND_VARIANT (v3 default), HS_INIT (same-layout checkpoint), HS_EXTRA (hydra overrides), HS_GPU (0),
#   HANDSTAND_MIN_STD, HANDSTAND_ANCHOR_WEIGHT, HANDSTAND_START_LEVEL; GD_LAB_VRL_ROOT = training framework checkout.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${GD_LAB_VRL_ROOT:-$(cd "$root/../.." && pwd)}"
run=${1:?run name}; envs=${2:-1024}; iters=${3:-30}
dwb="$repo/checkpoints/00_blind/0_blind_walk_a6000/teacher/model_38000.pt"
if [[ -z "${GD_LAB_SIF:-}" ]]; then
  for c in /data/users/bsseo/gd_lab_isaaclab.sif "$HOME/workspace/gd_lab_isaaclab.sif"; do [[ -f $c ]] && GD_LAB_SIF=$c && break; done
fi
if [[ -z "${GD_LAB_PYTHON:-}" ]]; then
  for c in /data/users/bsseo/venv/bin/python "$HOME/workspace/venv/bin/python" "$HOME/workspace/venv_apptainer/bin/python"; do
    [[ -L $c || -e $c ]] && GD_LAB_PYTHON=$c && break
  done
fi
usd_tmp="$root/logs/usd_tmp/handstand_$run"; mkdir -p "$usd_tmp"
cd "$root"
exec env -u PYTHONPATH PYTHONPATH="$repo/src:$root/src" GD_LAB_VRL_ROOT="$repo" GD_LAB_LOG_ROOT="$root/logs" \
  CUDA_VISIBLE_DEVICES="${HS_GPU:-0}" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES \
  apptainer exec --nv --writable-tmpfs --bind "$usd_tmp:/tmp/IsaacLab" "$GD_LAB_SIF" \
  "$GD_LAB_PYTHON" scripts/train_handstand.py --headless --device cuda:0 --seed 42 \
  --num_envs "$envs" --max_iterations "$iters" --run_name "$run" --warm_start "$dwb" \
  ${HS_INIT:+--init_from "$HS_INIT"} ${HS_EXTRA:-}
