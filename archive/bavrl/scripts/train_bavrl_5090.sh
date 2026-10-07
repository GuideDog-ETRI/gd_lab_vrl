#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run=${1:?run name required}
shift
[[ $run =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
mkdir -p "logs/usd_tmp/$run"
export PYTHONPATH="$repo_root/src"
export CUDA_VISIBLE_DEVICES=0 TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
exec apptainer exec --nv --writable-tmpfs --bind "$repo_root/logs/usd_tmp/$run:/tmp/IsaacLab" \
  /home/user/workspace/gd_lab_isaaclab.sif /home/user/workspace/venv_apptainer/bin/python \
  scripts/train_bavrl.py --headless --device cuda:0 \
  --teacher checkpoints/teachers/blind/arm4_38000/teacher/model_38000.pt \
  --teacher-agent checkpoints/teachers/blind/arm4_38000/teacher/params/agent.yaml \
  --output "logs/bavrl/$run" "$@"
