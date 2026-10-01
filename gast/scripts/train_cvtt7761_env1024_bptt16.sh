#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if pgrep -f '^/home/user/workspace/venv_apptainer/bin/python (scripts/train_student.py|scripts/distill_student.py)' >/dev/null; then
  echo 'Another student training is active; capacity preflight must run alone.' >&2
  exit 1
fi
exec bash "$root/scripts/train_cvtt7761_student_20k.sh" \
  --num_envs 1024 --bptt_steps 16 --iterations "${1:-20000}" \
  --perception_run_name "${2:-cvtt7761_gast_env1024_bptt16_20000_20261001}"
