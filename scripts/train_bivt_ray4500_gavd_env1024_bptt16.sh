#!/usr/bin/env bash
# Explicit requested configuration. Never stops another training job.
set -euo pipefail
cd "$(dirname "$0")/.."
if pgrep -f '^/home/user/workspace/venv_apptainer/bin/python (scripts/train_student.py|scripts/distill_student.py)' >/dev/null; then
  echo 'Another student training is active. Do not launch 1024-camera-env training concurrently; capacity preflight required.' >&2
  exit 1
fi
# Use a short capture count and unique run name for capacity preflight.
exec bash scripts/train_bivt_ray4500_gavd_5090.sh "${1:-20000}" 1024 \
  "${2:-bivt_ray4500_gavd_env1024_bptt16_20000_20261001}" 16
