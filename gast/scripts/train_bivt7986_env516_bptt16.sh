#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
teacher="$root/../checkpoints/teachers/bivt/ray_enhanced_top1_7986_20261002/teacher/7986_top1.pt"
test -f "$teacher"
actual=$(sha256sum "$teacher" | cut -d ' ' -f1)
[[ "$actual" == 718517b53038384211b5a5d4282061ce0c5830695a1f6f26ba0caf8d07111fda ]] || { echo 'Teacher hash mismatch' >&2; exit 1; }
run_name="${2:-bivt7986_gast_env516_bptt16_20000_20261002}"
[[ ! -e "$root/logs/gast/arm4/$run_name" ]] || { echo 'Run already exists; refusing overwrite' >&2; exit 1; }
# Match the 4500 student experiment, including its deliberately wider 0-150ms delay.
exec bash "$root/scripts/run.sh" student \
  --task Gd-BivtGastStudent-Rbq10-Dreamwaq-Vision-v0 \
  --teacher_checkpoint "$teacher" --num_envs 516 --bptt_steps 16 \
  --iterations "${1:-20000}" --lr .0003 --save_interval 200 \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --perception_run_name "$run_name"
