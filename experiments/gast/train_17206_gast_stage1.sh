#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
teacher="$root/logs/vision_rbq10_dreamwaq/arm_4/2026-10-03_18-52-19_dwb_v3_6_21_b1_18_ray_20k_1024_resume2000r1_20261003/best_top5/17206_top1.pt"
test -f "$teacher" || { echo "Missing teacher checkpoint: $teacher" >&2; exit 1; }
actual=$(sha256sum "$teacher" | cut -d ' ' -f1)
[[ "$actual" == a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d ]] || { echo 'Teacher hash mismatch' >&2; exit 1; }
run_name="${1:-gast17206_env256_stage500_20261004}"
[[ ! -e "$root/logs/gast/arm4/$run_name" ]] || { echo 'Run already exists; refusing overwrite' >&2; exit 1; }
# First staged smoke: 256 env x 500 captures; later stages resume model+optimizer.
exec bash "$root/scripts/gast/run_student_live.sh" student \
  --task Gd-BivtGastStudent-Rbq10-Dreamwaq-Vision-v0 \
  --teacher_checkpoint "$teacher" --num_envs 256 --bptt_steps 16 \
  --iterations 500 --lr .0003 --save_interval 1000 \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --top5_start_iteration 5000 --perception_run_name "$run_name" \
  --live_config "$root/logs/gast/arm4/gast17206_student_live_20261004.json"
