#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
teacher="$root/../checkpoints/teachers/cvtt/arm4_7761/7761_top1.pt"
test -f "$teacher"
actual=$(sha256sum "$teacher" | cut -d ' ' -f1)
[[ "$actual" == f7c010331a5a3dac7b2b6c0e78c9678c49fdbb5661cb9f0265ba62b1fd9e0c87 ]] || { echo 'Teacher hash mismatch' >&2; exit 1; }
exec bash "$root/scripts/run.sh" student \
  --task Gd-CvttGastStudent-Rbq10-Dreamwaq-Vision-v0 \
  --teacher_checkpoint "$teacher" --num_envs 16 --iterations 20000 \
  --bptt_steps 64 --lr .0003 --save_interval 200 \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --perception_run_name cvtt7761_gast_student_20000_20261001 "$@"
