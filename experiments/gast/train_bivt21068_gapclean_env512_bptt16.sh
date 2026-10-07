#!/usr/bin/env bash
# GAST student from the BIVT-Ray Clean-gap teacher 21068 (vendor_new cameras, ARM4).
#   experiments/gast/train_bivt21068_gapclean_env512_bptt16.sh [iterations=20000] [run_name]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../gast" && pwd)"
package="$root/../checkpoints/teachers/bivt/ray_gap_clean_vendor_new_top1_21068_20261005"
teacher="$package/teacher/21068_top1.pt"
test -f "$teacher" && test -f "$package/params/env.yaml"
actual=$(sha256sum "$teacher" | cut -d ' ' -f1)
[[ "$actual" == fa397d22a949e24f74312b5270b5954a4ddb15c7451ea85ba8f64dbeeb7c9501 ]] || { echo 'Teacher hash mismatch' >&2; exit 1; }
grep -qx 'camera_profile: vendor_new' "$package/params/env.yaml" || { echo 'Teacher package is not vendor_new' >&2; exit 1; }
exec bash "$root/scripts/run.sh" student \
  --task Gd-BivtGastStudent-Rbq10-Dreamwaq-Vision-v0 \
  --teacher_checkpoint "$teacher" --num_envs 512 --bptt_steps 16 \
  --iterations "${1:-20000}" --lr .0003 --save_interval 200 \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --perception_run_name "${2:-bivt21068_gapclean_gast_env512_bptt16_20000_$(date +%Y%m%d)}"
