#!/usr/bin/env bash
# Gap-focused fine-tune of the GAST student 19840 (teacher BIVT-Ray 21068, vendor_new, ARM4).
# Student drives (DAgger), 2/3 of the terrain columns are platform gaps, near-gap samples weigh more.
#   gast/scripts/train_gast19840_gapfocus.sh <student_19840.pt> [captures=8000] [num_envs=256] [run_name]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
student="$(realpath "${1:?student checkpoint (student_top5_iter_19840.pt)}")"
captures="${2:-8000}"
envs="${3:-256}"
package="$root/../checkpoints/teachers/bivt/ray_gap_clean_vendor_new_top1_21068_20261005"
teacher="$package/teacher/21068_top1.pt"
test -f "$teacher" && test -f "$package/params/env.yaml" && test -f "$student"
[[ "$(sha256sum "$teacher" | cut -d ' ' -f1)" == fa397d22a949e24f74312b5270b5954a4ddb15c7451ea85ba8f64dbeeb7c9501 ]] || { echo 'Teacher hash mismatch' >&2; exit 1; }
grep -qx 'camera_profile: vendor_new' "$package/params/env.yaml" || { echo 'Teacher package is not vendor_new' >&2; exit 1; }
start=$(basename "$student" .pt | grep -oE '[0-9]+$')
total=$((start + captures))
exec bash "$root/scripts/run_student_live.sh" student \
  --task Gd-BivtGastStudent-Rbq10-Dreamwaq-Vision-v0 \
  --teacher_checkpoint "$teacher" --student_resume "$student" --resume_lr "${GAST_RESUME_LR:-1e-4}" \
  --num_envs "$envs" --bptt_steps 16 --iterations "$total" --save_interval "${GAST_SAVE_INTERVAL:-500}" \
  --top5_start_iteration $((start + ${GAST_TOP5_DELAY:-1000})) \
  --gap_loss_weight "${GAST_GAP_WEIGHT:-4}" --gap_terrain_columns "${GAST_GAP_COLUMNS:-20}" \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --perception_run_name "${4:-gast${start}_gapfocus_${envs}env_w${GAST_GAP_WEIGHT:-4}_$(date +%Y%m%d_%H%M%S)}"
