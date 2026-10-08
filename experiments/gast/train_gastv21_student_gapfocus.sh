#!/usr/bin/env bash
# Gap-focused fine-tune of a GAST student distilled from a GAST v2.1 teacher: same recipe as
# train_gast19840_gapfocus.sh (student drives, 20/30 terrain columns are platform gaps, near-gap samples
# weigh 4x, lr 1e-4), only the teacher differs, so the two fine-tunes are comparable.
#   experiments/gast/train_gastv21_student_gapfocus.sh <teacher_package_dir> <student.pt> [captures=8000] [num_envs=256] [run_name]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
package="$(realpath "${1:?teacher package dir, e.g. checkpoints/teachers/gast/gast_v21_2000_20261007}")"
student="$(realpath "${2:?student checkpoint}")"
captures="${3:-8000}"; envs="${4:-256}"
teacher="$(ls "$package"/teacher/model_*.pt | head -1)"
(cd "$package" && sha256sum -c --quiet SHA256SUMS.txt) || { echo 'Teacher package hash mismatch' >&2; exit 1; }
grep -qx 'camera_profile: vendor_new' "$package/params/env.yaml" || { echo 'Teacher package is not vendor_new' >&2; exit 1; }
test -f "$student"
start=$(basename "$student" .pt | grep -oE '[0-9]+$' | sed 's/^0*//')
total=$((start + captures))
export GD_LAB_V21_SPEED_RAMP_STEPS=0
exec bash "$root/scripts/gast/run_student_live.sh" student \
  --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
  --teacher_checkpoint "$teacher" --student_resume "$student" --resume_lr "${GAST_RESUME_LR:-1e-4}" \
  --num_envs "$envs" --bptt_steps 16 --iterations "$total" --save_interval "${GAST_SAVE_INTERVAL:-500}" \
  --top5_start_iteration $((start + ${GAST_TOP5_DELAY:-1000})) \
  --gap_loss_weight "${GAST_GAP_WEIGHT:-4}" --gap_terrain_columns "${GAST_GAP_COLUMNS:-20}" \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --perception_run_name "${5:-gastv21_$(basename "$package")_student${start}_gapfocus_${envs}env_w${GAST_GAP_WEIGHT:-4}_$(date +%Y%m%d_%H%M%S)}"
