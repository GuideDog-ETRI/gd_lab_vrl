#!/usr/bin/env bash
# GAST camera student from a frozen GAST v2.1 teacher (full-grid 8-step memory teacher, vendor_new cameras).
#   experiments/gast/train_gastv21_student_live.sh <teacher_package_dir> [captures=20000] [num_envs=512] [run_name]
# The package needs teacher/model_*.pt, params/{agent,env}.yaml and SHA256SUMS.txt (checked here).
# Distills in the teacher's own v2.1 env (26 cm gaps, stair disturbance at full force, commands to 1.2 m/s).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
package="$(realpath "${1:?teacher package dir, e.g. checkpoints/teachers/gast/gast_v21_2000_20261007}")"
captures="${2:-20000}"; envs="${3:-512}"
teacher="$(ls "$package"/teacher/model_*.pt | head -1)"
(cd "$package" && sha256sum -c --quiet SHA256SUMS.txt) || { echo 'Teacher package hash mismatch' >&2; exit 1; }
grep -qx 'camera_profile: vendor_new' "$package/params/env.yaml" || { echo 'Teacher package is not vendor_new' >&2; exit 1; }
name="$(basename "$package")"
export GD_LAB_V21_SPEED_RAMP_STEPS=0   # the teacher already walks the full 0.2-1.2 m/s range
exec bash "$root/scripts/gast/run_student_live.sh" student \
  --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
  --teacher_checkpoint "$teacher" --num_envs "$envs" --bptt_steps 16 --iterations "$captures" --lr .0003 \
  --save_interval "${GAST_SAVE_INTERVAL:-500}" --top5_start_iteration "${GAST_TOP5_START:-2000}" \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --perception_run_name "${4:-${name}_gast_student_${envs}env_${captures}_$(date +%Y%m%d_%H%M%S)}"
