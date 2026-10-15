#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ "${UPDATE116_GPU_LOCK_HELD:-0}" == 1 ]] || exit 2
export U118=1 E1_B2_METRICS=1 U116_ARM=$1 U116_FAMILY=$6 U116_NOISE=realistic U116_IR=physical
export PYTHONDONTWRITEBYTECODE=1 GD_LAB_V21_SPEED_RAMP_STEPS=0
export TMPDIR="$root/logs/tmp" XDG_CACHE_HOME="$root/logs/cache" MPLCONFIGDIR="$root/logs/mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
unset GAST_PROBE_OUT GAST_PROBE_DRIVER
bash "$root/scripts/gast/run_heightmap121.sh" teacher \
 --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
 --teacher_checkpoint "$root/checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt" \
 --seed "$2" --num_envs "$3" --gap_terrain_columns 20 --e1_condition c_teacher_full \
 --replay_out "$4" --replay_seconds "${5:-30}" --replay_warmup_seconds 0 --replay_vx 1.0 --replay_cloud_envs 0 \
 --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 --no_camera_noise --gap_ghost_probability 0
