#!/usr/bin/env bash
# PREPARED ONLY. Do not execute until Claude has approved GPU ownership.
set -euo pipefail
[[ "${E1_GPU_APPROVED:-0}" == 1 ]] || { echo 'NEED APPROVAL: Claude GPU go for E1'; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ "$root" == /home/user/gd_project/archive/codex_worktrees/update26_e1 ]] || exit 2
cd "$root"
num_envs="${E1_NUM_ENVS:-32}"
[[ "$num_envs" == 32 || "$num_envs" == 64 ]] || { echo 'Use 32 or 64 envs'; exit 2; }
run="${E1_RUN_NAME:-e1_fix_smoke_$(date +%Y%m%d_%H%M%S)}"
[[ "$run" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
out="$root/outputs/$run"
mkdir "$out"
export PYTHONDONTWRITEBYTECODE=1 GD_LAB_V21_SPEED_RAMP_STEPS=0
export TMPDIR="$root/logs/e1_tmp" XDG_CACHE_HOME="$root/logs/e1_cache" MPLCONFIGDIR="$root/logs/e1_mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
unset GAST_PROBE_OUT GAST_PROBE_DRIVER
teacher="$root/checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt"
student="$root/checkpoints/students/control_seen0_perception_34624.pt"
for seed in ${E1_SEEDS:-42}; do
  for vx in ${E1_SPEEDS:-1.0}; do
    for condition in a_student_gate; do
      tag="s${seed}_vx${vx}_${condition}"
      bash "$root/scripts/gast/run_student_live.sh" student \
        --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
        --teacher_checkpoint "$teacher" --student_resume "$student" \
        --seed "$seed" --num_envs "$num_envs" --gap_terrain_columns 20 \
        --e1_condition "$condition" --replay_out "$out/$tag.json" \
        --replay_seconds 3 --replay_warmup_seconds 0 --replay_vx "$vx" --replay_cloud_envs 0 \
        --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
        --no_camera_noise --gap_ghost_probability 0 \
        > "$out/$tag.log" 2>&1
    done
  done
done
echo "E1 geometry smoke completed: $out (3 simulated seconds, baseline only)"
