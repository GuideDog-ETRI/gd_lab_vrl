#!/usr/bin/env bash
# UPDATE157: E1 pose(odom)-error evaluation of the frozen 3906 teacher (inference only).
#   run_pose_error157.sh P0|P1|P2|P3|off [seed=42] [seconds=30] [tag]   env: GAST_GPU, GD_LAB_SIF, GD_LAB_PYTHON
#   overrides: U157_RW_XY U157_RW_YAW_DEG U157_K_SLIP U157_WHITE_XY U157_WHITE_YAW_DEG U157_DELAY_MS
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
level=${1:?usage: run_pose_error157.sh P0|P1|P2|P3|off [seed=42] [seconds=30] [tag]}
seed=${2:-42}; seconds=${3:-30}; tag=${4:-${level}_s${seed}}
case "$level" in P0|P1|P2|P3|off) ;; *) exit 2 ;; esac
[[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
out="$root/outputs/update157_pose_err_20261018/$tag"
mkdir -p "$out/tmp" "$out/mpl" "$root/logs/usd_tmp"
[[ ! -e "$out/result.json" && ! -e "$out/run.log" ]] || { echo 'refusing output overwrite'; exit 2; }
cd "$root"
export PYTHONPATH="$root/src:$root/tools/rl_replay" PYTHONDONTWRITEBYTECODE=1
export TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GAST_GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$out/tmp" MPLCONFIGDIR="$out/mpl"
export U118=1 U126_ARM=L0S U116_FAMILY=gap U130_SENSOR=cam
export U130_DUAL=2 U130_DDEEP=0.7 U127_SENSOR_COUNT=1 U136_CAM_FILTER=1 U136_IR=native
export GD_LAB_V21_SPEED_RAMP_STEPS=0 GD_LAB_REPLAY_LOCK="$out/replay.lock"
if [[ "$level" == off ]]; then unset U157_POSE_ERR; else export U157_POSE_ERR="$level"; fi
teacher="$root/checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt"
entry=scripts/gast/eval_heightmap121.py
if [[ ${U157_REFERENCE_ENTRY:-} == 1 ]]; then
  [[ "$level" == off ]] || exit 2
  entry="$root/outputs/update157_pose_err_20261018/off_reference_driver.py"
fi
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
  "${GD_LAB_SIF:-$HOME/workspace/gd_lab_isaaclab.sif}" "${GD_LAB_PYTHON:-$HOME/workspace/venv/bin/python}" \
  "$entry" --headless --device cuda:0 --seed "$seed" \
  --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
  --teacher_checkpoint "$teacher" --num_envs 32 --gap_terrain_columns 20 \
  --e1_condition c_teacher_full --replay_out "$out/result.json" --replay_seconds "$seconds" \
  --replay_warmup_seconds 0 --replay_vx 1.0 --replay_cloud_envs 0 \
  --camera_interval_ms 100 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --no_camera_noise --gap_ghost_probability 0 > "$out/run.log" 2>&1
