#!/usr/bin/env bash
# Prepared only: invoke for v3 completion and explicit GPU execution approval.
set -euo pipefail
student="${1:?usage: run_e1_v3_seed.sh <student_ckpt> <run_name>}"
run="${2:?run_name required}"
[[ -f "$student" && "$run" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
student="$(realpath "$student")"
[[ "${E1_GPU_APPROVED:-0}" == 1 ]] || { echo 'NEED APPROVAL: E1 GPU execution for v3'; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
out="$root/outputs/$run"
mkdir "$out"
export PYTHONPATH="$root/src:$root/tools/rl_replay"
export PYTHONDONTWRITEBYTECODE=1 GD_LAB_V21_SPEED_RAMP_STEPS=0 E1_B2_METRICS=1
export TMPDIR="$root/logs/e1_tmp" XDG_CACHE_HOME="$root/logs/e1_cache" MPLCONFIGDIR="$root/logs/e1_mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
unset GAST_PROBE_OUT GAST_PROBE_DRIVER
teacher="$root/checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt"
CUDA_VISIBLE_DEVICES='' APPTAINERENV_CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 nice -n 19 apptainer exec \
 "${GAST_IMAGE:-/home/user/workspace/gd_lab_isaaclab.sif}" \
 "${GAST_CONTAINER_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}" -B \
 "$root/tools/rl_replay/e1_v3_adapter.py" "$student" > "$out/resume_preflight.json"
seed="${E1_SEED:-42}"; tag=s${seed}_vx1.0_a_student_gate
bash "$root/tools/rl_replay/run_e1_v3_live.sh" \
 --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
 --teacher_checkpoint "$teacher" --student_resume "$student" --seed "$seed" --num_envs 32 --gap_terrain_columns 20 \
 --e1_condition a_student_gate --replay_out "$out/$tag.json" \
 --replay_seconds 30 --replay_warmup_seconds 0 --replay_vx 1.0 --replay_cloud_envs 0 \
 --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
 --no_camera_noise --gap_ghost_probability 0 > "$out/$tag.log" 2>&1
python3 -B "$root/tools/rl_replay/summarize_e1.py" --b2 "$out" "$out/$tag.json"
