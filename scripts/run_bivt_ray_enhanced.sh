#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode=${1:-train}
checkpoint=${2:?Provide an absolute resume checkpoint}
test -s "$checkpoint"
cd "$root"
case "$mode" in
 smoke) envs=64; iterations=(--max_iterations 3) ;;
 train) envs=4096; iterations=(--target_iterations 20000) ;;
 *) exit 2 ;;
esac
run_id="$(date +%Y-%m-%d_%H-%M-%S)_ray_occlusion_v2_${mode}"
export PYTHONPATH="$root/src" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${BIVT_GPU:-0}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/online_top5.json"
export GD_LAB_STOP_FILE="$root/logs/${run_id}.stop"
mkdir -p "$root/logs/usd_tmp/$run_id" "$root/logs/launches/$run_id"
git rev-parse HEAD > "$root/logs/launches/$run_id/commit.txt"
git diff --binary > "$root/logs/launches/$run_id/changes.patch"
sha256sum "$checkpoint" > "$root/logs/launches/$run_id/resume.sha256"
tar --exclude='__pycache__' -czf "$root/logs/launches/$run_id/source.tar.gz" src scripts configs
printf 'Ray v2 run=%s stop_file=%s checkpoint=%s\n' "$run_id" "$GD_LAB_STOP_FILE" "$checkpoint"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp/$run_id:/tmp/IsaacLab" \
 "${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}" \
 "${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}" scripts/train_bivt.py \
 --headless --device cuda:0 --seed 42 --logger tensorboard \
 --task Gd-VrlBlindStartRaycast-Rbq10-Dreamwaq-v0 --num_envs "$envs" \
 --resume_checkpoint "$checkpoint" "${iterations[@]}" --run_name "$run_id" \
 agent.num_steps_per_env=100 agent.algorithm.num_learning_epochs=5 agent.algorithm.num_mini_batches=4 \
 agent.algorithm.min_learning_rate=0.00003 agent.save_interval=100 agent.top5_min_spacing=100
