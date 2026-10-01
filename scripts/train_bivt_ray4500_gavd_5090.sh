#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo="$PWD"
count=${1:-20000}
envs=${2:-64}
run=${3:-bivt_ray4500_gavd_20000_20261001}
[[ $count =~ ^[1-9][0-9]*$ && $envs =~ ^[1-9][0-9]*$ && $run =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
teacher="$repo/checkpoints/teachers/bivt/ray_4500_20261001"
(cd "$teacher" && sha256sum -c SHA256SUMS.txt)
mkdir "logs/$run.launch"
cp "$teacher/teacher/model_4500.pt" "logs/$run.launch/teacher.pt"
cp "$teacher/source_4500.json" "logs/$run.launch/"
cp -r "$teacher/params" "logs/$run.launch/teacher_params"
cp -r src/gd_lab/students src/gd_lab/teachers/bivt "logs/$run.launch/"
cp scripts/distill_student.py "$0" "logs/$run.launch/"
mkdir -p "logs/usd_tmp/$run"
export PYTHONPATH="$repo/src" CUDA_VISIBLE_DEVICES=0 TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
exec apptainer exec --nv --writable-tmpfs --bind "$repo/logs/usd_tmp/$run:/tmp/IsaacLab" \
 /home/user/workspace/gd_lab_isaaclab.sif /home/user/workspace/venv_apptainer/bin/python \
 scripts/distill_student.py --task Gd-VrlRayStudent-Rbq10-Dreamwaq-Vision-v0 \
 --headless --device cuda:0 --num_envs "$envs" --seed 42 \
 --teacher_checkpoint "$repo/logs/$run.launch/teacher.pt" --student_arch grid_attention_v1 \
 --iterations "$count" --save_interval 200 --lr .0003 --bptt_steps 8 --perception_run_name "$run" \
 --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05
