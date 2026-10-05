#!/usr/bin/env bash
# Prepare/run a GAVD student from the fixed BIVT-Ray-17206 teacher.
# This script is intentionally opt-in; creating it does not launch training.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

iterations=${1:-20000}
envs=${2:-64}
run=${3:-bivt_ray17206_gavd_20k_$(date +%Y%m%d_%H%M%S)}
bptt=${4:-16}
teacher="$repo/checkpoints/teachers/bivt/ray_top1_17206_20261003/teacher/17206_top1.pt"
expected_sha="a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d"
launch_dir="$repo/logs/${run}.launch"
output_dir="$repo/logs/vision_rbq10_dreamwaq/arm_4/${run}"

[[ $iterations =~ ^[1-9][0-9]*$ && $envs =~ ^[1-9][0-9]*$ && $bptt =~ ^[1-9][0-9]*$ && $run =~ ^[a-zA-Z0-9_-]+$ ]] || {
  echo "Usage: $0 [iterations=20000] [envs=64] [run_name] [bptt=16]" >&2
  exit 2
}
[[ -f "$teacher" ]] || { echo "Missing fixed teacher: $teacher" >&2; exit 1; }
actual_sha="$(sha256sum "$teacher" | awk '{print $1}')"
[[ $actual_sha == "$expected_sha" ]] || {
  echo "Teacher SHA256 mismatch: expected $expected_sha, got $actual_sha" >&2
  exit 1
}
[[ ! -e "$launch_dir" && ! -e "$output_dir" ]] || {
  echo "Refusing to reuse existing output: $launch_dir or $output_dir" >&2
  exit 1
}
if ps -eo pid=,args= | awk -v self="$$" '$1 != self && /distill_student\.py/ { found=1; print > "/dev/stderr" } END { exit !found }'; then
  echo "Another distill_student.py process is active; refusing concurrent launch." >&2
  exit 1
fi

mkdir "$launch_dir"
cp "$teacher" "$launch_dir/teacher.pt"
mkdir -p "$launch_dir/params"
cp "$repo/checkpoints/teachers/bivt/ray_top1_17206_20261003/params/env.yaml" "$launch_dir/params/env.yaml"
cp "$repo/checkpoints/teachers/bivt/ray_top1_17206_20261003/params/agent.yaml" "$launch_dir/params/agent.yaml"
mkdir -p "$launch_dir/source/scripts" \
  "$launch_dir/source/src/gd_lab/students/gavd" \
  "$launch_dir/source/src/gd_lab/students/rvld"
cp scripts/distill_student.py scripts/train_bivt_ray17206_gavd.sh "$launch_dir/source/scripts/"
cp src/gd_lab/students/alignment.py "$launch_dir/source/src/gd_lab/students/"
cp src/gd_lab/students/gavd/model.py src/gd_lab/students/gavd/distillation.py \
  "$launch_dir/source/src/gd_lab/students/gavd/"
cp src/gd_lab/students/rvld/model.py src/gd_lab/students/rvld/distillation.py \
  "$launch_dir/source/src/gd_lab/students/rvld/"
git rev-parse HEAD > "$launch_dir/git_sha"
printf '%s  %s\n' "$expected_sha" "teacher.pt" > "$launch_dir/teacher.sha256"
printf 'architecture=grid_attention_v1\niterations=%s\nenvs=%s\nbptt=%s\nseed=42\nlr=0.0003\nsave_interval=1000\ncamera_interval_ms=70,100\ncamera_delay_ms=0,150\ncamera_drop_prob=0.05\n' \
  "$iterations" "$envs" "$bptt" > "$launch_dir/settings.txt"
mkdir -p "$repo/logs/usd_tmp/$run"

export PYTHONPATH="$repo/src"
export CUDA_VISIBLE_DEVICES="${GAVD_GPU:-0}" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
exec apptainer exec --nv --writable-tmpfs \
  --bind "$repo/logs/usd_tmp/$run:/tmp/IsaacLab" \
  "${GAVD_IMAGE:-/home/user/workspace/gd_lab_isaaclab.sif}" "${GAVD_CONTAINER_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}" \
  scripts/distill_student.py --task Gd-VrlRayStudent-Rbq10-Dreamwaq-Vision-v0 \
  --headless --device cuda:0 --num_envs "$envs" --seed 42 \
  --teacher_checkpoint "$launch_dir/teacher.pt" --student_arch grid_attention_v1 \
  --iterations "$iterations" --save_interval 1000 --lr 0.0003 \
  --bptt_steps "$bptt" --perception_run_name "$run" \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob 0.05
