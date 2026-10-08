#!/usr/bin/env bash
# RVLD or GAVD camera student from a frozen GAST teacher (full-grid 8-step memory), in the teacher's v2.1 env.
#   experiments/common/distill_from_gast_teacher.sh <cnn_gru|grid_attention_v1> <teacher_package> [iterations=20000] [envs=512] [run] [bptt=16]
# The package needs teacher/model_*.pt, params/{agent,env}.yaml, SHA256SUMS.txt (checked here).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
arch=${1:?cnn_gru or grid_attention_v1}
package="$(realpath "${2:?teacher package, e.g. checkpoints/teachers/gast/gast_v21_2000_20261007}")"
iterations=${3:-20000}; envs=${4:-512}; bptt=${6:-16}
short=$([ "$arch" = cnn_gru ] && echo rvld || echo gavd)
run=${5:-$(basename "$package")_${short}_${envs}env_${iterations}_$(date +%Y%m%d_%H%M%S)}
case "$arch" in cnn_gru|grid_attention_v1) ;; *) echo "unknown arch $arch" >&2; exit 2 ;; esac
(cd "$package" && sha256sum -c --quiet SHA256SUMS.txt) || { echo 'Teacher package hash mismatch' >&2; exit 1; }
grep -qx 'camera_profile: vendor_new' "$package/params/env.yaml" || { echo 'Teacher package is not vendor_new' >&2; exit 1; }
teacher="$(ls "$package"/teacher/model_*.pt | head -1)"
launch_dir="$repo/logs/${run}.launch"
[[ ! -e "$launch_dir" ]] || { echo "Refusing to reuse $launch_dir" >&2; exit 1; }
mkdir -p "$launch_dir/params" "$repo/logs/usd_tmp/$run"
cp "$teacher" "$launch_dir/teacher.pt"
cp "$package/params/env.yaml" "$package/params/agent.yaml" "$launch_dir/params/"
git rev-parse HEAD > "$launch_dir/git_sha"
printf 'teacher_package=%s\narchitecture=%s\niterations=%s\nenvs=%s\nbptt=%s\n' "$package" "$arch" "$iterations" "$envs" "$bptt" > "$launch_dir/settings.txt"
export PYTHONPATH="$repo/src"
export CUDA_VISIBLE_DEVICES="${DISTILL_GPU:-0}" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
exec apptainer exec --nv --writable-tmpfs --bind "$repo/logs/usd_tmp/$run:/tmp/IsaacLab" \
  "${DISTILL_IMAGE:-${GD_LAB_SIF:-$HOME/workspace/gd_lab_isaaclab.sif}}" \
  "${DISTILL_PYTHON:-${GD_LAB_PYTHON:-$HOME/workspace/venv_apptainer/bin/python}}" \
  scripts/distill_student.py --task Gd-GastTeacherGastStudent-Rbq10-Dreamwaq-Vision-v0 --v21_env \
  --headless --device cuda:0 --num_envs "$envs" --seed 42 \
  --teacher_checkpoint "$launch_dir/teacher.pt" --student_arch "$arch" \
  --iterations "$iterations" --save_interval 1000 --lr 0.0003 \
  --bptt_steps "$bptt" --perception_run_name "$run" \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob 0.05
