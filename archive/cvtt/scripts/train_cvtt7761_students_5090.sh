#!/usr/bin/env bash
# Frozen CVTT-7761; identical capture/runtime settings, architecture-specific losses.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
method=${1:?Usage: train_cvtt7761_students_5090.sh rvld\|gavd [captures] [envs] [run]}
case "$method" in rvld) arch=cnn_gru ;; gavd) arch=grid_attention_v1 ;; *) exit 2 ;; esac
captures=${2:-20000}
resume_args=()
if [[ -n "${STUDENT_RESUME:-}" ]]; then
    test -f "$STUDENT_RESUME"
    resume_args=(--student_resume "$STUDENT_RESUME")
fi
envs=${3:-64}
run=${4:-cvtt7761_${method}_20k_$(date +%Y%m%d_%H%M%S)}
[[ $captures =~ ^[1-9][0-9]*$ && $envs =~ ^[1-9][0-9]*$ && $run =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
teacher_dir="$repo_root/checkpoints/teachers/cvtt/arm4_7761"
python3 - "$teacher_dir" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); meta=json.loads((p/'source_7761.json').read_text())
assert meta['iteration']==7761
for f,key in [('7761_top1.pt','packaged_checkpoint_sha256'),('params/agent.yaml','agent_yaml_sha256'),('params/env.yaml','env_yaml_sha256')]:
    assert hashlib.sha256((p/f).read_bytes()).hexdigest()==meta[key], f
print('CVTT-7761 checkpoint and source params hashes verified',flush=True)
PY
mkdir "logs/$run.launch"
cp "$teacher_dir/7761_top1.pt" "logs/$run.launch/teacher.pt"
cp "$teacher_dir/source_7761.json" "$teacher_dir/leaderboard_7761.json" "logs/$run.launch/"
cp -r "$teacher_dir/params" "logs/$run.launch/teacher_params"
cp -r src/gd_lab/students "logs/$run.launch/student_source"
cp scripts/distill_student.py "$0" "logs/$run.launch/"
git diff > "logs/$run.launch/source.diff"
git rev-parse HEAD > "logs/$run.launch/git_sha"
mkdir -p "logs/usd_tmp/$run"
export PYTHONPATH="$repo_root/src"
export CUDA_VISIBLE_DEVICES=0 TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
exec apptainer exec --nv --writable-tmpfs --bind "$repo_root/logs/usd_tmp/$run:/tmp/IsaacLab" \
  /home/user/workspace/gd_lab_isaaclab.sif /home/user/workspace/venv_apptainer/bin/python \
  scripts/distill_student.py --headless --device cuda:0 --num_envs "$envs" --seed 42 \
  --teacher_checkpoint "$repo_root/logs/$run.launch/teacher.pt" \
  --student_arch "$arch" --iterations "$captures" --save_interval 200 "${resume_args[@]}" \
  --lr 0.0003 --bptt_steps 8 --perception_run_name "$run" \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob 0.05
