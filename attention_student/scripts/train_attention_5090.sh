#!/usr/bin/env bash
# Local single-GPU entry point. Does not modify the multi-GPU launcher or deployment.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
captures=${1:-20000}
envs=${2:-64}
run=${3:-arm4_teacher6987_attention_$(date +%Y%m%d_%H%M%S)}
[[ $captures =~ ^[1-9][0-9]*$ && $envs =~ ^[1-9][0-9]*$ && $run =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
teacher_dir="$repo_root/checkpoints/arm4_teacher_top5_resume4800_20260930"
(cd "$teacher_dir" && sha256sum -c SHA256SUMS.txt)
python3 - "$teacher_dir" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
b = json.loads((p / 'leaderboard.json').read_text())['entries']
assert b[0]['iteration'] == 6987 and b[0]['score'] == max(e['score'] for e in b)
PY
mkdir "logs/$run.launch"
cp "$teacher_dir/6987_top1.pt" "logs/$run.launch/teacher.pt"
cp "$teacher_dir/leaderboard.json" "logs/$run.launch/leaderboard.json"
cp -r "$teacher_dir/params" "logs/$run.launch/teacher_params"
mkdir "logs/$run.launch/source"
cp src/gd_lab/rl/perception_attention.py src/gd_lab/rl/attention_distillation.py \
  scripts/train_perception.py scripts/train_attention_5090.sh "logs/$run.launch/source/"
git diff > "logs/$run.launch/source.diff"
git rev-parse HEAD > "logs/$run.launch/git_sha"
mkdir -p "logs/usd_tmp/$run"
export PYTHONPATH="$repo_root/src"
export CUDA_VISIBLE_DEVICES=0 TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
exec apptainer exec --nv --writable-tmpfs --bind "$repo_root/logs/usd_tmp/$run:/tmp/IsaacLab" \
  /home/user/workspace/gd_lab_isaaclab.sif /home/user/workspace/venv_apptainer/bin/python \
  scripts/train_perception.py --headless --device cuda:0 --num_envs "$envs" --seed 42 \
  --teacher_checkpoint "$repo_root/logs/$run.launch/teacher.pt" \
  --student_arch grid_attention_v1 --iterations "$captures" --save_interval 200 \
  --lr 0.0003 --bptt_steps 8 --perception_run_name "$run" \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob 0.05 \
  "${@:4}"
