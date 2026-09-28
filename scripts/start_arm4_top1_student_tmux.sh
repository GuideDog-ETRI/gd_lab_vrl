#!/usr/bin/env bash
# Freeze the current Arm4 Top-1 teacher, then distill in detached tmux.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
session=vrl_arm4_student_top1
if [[ ${1:-} != --worker ]]; then
    gpu=${1:-0}
    captures=${2:-20000}
    envs=${3:-64}
    [[ $gpu =~ ^[0-9]+$ && $captures =~ ^[1-9][0-9]*$ && $envs =~ ^[1-9][0-9]*$ ]] || exit 2
    if tmux has-session -t "=$session" 2>/dev/null || pgrep -f '[s]cripts/train_perception.py' >/dev/null; then
        echo 'Student already running' >&2
        exit 1
    fi
    teacher_source=logs/vision_rbq10_dreamwaq/arm_4/2026-09-27_08-04-53_arm4_3gpu_top5_gap8_resume_model3700
    # Snapshot weights and ranking metadata together so later ranking changes cannot alter this run.
    snapshot=$(/usr/bin/python3 - "$teacher_source" <<'PY'
import datetime, json, pathlib, shutil, sys
source = pathlib.Path(sys.argv[1])
board = json.loads((source / 'best_top5/leaderboard.json').read_text())
iteration = board['entries'][0]['iteration']
name = f"arm4_top1_teacher{iteration}_frozen_{datetime.datetime.now():%Y%m%d_%H%M%S}"
target = source.parent / name
target.mkdir()
shutil.copy2(source / f'best_top5/{iteration}_top1.pt', target / 'model_top1.pt')
shutil.copytree(source / 'params', target / 'params')
(target / 'leaderboard_at_freeze.json').write_text(json.dumps(board, indent=2))
print(name)
PY
    )
    run="${snapshot/_frozen_/_student_}"
    printf -v command '%q ' bash "$repo_root/scripts/start_arm4_top1_student_tmux.sh" --worker "$gpu" "$captures" "$envs" "$snapshot" "$run"
    tmux new-session -d -s "$session" -c "$repo_root" "$command"
    printf '%s\n' "$run" > logs/arm4_top1_student_latest_run.txt
    printf 'Session: %s\nTeacher: %s\nStudent: %s\n' "$session" "$snapshot" "$run"
    exit 0
fi
gpu=$2
captures=$3
envs=$4
snapshot=$5
run=$6
usd_tmp="$repo_root/logs/usd_tmp/$run"
mkdir -p "$usd_tmp"
git rev-parse HEAD > "logs/$run.git_head"
git diff > "logs/$run.git_diff"
set +e
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    apptainer exec --nv --writable-tmpfs --bind "$usd_tmp:/tmp/IsaacLab" \
    "${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}" \
    "${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}" scripts/train_perception.py \
    --headless --device cuda:0 --num_envs "$envs" --seed 42 \
    --load_run "$snapshot" --checkpoint model_top1.pt \
    --iterations "$captures" --save_interval 200 --bptt_steps 8 \
    --camera_interval_ms 70 100 --camera_delay_ms 0 50 --camera_drop_prob 0.05 \
    --perception_run_name "$run" 2>&1 | tee "logs/$run.console.log"
result=${PIPESTATUS[0]}
printf '%s exit=%s\n' "$(date -Is)" "$result" | tee "logs/$run.exit"
exit "$result"
