#!/usr/bin/env bash
# Distill frozen Arm4 teacher 3700 in a detached tmux session.
set -euo pipefail

worker=0
if [[ ${1:-} == --worker ]]; then
    worker=1
    shift
fi
gpu_index=${1:-0}
iterations=${2:-20000}
num_envs=${3:-64}
if [[ ! $gpu_index =~ ^[0-9]+$ || ! $iterations =~ ^[1-9][0-9]*$ || ! $num_envs =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [GPU_INDEX=0] [CAMERA_CAPTURES=20000] [NUM_ENVS=64]" >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
session_name=vrl_arm4_student3700
teacher_run=2026-09-26_17-03-13_arm4_3gpu_top5_resume_model3000
teacher_path="$repo_root/logs/vision_rbq10_dreamwaq/arm_4/$teacher_run/model_3700.pt"
sif_path=${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}
python_path=${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}
[[ -f $teacher_path && -f $sif_path ]] || { echo "Teacher checkpoint or container missing" >&2; exit 1; }

if (( ! worker )); then
    if (( $# > 3 )); then echo "Too many arguments" >&2; exit 2; fi
    command -v tmux >/dev/null
    command -v apptainer >/dev/null
    if tmux has-session -t "=$session_name" 2>/dev/null; then
        echo "Already running. Attach: tmux attach -t $session_name" >&2
        exit 1
    fi
    if pgrep -f 'scripts/train_perception.py' >/dev/null; then
        echo "A student distillation process already exists; refusing a duplicate." >&2
        exit 1
    fi
    run_name="arm4_teacher3700_student_$(date +%Y%m%d_%H%M%S)"
    mkdir -p logs
    printf -v command '%q ' /bin/bash "$repo_root/scripts/start_arm4_student_tmux.sh" \
        --worker "$gpu_index" "$iterations" "$num_envs" "$run_name"
    tmux new-session -d -s "$session_name" -c "$repo_root" "$command"
    printf '%s\n' "$run_name" > logs/arm4_student3700_latest_run.txt
    echo "Started: $session_name"
    echo "Console: $repo_root/logs/$run_name.console.log"
    echo "Checkpoints: $repo_root/logs/vision_rbq10_dreamwaq/arm_4/$run_name"
    echo "Attach: tmux attach -t $session_name"
    exit 0
fi

run_name=${4:?Missing worker run name}
[[ $run_name =~ ^arm4_teacher3700_student_[0-9_]+$ ]] || exit 2
usd_tmp="$repo_root/logs/usd_tmp/$run_name"
mkdir -p "$usd_tmp"
set +e
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu_index" TRAIN_ARM=4 \
    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    apptainer exec --nv --writable-tmpfs --bind "$usd_tmp:/tmp/IsaacLab" \
    "$sif_path" "$python_path" scripts/train_perception.py \
    --headless --device cuda:0 --num_envs "$num_envs" --seed 42 \
    --load_run "$teacher_run" --checkpoint model_3700.pt \
    --iterations "$iterations" --save_interval 200 --bptt_steps 8 \
    --camera_interval_ms 70 100 --camera_delay_ms 0 50 --camera_drop_prob 0.05 \
    --perception_run_name "$run_name" 2>&1 | tee "$repo_root/logs/$run_name.console.log"
result=${PIPESTATUS[0]}
printf '%s exit=%s\n' "$(date -Is)" "$result" | tee "$repo_root/logs/$run_name.exit"
exit "$result"
