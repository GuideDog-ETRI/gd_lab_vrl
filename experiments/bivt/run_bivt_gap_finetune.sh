#!/usr/bin/env bash
# Gap fine-tuning arms for the frozen BIVT-Ray 17206 teacher (spec: claude_handoff/CLAUDE_FINAL_SPEC_V4.md).
#   experiments/bivt/run_bivt_gap_finetune.sh <baseline|intrusion|clean> <smoke|train> [checkpoint] [seed]
# One GPU, one process, 1024 env, fixed PPO LR. Never distributed (CENet is not synchronized across ranks).
#   smoke : rollout only (no PPO update, no optimizer step): checks the manager graph, logs, manifest.
#   train : refuses to start unless GAP_BASELINE_GATE is a structured, positive gate-C record (validated again in
#           scripts/train_bivt.py, gd_lab.teachers.bivt.gap_guard.validate_gate_record). Nothing in this repo writes one.
#           GAP_UPDATES (default 4000) = number of PPO updates to run on top of the 17206 checkpoint.
# Container defaults are this PC's; on the training server export GD_LAB_SIF=/data/users/bsseo/gd_lab_isaaclab.sif
# and GD_LAB_PYTHON=/data/users/bsseo/venv/bin/python.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arm=${1:?arm: baseline|intrusion|clean}
mode=${2:?mode: smoke|train}
checkpoint=${3:-$root/checkpoints/teachers/bivt/ray_top1_17206_20261003/teacher/17206_top1.pt}
seed=${4:-42}
case "$arm" in baseline) task=Baseline ;; intrusion) task=Intrusion ;; clean) task=Clean ;; *) echo "bad arm: $arm" >&2; exit 2 ;; esac
test -s "$checkpoint"
case "$mode" in
 smoke) envs=64; extra=(--rollout_only_steps "${GAP_SMOKE_STEPS:-300}") ;;
 train)
  test -s "${GAP_BASELINE_GATE:?train needs the passed baseline-evaluation record (GAP_BASELINE_GATE=/path/to/record)}"
  # The pinned checkpoint stores iter=17206 and DreamwaqRunner.load resumes at 17207, so +N updates means 17206+1+N.
  envs=1024; extra=(--baseline_gate "$GAP_BASELINE_GATE" --target_iterations "$((17206 + 1 + ${GAP_UPDATES:-4000}))") ;;
 *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
cd "$root"
run_id="$(date +%Y-%m-%d_%H-%M-%S)_gap_${arm}_s${seed}_${mode}"
export PYTHONPATH="$root/src" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${BIVT_GPU:-0}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/online_top5.json"
export GD_LAB_STOP_FILE="$root/logs/${run_id}.stop"
# The complete implementation = tracked changes + untracked new files (plain `git diff HEAD` omits the latter).
full_patch() { git diff --binary HEAD; for f in $(git ls-files --others --exclude-standard); do git diff --no-index --binary /dev/null "$f" || true; done; }
export GD_LAB_SOURCE_COMMIT="$(git rev-parse HEAD)"
export GD_LAB_PATCH_SHA256="$(full_patch | sha256sum | cut -d' ' -f1)"
mkdir -p "$root/logs/usd_tmp/$run_id" "$root/logs/launches/$run_id"
git rev-parse HEAD > "$root/logs/launches/$run_id/commit.txt"
full_patch > "$root/logs/launches/$run_id/changes.patch"
sha256sum "$checkpoint" > "$root/logs/launches/$run_id/resume.sha256"
tar --exclude='__pycache__' -czf "$root/logs/launches/$run_id/source.tar.gz" src scripts configs
printf 'Gap run=%s arm=%s mode=%s envs=%s checkpoint=%s patch_sha256=%s\n' \
 "$run_id" "$arm" "$mode" "$envs" "$checkpoint" "$GD_LAB_PATCH_SHA256"
# PPO settings below are those preserved in the 17206 run's params/agent.yaml; only the LR schedule is pinned.
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp/$run_id:/tmp/IsaacLab" \
 "${GD_LAB_SIF:-/home/user/workspace/gd_lab_isaaclab.sif}" \
 "${GD_LAB_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}" scripts/train_bivt.py \
 --headless --device cuda:0 --seed "$seed" --logger tensorboard \
 --task "Gd-VrlGapFinetune${task}Raycast-Rbq10-Dreamwaq-v0" --num_envs "$envs" \
 --resume_checkpoint "$checkpoint" --force_ppo_lr 1e-4 "${extra[@]}" --run_name "$run_id" \
 agent.num_steps_per_env=100 agent.algorithm.num_learning_epochs=5 agent.algorithm.num_mini_batches=4 \
 agent.algorithm.schedule=fixed agent.algorithm.min_learning_rate=0.00003 \
 agent.save_interval=100 agent.top5_min_spacing=100
