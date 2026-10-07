#!/usr/bin/env bash
set -euo pipefail
script="$(realpath "${BASH_SOURCE[0]}")"  # before cd: a caller-relative $0 breaks after it
root="$(cd "$(dirname "$script")/../.." && pwd)"
cd "$root"
mode="${1:-train}"
shift || true
case "$mode" in
  smoke) envs=98; horizon=16; updates=3; verify=1 ;;
  benchmark) envs=4096; horizon=100; updates=8; verify=1 ;;
  train) envs=4096; horizon=100; updates=30000; verify=100 ;;
  *) echo 'Usage: train_teacher_3gpu.sh smoke|benchmark|train [overrides]'; exit 2 ;;
esac
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${mode}}"
export GAST_VERIFY_SYNC_EVERY="$verify"
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast/online_top5.json"
mkdir -p "$root/logs/usd_tmp" "$root/logs/launches/$GAST_RUN_ID"
git -C "$root" rev-parse HEAD > "$root/logs/launches/$GAST_RUN_ID/commit.txt"
git -C "$root" diff --binary > "$root/logs/launches/$GAST_RUN_ID/tracked_changes.patch"
cp "$GD_LAB_TOP5_CRITERIA_FILE" "$root/logs/launches/$GAST_RUN_ID/online_top5.json"
cp "$script" "$root/logs/launches/$GAST_RUN_ID/launcher.sh"
tar --exclude='__pycache__' -czf "$root/logs/launches/$GAST_RUN_ID/source.tar.gz" -C "$root" src scripts configs
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
 /data/users/bsseo/gd_lab_isaaclab.sif \
 /data/users/bsseo/venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
 scripts/gast/train_teacher.py --headless --distributed --total_envs "$envs" --seed 42 \
 --logger tensorboard --experiment_name gast/arm4 --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 \
 agent.algorithm.num_learning_epochs=5 agent.save_interval=1000 agent.top5_min_spacing=1 "$@"
