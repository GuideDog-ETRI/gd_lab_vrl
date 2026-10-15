#!/usr/bin/env bash
# Resume the GAST v2.1 teacher run 2026-10-07_11-02-05_gast_gapclean_from_bivt_train from model_8000.pt to 30000
# (crashed at 8016 on a non-finite PPO step; ppo.py now guards steps). GPUs 0,1,2 only (GPU 3 = VLLM).
# Same task/lrs/agent overrides as the original launcher; --max_iterations is additive (22000 -> 30000).
# Ramps keyed on the step counter restart on resume, so they are disabled (already finished before 8000).
set -euo pipefail
root="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"; cd "$root"
run=2026-10-07_11-02-05_gast_gapclean_from_bivt_train
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)_gast_gapclean_from_bivt_train_resume8000"
export GAST_VERIFY_SYNC_EVERY=100 GD_LAB_V2_FORCE_RAMP_STEPS=0 GD_LAB_V21_SPEED_RAMP_STEPS=0
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast/online_top5_gapclean.json"
launch="$root/logs/launches/$GAST_RUN_ID"; mkdir -p "$root/logs/usd_tmp" "$launch"
git -C "$root" rev-parse HEAD > "$launch/commit.txt"; git -C "$root" diff --binary > "$launch/tracked_changes.patch"
cp "$0" "$launch/launcher.sh"; cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
echo "GAST v2.1 resume run=$GAST_RUN_ID from $run/model_8000.pt to 30000"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" /data/users/bsseo/gd_lab_isaaclab.sif \
 /data/users/bsseo/venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
 scripts/gast/train_teacher.py --headless --distributed --total_envs 4096 --seed 42 \
 --task Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0 --policy_lr 1e-4 --terrain_lr 1e-3 \
 --logger tensorboard --experiment_name gast/arm4 --max_iterations 22000 \
 agent.resume=true agent.load_run="$run" agent.load_checkpoint=model_8000.pt \
 agent.num_steps_per_env=100 agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval=100 agent.top5_min_spacing=100
