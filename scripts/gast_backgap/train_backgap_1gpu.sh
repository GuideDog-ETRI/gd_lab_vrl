#!/usr/bin/env bash
# Backward gap crossing fine-tune of the original Ray+IR 3906 on ONE GPU (local 5090).
#   scripts/gast_backgap/train_backgap_1gpu.sh <smoke|train> [gpu=0] [extra hydra overrides]
# - Start: 3906_top1 (f3e7afe5...) via --warm_start_rayir: same weights, new PPO, CENet optimizer and terrain levels restored.
# - Task Gd-GastRayIrV21Gap50Back-Rbq10-Dreamwaq-v0 = 3906 task + reversed gap commands (p_back .4 -> .7, |v| <= 1.0 m/s)
#   + direction-aware hop penalties. Same gap widths/height offsets and curriculum as forward.
# - train: 1024 envs x 100 steps x 3000 updates, seed 42, save 100, online Top-5 every update (spacing 1) from 200.
set -euo pipefail
script="$(realpath "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "$script")/../.." && pwd)"
mode="${1:?usage: train_backgap_1gpu.sh smoke|train [gpu] [overrides]}"
gpu="${2:-0}"
shift $(( $# >= 2 ? 2 : 1 ))
case "$mode" in
  smoke) envs=64; horizon=16; updates=5; save=5 ;;
  train) envs=1024; horizon=100; updates=3000; save=100 ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
teacher="${BACKGAP_SOURCE:-/home/user/gd_project/archive/gd_lab_vrl_worktrees/gast_nav_v1/checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt}"
sha=f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98
test -s "$teacher" || { echo "missing 3906: $teacher" >&2; exit 2; }
cd "$root"
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_MATH_SDPA=${GAST_MATH_SDPA:-1} GAST_NAN_DIAG=0
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_backgap_3906_${envs}x${horizon}_${updates}_${mode}}"
export GAST_VERIFY_SYNC_EVERY=100
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast_backgap/online_top5.json"
launch="$root/logs/launches/$GAST_RUN_ID"
mkdir -p "$root/logs/usd_tmp" "$launch"
sha256sum "$teacher" > "$launch/source_teacher.sha256"
cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
cp "$script" "$launch/launcher.sh"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" -C "$root" src scripts configs
printf 'backgap run=%s mode=%s gpu=%s envs=%s horizon=%s updates=%s\n' "$GAST_RUN_ID" "$mode" "$gpu" "$envs" "$horizon" "$updates"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
 "${GD_LAB_SIF:-/home/user/workspace/gd_lab_isaaclab.sif}" \
 "${GD_LAB_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}" -m torch.distributed.run --standalone --nnodes=1 \
 --nproc_per_node=1 scripts/gast_backgap/train_backgap.py --headless --distributed --total_envs "$envs" --seed 42 \
 --task Gd-GastRayIrV21Gap50Back-Rbq10-Dreamwaq-v0 --warm_start_rayir "$teacher" --warm_start_sha256 "$sha" \
 --policy_lr 1e-4 --terrain_lr 3e-4 \
 --logger tensorboard --experiment_name gast/backgap --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval="$save" agent.top5_min_spacing=1 "$@"
