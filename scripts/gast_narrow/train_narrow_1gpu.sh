#!/usr/bin/env bash
# GAST-Narrow v1: narrow-passage fine-tune of backgap V2 update 6566 on ONE GPU.
#   scripts/gast_narrow/train_narrow_1gpu.sh <smoke|pilot|train> [gpu=0] [extra hydra overrides]
# - Start: backgapv2_6566.pt (22cf918c...) via --warm_start_rayir (train_narrow.py): 6566 weights, fresh LiDAR
#   encoder, zero new actor/critic columns, new PPO, CENet optimizer + retention terrain levels restored.
# - smoke: 64 envs x 16 steps x 5 updates. pilot: 1024 x 100 x 50 (timing). train: 1024 x 100 x NARROW_UPDATES (3000).
# - Online Top-5 every update (spacing 1) from 150, narrow score (configs/gast_narrow/online_top5.json, live-reloaded).
# - Stop with a save: touch $GD_LAB_STOP_FILE (default logs/launches/<run>/STOP).
# Paths: NARROW_SOURCE / NARROW_SOURCE_SHA256, GD_LAB_SIF, GD_LAB_PYTHON (defaults below, per host).
set -euo pipefail
script="$(realpath "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "$script")/../.." && pwd)"
mode="${1:?usage: train_narrow_1gpu.sh smoke|pilot|train [gpu] [overrides]}"
gpu="${2:-0}"
shift $(( $# >= 2 ? 2 : 1 ))
case "$mode" in
  smoke) envs=64; horizon=16; updates=5; save=5 ;;
  pilot) envs=1024; horizon=100; updates=50; save=50 ;;
  train) envs=1024; horizon=100; updates=${NARROW_UPDATES:-3000}; save=100 ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
teacher="${NARROW_SOURCE:-$root/checkpoints/gast_narrow/backgapv2_6566.pt}"
sha=${NARROW_SOURCE_SHA256:-22cf918c5029e0b6f55a51b50739b40958d41244514262a525750b088611ba78}
test -s "$teacher" || { echo "missing 6566: $teacher" >&2; exit 2; }
if [[ -z "${GD_LAB_SIF:-}" ]]; then
  for c in "$HOME/workspace/gd_lab_isaaclab.sif" /data/users/bsseo/gd_lab_isaaclab.sif; do [[ -f $c ]] && GD_LAB_SIF=$c && break; done
fi
if [[ -z "${GD_LAB_PYTHON:-}" ]]; then
  for c in "$HOME/workspace/venv_apptainer/bin/python" "$HOME/workspace/venv/bin/python"; do [[ -L $c || -e $c ]] && GD_LAB_PYTHON=$c && break; done
fi
cd "$root"
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_MATH_SDPA=${GAST_MATH_SDPA:-1} GAST_NAN_DIAG=0
# 1024 envs fill the 32 GB card: without this cuBLAS fails to get a workspace (CUBLAS_STATUS_INTERNAL_ERROR, 90.111).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# 6566 finished its ramps (backward share, speed ceiling, disturbance force): do not restart them.
export GD_LAB_BACKGAP_RAMP_STEPS=${GD_LAB_BACKGAP_RAMP_STEPS:-0}
export GD_LAB_V21_SPEED_RAMP_STEPS=${GD_LAB_V21_SPEED_RAMP_STEPS:-0} GD_LAB_V2_FORCE_RAMP_STEPS=${GD_LAB_V2_FORCE_RAMP_STEPS:-0}
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_narrow_6566_${envs}x${horizon}_${updates}_${mode}}"
export GAST_VERIFY_SYNC_EVERY=100
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast_narrow/online_top5.json"
launch="$root/logs/launches/$GAST_RUN_ID"
export GD_LAB_STOP_FILE="${GD_LAB_STOP_FILE:-$launch/STOP}"
mkdir -p "$root/logs/usd_tmp" "$launch"
sha256sum "$teacher" > "$launch/source_teacher.sha256"
cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
cp "$script" "$launch/launcher.sh"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" -C "$root" src scripts configs
git -C "$root" rev-parse HEAD > "$launch/git_head.txt" 2>/dev/null || true
printf 'narrow run=%s mode=%s gpu=%s envs=%s horizon=%s updates=%s sif=%s python=%s\n' \
  "$GAST_RUN_ID" "$mode" "$gpu" "$envs" "$horizon" "$updates" "$GD_LAB_SIF" "$GD_LAB_PYTHON"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" "$GD_LAB_SIF" \
 "$GD_LAB_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
 --nproc_per_node=1 scripts/gast_narrow/train_narrow.py --headless --distributed --total_envs "$envs" --seed 42 \
 --task Gd-GastNarrowV1-Rbq10-Dreamwaq-v0 --warm_start_rayir "$teacher" --warm_start_sha256 "$sha" \
 --policy_lr 1e-4 --terrain_lr 3e-4 \
 --logger tensorboard --experiment_name gast/narrow --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval="$save" agent.top5_min_spacing=1 "$@"
