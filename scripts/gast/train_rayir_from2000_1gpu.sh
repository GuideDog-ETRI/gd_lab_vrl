#!/usr/bin/env bash
# UPDATE134B: GAST v2.1 teacher 2000 continued as the Ray+IR teacher on ONE GPU (multi-GPU server; never GPU 3/VLLM).
#   scripts/gast/train_rayir_from2000_1gpu.sh <smoke|train> [gpu=0] [extra hydra overrides]
# - Start: model_2000.pt (sha256 db500e96...) copied whole; first terrain Conv 2->3 input channels, IR columns zero
#   (start behaviour = 2000). New PPO optimizer (policy 1e-4 / terrain 1e-3, fixed), CENet optimizer restored,
#   iteration 0, terrain levels restored from 2000.
# - Task Gd-GastRayIrV21-Rbq10-Dreamwaq-v0: v2.1 rewards/terrain/disturbance; terrain input = raycast camera-visible
#   cells (hidden = unknown) + NoisyTerrain noise; history frames [height, valid, IR] (gd_lab.gast.rayir).
# - train: 1024 envs x 100 steps x 5000 updates, seed 42, save every 250, online Top-5 from 500, spacing 500 (configs/online_top5_rayir.json; level/contact thresholds as gap-clean).
# - smoke: 64 envs x 16 steps x 5 updates.
# Ramps (v2 force 150k steps, v2.1 speed 300k, noise 200k) restart at step 0 and are left on.
set -euo pipefail
script="$(realpath "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "$script")/../.." && pwd)"
mode="${1:?usage: train_rayir_from2000_1gpu.sh smoke|train [gpu] [overrides]}"
gpu="${2:-0}"
shift $(( $# >= 2 ? 2 : 1 ))
case "$gpu" in 3) echo "GPU 3 belongs to VLLM; refusing" >&2; exit 2 ;; esac
case "$mode" in
  smoke) envs=64; horizon=16; updates=5; save=5 ;;
  train) envs=1024; horizon=100; updates=5000; save=250 ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
teacher="${GAST_SOURCE_TEACHER:-$root/checkpoints/teachers/gast/gast_v21_2000_20261007/teacher/model_2000.pt}"
sha="${GAST_SOURCE_SHA256:-db500e96ddf2325d415ac960417d2e34417ded87d804f630ae165675eaba24b4}"
test -s "$teacher" || { echo "missing GAST 2000 teacher: $teacher" >&2; exit 2; }
cd "$root"
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_MATH_SDPA=${GAST_MATH_SDPA:-1} GAST_NAN_DIAG=0
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_gast_rayir_from2000_${envs}x${horizon}_${updates}_${mode}}"
export GAST_VERIFY_SYNC_EVERY=100
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast/online_top5_rayir.json"
launch="$root/logs/launches/$GAST_RUN_ID"
mkdir -p "$root/logs/usd_tmp" "$launch"
sha256sum "$teacher" > "$launch/source_teacher.sha256"
cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
cp "$script" "$launch/launcher.sh"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" -C "$root" src scripts configs
printf 'GAST Ray+IR from 2000 run=%s mode=%s gpu=%s envs=%s horizon=%s updates=%s\n' \
  "$GAST_RUN_ID" "$mode" "$gpu" "$envs" "$horizon" "$updates"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
 "${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}" \
 "${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}" -m torch.distributed.run --standalone --nnodes=1 \
 --nproc_per_node=1 scripts/gast/train_teacher.py --headless --distributed --total_envs "$envs" --seed 42 \
 --task Gd-GastRayIrV21-Rbq10-Dreamwaq-v0 --warm_start_gast "$teacher" --warm_start_sha256 "$sha" \
 --policy_lr 1e-4 --terrain_lr 1e-3 \
 --logger tensorboard --experiment_name gast/arm4 --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval="$save" agent.top5_min_spacing=500 "$@"
