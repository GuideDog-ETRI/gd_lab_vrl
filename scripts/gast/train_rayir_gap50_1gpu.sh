#!/usr/bin/env bash
# UPDATE135: Ray+IR GAST teacher with 50% gap columns on ONE GPU (5090 local / 90.111).
#   scripts/gast/train_rayir_gap50_1gpu.sh <gast|dwb> <smoke|bench|train> [gpu=0] [extra hydra overrides]
# - gast: continue GAST v2.1 26123 (sha256 113c2b88...); first terrain Conv 2->3, IR columns zero; CENet optimizer
#   and terrain levels restored.
# - dwb: blind DWB-38000 (sha256 c6cfe178...) backbone; fresh terrain encoder/decoder, latent head zero,
#   actor latent columns zero; new PPO/CENet optimizers; terrain levels from the default start.
# - Task Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0: as the onvlm2 UPDATE134B run (v2.1 rewards/disturbance,
#   raycast-visible cells + NoisyTerrain + IR) but 20 terrain columns = 10 families x 1 + platform_gap x 10.
# - train: 1024 envs x 100 steps x 5000 updates, seed 42, save 250, Top-5 from 500 spacing 500.
# - bench: 1024 envs x 100 steps x 30 updates (timing). smoke: 64 envs x 16 steps x 5 updates.
set -euo pipefail
script="$(realpath "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "$script")/../.." && pwd)"
source_kind="${1:?usage: train_rayir_gap50_1gpu.sh gast|dwb smoke|bench|train [gpu] [overrides]}"
mode="${2:?usage: train_rayir_gap50_1gpu.sh gast|dwb smoke|bench|train [gpu] [overrides]}"
gpu="${3:-0}"
shift $(( $# >= 3 ? 3 : 2 ))
case "$mode" in
  smoke) envs=64; horizon=16; updates=5; save=5 ;;
  bench) envs=1024; horizon=100; updates=30; save=1000 ;;
  train) envs=1024; horizon=100; updates=5000; save=250 ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
case "$source_kind" in
  gast) teacher="${GAST_SOURCE_TEACHER:-$root/checkpoints/teachers/gast/gast_v21_top5_20261013/teacher/26123_top1.pt}"
        sha="${GAST_SOURCE_SHA256:-113c2b88a0518edfe85bdc33e3984f4f89d6412edb7773649548beaa65ed429e}"
        flag=--warm_start_gast ;;
  dwb)  teacher="${GAST_SOURCE_TEACHER:-$root/checkpoints/teachers/blind/arm4_38000/teacher/model_38000.pt}"
        sha="${GAST_SOURCE_SHA256:-c6cfe178cc069766f9543fb5fba983eb05a322d507e4302ee2495d4d26d2d46b}"
        flag=--warm_start_dwb ;;
  *) echo "bad source: $source_kind" >&2; exit 2 ;;
esac
test -s "$teacher" || { echo "missing source teacher: $teacher" >&2; exit 2; }
cd "$root"
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_MATH_SDPA=${GAST_MATH_SDPA:-1} GAST_NAN_DIAG=0
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_gast_rayir_gap50_${source_kind}_${envs}x${horizon}_${updates}_${mode}}"
export GAST_VERIFY_SYNC_EVERY=100
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast/online_top5_rayir.json"
launch="$root/logs/launches/$GAST_RUN_ID"
mkdir -p "$root/logs/usd_tmp" "$launch"
sha256sum "$teacher" > "$launch/source_teacher.sha256"
cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
cp "$script" "$launch/launcher.sh"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" -C "$root" src scripts configs
printf 'GAST Ray+IR gap50 run=%s source=%s mode=%s gpu=%s envs=%s horizon=%s updates=%s\n' \
  "$GAST_RUN_ID" "$source_kind" "$mode" "$gpu" "$envs" "$horizon" "$updates"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
 "${GD_LAB_SIF:?set GD_LAB_SIF}" \
 "${GD_LAB_PYTHON:?set GD_LAB_PYTHON}" -m torch.distributed.run --standalone --nnodes=1 \
 --nproc_per_node=1 scripts/gast/train_teacher.py --headless --distributed --total_envs "$envs" --seed 42 \
 --task Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0 "$flag" "$teacher" --warm_start_sha256 "$sha" \
 --policy_lr 1e-4 --terrain_lr 1e-3 \
 --logger tensorboard --experiment_name gast/arm4 --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval="$save" agent.top5_min_spacing=500 "$@"
