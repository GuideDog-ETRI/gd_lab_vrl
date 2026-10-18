#!/usr/bin/env bash
# Prepared only: runs start after the two reviews. No automatic launch.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
arm="${1:?usage: train_mapper_aug149_1gpu.sh A2769|C2769|A3906 smoke|train}"
mode="${2:?missing smoke|train}"
if [ "$#" -ne 2 ]; then echo 'No untracked overrides: edit/review the launcher for a changed protocol.' >&2; exit 2; fi
case "$mode" in
 smoke) envs=64; horizon=16; updates=5; save=5; ramp=32 ;;
 train) envs=1024; horizon=100; updates=2000; save=250; ramp=50000 ;;
 *) exit 2 ;;
esac
case "$arm" in
 A2769|C2769)
  checkpoint=2769_from2000/best_top5/2769_top1.pt
  sha=65da772b8b793caf5e210f7db778c17d006999010ce964b2fc2af2deadad20c9
  task=Gd-GastRayIrV21-Rbq10-Dreamwaq-v0
  if [ "$arm" = A2769 ]; then gpu=0; aug=true; else gpu=1; aug=false; fi ;;
 A3906)
  checkpoint=3906_gap50_from_dwb38000/best_top5/3906_top1.pt
  sha=f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98
  task=Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0; gpu=0; aug=true ;;
 *) exit 2 ;;
esac
# Arm mapping is fixed: A2769/C2769 on onvlm2 GPU0/1; A3906 on 90.111 GPU0.
# Never expose GPU3. Host placement remains the launcher's caller responsibility.
export CUDA_VISIBLE_DEVICES="$gpu" TRAIN_ARM=4 PYTHONPATH="$root/src" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMNI_KIT_ACCEPT_EULA=YES
export GAST_MATH_SDPA=1 GAST_NAN_DIAG=0 GAST_VERIFY_SYNC_EVERY=100
teacher="$root/checkpoints/teachers/gast/rayir_top5_20261016/$checkpoint"
test -s "$teacher"
actual="$(sha256sum "$teacher")"; actual="${actual%% *}"
if [ "$actual" != "$sha" ]; then echo 'Checkpoint hash mismatch' >&2; exit 2; fi
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_aug149_${arm}_${mode}}"
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast/online_top5_rayir.json"
if [ "${AUG149_DRY_RUN:-0}" != 1 ]; then
launch="$root/logs/launches/$GAST_RUN_ID"; mkdir -p "$launch" "$root/logs/usd_tmp"
cp "$0" "$launch/launcher.sh"; cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
sha256sum "$teacher" > "$launch/source.sha256"
tar --exclude=__pycache__ -czf "$launch/source.tar.gz" -C "$root" src scripts configs
fi
cd "$root"
command=(apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
 "${GD_LAB_SIF:?set GD_LAB_SIF for this server}" "${GD_LAB_PYTHON:?set GD_LAB_PYTHON for this server}" \
 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
 scripts/gast/train_teacher.py --headless --distributed --total_envs "$envs" --seed 42 \
 --task "$task" --resume_rayir149 "$teacher" --warm_start_sha256 "$sha" \
 --policy_lr 1e-4 --terrain_lr 1e-3 --logger tensorboard --experiment_name gast/arm4 \
 --max_iterations "$updates" agent.num_steps_per_env="$horizon" \
 agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval="$save" agent.top5_min_spacing=500 \
 env.mapper_aug_enabled="$aug" env.mapper_aug_ramp_steps="$ramp" \
 env.mapper_aug_clean_fraction=0.25 env.mapper_aug_seed=14942)
if [ "${AUG149_DRY_RUN:-0}" = 1 ]; then
 printf "arm=%s gpu=%s augmentation=%s\n" "$arm" "$CUDA_VISIBLE_DEVICES" "$aug"
 printf "%q " "${command[@]}"; printf "\n"
else
 exec "${command[@]}"
fi
