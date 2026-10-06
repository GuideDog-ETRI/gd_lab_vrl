#!/usr/bin/env bash
# GAST teacher continued from a BIVT-Ray Clean-gap teacher (multi-GPU server, GPUs 0,1,2; never GPU 3/VLLM).
#   gast/scripts/train_teacher_gapclean_from_bivt_3gpu.sh <smoke|train> [bivt_teacher.pt] [extra hydra overrides]
# - Backbone (actor, critic, normalizers, action std, CENet + its optimizer state) copied from the BIVT-Ray teacher;
#   GAST temporal terrain encoder/decoder fresh, latent head zero (gd_lab.gast.warm_start).
# - PPO: fixed schedule, backbone lr 1e-4 (as the BIVT-Ray Clean gap run), terrain encoder/decoder lr 1e-3.
# - Task Gd-GastGapClean-Rbq10-Dreamwaq-v0: GAST teacher + Clean gap rewards (intrusion -3, clean +1.5), ARM4.
# - 4096 envs total, 30000 updates from iteration 0 (train); Top-5 every >=100 iterations, G = clean crossing rate.
# Default teacher: the pushed 21068 package. Use the BIVT-Ray run's final Top-1 once it has finished, e.g.
#   .../logs/vision_rbq10_dreamwaq/arm_4/<run>/best_top5/<iter>_top1.pt
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
mode="${1:?usage: train_teacher_gapclean_from_bivt_3gpu.sh smoke|train [bivt_teacher.pt] [overrides]}"
teacher="$(realpath "${2:-$root/../checkpoints/teachers/bivt/ray_gap_clean_vendor_new_top1_21068_20261005/teacher/21068_top1.pt}")"
shift $(( $# >= 2 ? 2 : 1 ))
test -s "$teacher" || { echo "missing BIVT-Ray teacher: $teacher" >&2; exit 2; }
case "$mode" in
  smoke) envs=98; horizon=16; updates=3; verify=1 ;;
  train) envs=4096; horizon=100; updates=30000; verify=100 ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
gpus="${GAST_GPUS:-0,1,2}"
case ",$gpus," in *,3,*) echo "GPU 3 belongs to VLLM; refusing" >&2; exit 2 ;; esac
nproc=$(( $(tr -cd , <<<"$gpus" | wc -c) + 1 ))
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$gpus" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_gast_gapclean_from_bivt_${mode}}"
export GAST_VERIFY_SYNC_EVERY="$verify"
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/online_top5_gapclean.json"
launch="$root/logs/launches/$GAST_RUN_ID"
mkdir -p "$root/logs/usd_tmp" "$launch"
git -C "$root" rev-parse HEAD > "$launch/commit.txt"
git -C "$root" diff --binary > "$launch/tracked_changes.patch"
sha256sum "$teacher" > "$launch/bivt_teacher.sha256"
cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
cp "$0" "$launch/launcher.sh"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" -C "$root" src scripts configs
printf 'GAST gap-clean warm start run=%s mode=%s gpus=%s total_envs=%s updates=%s teacher=%s\n' \
  "$GAST_RUN_ID" "$mode" "$gpus" "$envs" "$updates" "$teacher"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" \
 "${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}" \
 "${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}" -m torch.distributed.run --standalone --nnodes=1 \
 --nproc_per_node="$nproc" scripts/train_teacher.py --headless --distributed --total_envs "$envs" --seed 42 \
 --task Gd-GastGapClean-Rbq10-Dreamwaq-v0 --warm_start_bivt "$teacher" --policy_lr 1e-4 --terrain_lr 1e-3 \
 --logger tensorboard --experiment_name gast/arm4 --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval=100 agent.top5_min_spacing=100 "$@"
