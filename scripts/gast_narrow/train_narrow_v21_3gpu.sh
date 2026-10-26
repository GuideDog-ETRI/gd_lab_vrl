#!/usr/bin/env bash
# GAST-Narrow v2.1 from GAST-Narrow v2 update 1111 on onvlm2 GPU 0,1,2 (never GPU 3), 4096 envs in total.
#   scripts/gast_narrow/train_narrow_v21_3gpu.sh <smoke|pilot|train> [extra hydra overrides]
# - Start: v2 1111_top1.pt (34bb627e...) via --warm_start_rayir (train_narrow_v21.py): all weights copied, new PPO,
#   CENet optimizer and terrain levels restored. Terrain: platform_gap 8 columns (36). Anchor: 6566 (NARROW_ANCHOR_*).
# - smoke: 192 envs x 16 steps x 5 updates. pilot: 4096 x 100 x 30 (timing). train: 4096 x 100 x NARROW_UPDATES (700).
# - Online Top-5 every update (spacing 1) from 100, v2 score (configs/gast_narrow/online_top5_v2.json, live-reloaded).
# - Stop with a save: touch $GD_LAB_STOP_FILE (default logs/launches/<run>/STOP).
# Paths: NARROW_SOURCE / NARROW_SOURCE_SHA256, GD_LAB_SIF, GD_LAB_PYTHON (defaults below, per host).
set -euo pipefail
script="$(realpath "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "$script")/../.." && pwd)"
mode="${1:?usage: train_narrow_v2_3gpu.sh smoke|pilot|train [overrides]}"
shift
gpus="${NARROW_GPUS:-0,1,2}"
[[ ",$gpus," != *",3,"* ]] || { echo "GPU 3 is not ours on onvlm2" >&2; exit 2; }
nproc=$(awk -F, '{print NF}' <<<"$gpus")
case "$mode" in
  smoke) envs=192; horizon=16; updates=5; save=5 ;;
  pilot) envs=4096; horizon=100; updates=30; save=30 ;;
  train) envs=4096; horizon=100; updates=${NARROW_UPDATES:-700}; save=100 ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
teacher="${NARROW_SOURCE:-$root/checkpoints/gast_narrow/narrowv2_1111.pt}"
sha=${NARROW_SOURCE_SHA256:-34bb627e6c4b29c847178bd1f7a1a54f060db25c77f88d120a956183815783ea}
test -s "$teacher" || { echo "missing 1111: $teacher" >&2; exit 2; }
export NARROW_ANCHOR_CKPT="${NARROW_ANCHOR_CKPT:-$root/checkpoints/gast_narrow/backgapv2_6566.pt}"
export NARROW_ANCHOR_WEIGHT=${NARROW_ANCHOR_WEIGHT:-1.0} NARROW_ANCHOR_EPOCHS=${NARROW_ANCHOR_EPOCHS:-1}
test -s "$NARROW_ANCHOR_CKPT" || { echo "missing 6566 anchor: $NARROW_ANCHOR_CKPT" >&2; exit 2; }
if [[ -z "${GD_LAB_SIF:-}" ]]; then
  for c in /data/users/bsseo/gd_lab_isaaclab.sif "$HOME/workspace/gd_lab_isaaclab.sif"; do [[ -f $c ]] && GD_LAB_SIF=$c && break; done
fi
if [[ -z "${GD_LAB_PYTHON:-}" ]]; then
  for c in /data/users/bsseo/venv/bin/python "$HOME/workspace/venv/bin/python" "$HOME/workspace/venv_apptainer/bin/python"; do
    [[ -L $c || -e $c ]] && GD_LAB_PYTHON=$c && break
  done
fi
cd "$root"
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$gpus" APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpus" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# onvlm2 (2026-10-19): NCCL P2P/SHM transports hit illegal memory access; socket transport on loopback works.
# 2026-10-26 (after the repair): even then, small collectives (1-element all_reduce) with the default Tree
# algorithm hit an illegal memory access; 4 MB ones and NCCL_ALGO=Ring are fine (logs/probe on onvlm2).
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME="${NARROW_NCCL_IFNAME:-lo}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export GAST_MATH_SDPA=${GAST_MATH_SDPA:-1} GAST_NAN_DIAG=0
# No expandable_segments: the 96 GB cards do not need it (it was added for the 32 GB 90.111 card).
unset PYTORCH_CUDA_ALLOC_CONF
# 6566 (and so 1748) finished its ramps (backward share, speed ceiling, disturbance force): do not restart them.
export GD_LAB_BACKGAP_RAMP_STEPS=${GD_LAB_BACKGAP_RAMP_STEPS:-0}
export GD_LAB_V21_SPEED_RAMP_STEPS=${GD_LAB_V21_SPEED_RAMP_STEPS:-0} GD_LAB_V2_FORCE_RAMP_STEPS=${GD_LAB_V2_FORCE_RAMP_STEPS:-0}
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_narrow_v21_1111_${envs}x${horizon}_${updates}_${mode}}"
[[ "$GAST_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid run id' >&2; exit 2; }
export GAST_VERIFY_SYNC_EVERY=100
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast_narrow/online_top5_v2.json"
launch="$root/logs/launches/$GAST_RUN_ID"
export GD_LAB_STOP_FILE="${GD_LAB_STOP_FILE:-$launch/STOP}"
mkdir -p "$root/logs/usd_tmp" "$launch"
(set -o noclobber; printf '%s\n' "$mode" > "$launch/LAUNCH_STARTED")
sha256sum "$teacher" > "$launch/source_teacher.sha256"
cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
cp "$script" "$launch/launcher.sh"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" -C "$root" src scripts configs
cp "$root/GIT_COMMIT" "$launch/git_head.txt" 2>/dev/null || git -C "$root" rev-parse HEAD > "$launch/git_head.txt" 2>/dev/null || true
printf 'narrow v2.1 run=%s mode=%s gpus=%s nproc=%s envs=%s horizon=%s updates=%s sif=%s python=%s\n' \
  "$GAST_RUN_ID" "$mode" "$gpus" "$nproc" "$envs" "$horizon" "$updates" "$GD_LAB_SIF" "$GD_LAB_PYTHON"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp:/tmp/IsaacLab" "$GD_LAB_SIF" \
 "$GD_LAB_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
 --nproc_per_node="$nproc" scripts/gast_narrow/train_narrow_v21.py --headless --distributed --total_envs "$envs" --seed 42 \
 --task Gd-GastNarrowV21-Rbq10-Dreamwaq-v0 --warm_start_rayir "$teacher" --warm_start_sha256 "$sha" \
 --policy_lr 1e-4 --terrain_lr 3e-4 \
 --logger tensorboard --experiment_name gast/narrow_v21 --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval="$save" agent.top5_min_spacing=1 "$@"
