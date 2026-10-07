#!/usr/bin/env bash
# Multi-GPU (torchrun) v2 verification: Clean-arm teacher -> Clean + gap foothold margin / slot probing
# + stair hip-handle disturbance (Gd-VrlGapFinetuneCleanV2Raycast-Rbq10-Dreamwaq-v0), ARM4, vendor_new.
#   experiments/bivt/run_bivt_gap_v2_ddp.sh <clean_checkpoint.pt> <sha256> <smoke|train> [seed]
# Environment: GAP_GPUS (default 0,1,2 -- never GPU 3), GAP_TOTAL_ENVS (default 4096; smoke 64 per GPU),
#   GAP_V2_UPDATES (default 5000 additional updates), GAP_SMOKE_UPDATES (default 3),
#   GAP_GATE_WAIVER (train: who approved and why; recorded in params/gap_finetune_manifest.yaml),
#   GAP_V2_ARM (CleanV2 | CleanV21; default CleanV2), GAP_V21_SPEED_RAMP_STEPS (v2.1 speed ceiling ramp, default 300000;
#   resuming a v2.1 checkpoint: 0).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint="$(realpath "${1:?Clean-arm checkpoint}")"
sha="${2:?sha256 of the checkpoint}"
mode=${3:?mode: smoke|train}
seed=${4:-42}
test -s "$checkpoint"
[[ "$(sha256sum "$checkpoint" | cut -d' ' -f1)" == "$sha" ]] || { echo "checkpoint sha256 mismatch" >&2; exit 2; }
gpus=${GAP_GPUS:-0,1,2}
case ",$gpus," in *,3,*) echo "GPU 3 belongs to VLLM; refusing" >&2; exit 2 ;; esac
nproc=$(( $(tr -cd , <<<"$gpus" | wc -c) + 1 ))
sif="${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}"
python="${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}"
iter=$(apptainer exec "$sif" "$python" -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['iter']))" "$checkpoint")
case "$mode" in
 smoke) total=${GAP_TOTAL_ENVS:-$((nproc * 64))}; target=$((iter + 1 + ${GAP_SMOKE_UPDATES:-3}))
        waiver="v2 distributed smoke only: ${GAP_SMOKE_UPDATES:-3} updates to verify DDP, resume and the new terms" ;;
 train) total=${GAP_TOTAL_ENVS:-4096}; target=$((iter + ${GAP_V2_UPDATES:-5000}))
        waiver="${GAP_GATE_WAIVER:?train needs GAP_GATE_WAIVER (who approved and why)}" ;;
 *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
cd "$root"
arm_tag=$(tr 'A-Z' 'a-z' <<<"${GAP_V2_ARM:-CleanV2}")
run_id="$(date +%Y-%m-%d_%H-%M-%S)_gap_${arm_tag}_from${iter}_ddp${nproc}_s${seed}_${mode}"
export PYTHONPATH="$root/src" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpus" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/online_top5.json"
# Force ramp (env steps of THIS process): smoke = full force at once; resuming a v2 checkpoint: GAP_V2_FORCE_RAMP_STEPS=0.
if [ "$mode" = smoke ]; then export GD_LAB_V2_FORCE_RAMP_STEPS=0 GD_LAB_V21_SPEED_RAMP_STEPS=0
else export GD_LAB_V2_FORCE_RAMP_STEPS="${GAP_V2_FORCE_RAMP_STEPS:-150000}" GD_LAB_V21_SPEED_RAMP_STEPS="${GAP_V21_SPEED_RAMP_STEPS:-300000}"; fi
export GD_LAB_STOP_FILE="$root/logs/${run_id}.stop"
full_patch() { git diff --binary HEAD; for f in $(git ls-files --others --exclude-standard); do git diff --no-index --binary /dev/null "$f" || true; done; }
export GD_LAB_SOURCE_COMMIT="$(git rev-parse HEAD)"
export GD_LAB_PATCH_SHA256="$(full_patch | sha256sum | cut -d' ' -f1)"
mkdir -p "$root/logs/usd_tmp/$run_id" "$root/logs/launches/$run_id"
git rev-parse HEAD > "$root/logs/launches/$run_id/commit.txt"
full_patch > "$root/logs/launches/$run_id/changes.patch"
sha256sum "$checkpoint" > "$root/logs/launches/$run_id/resume.sha256"
tar --exclude='__pycache__' -czf "$root/logs/launches/$run_id/source.tar.gz" src scripts configs
printf 'Gap v2 DDP run=%s mode=%s from_iter=%s gpus=%s total_envs=%s target=%s stop_file=%s\n' \
 "$run_id" "$mode" "$iter" "$gpus" "$total" "$target" "$GD_LAB_STOP_FILE"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp/$run_id:/tmp/IsaacLab" "$sif" \
 "$python" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$nproc" scripts/train_bivt.py \
 --headless --distributed --seed "$seed" --logger tensorboard \
 --task "Gd-VrlGapFinetune${GAP_V2_ARM:-CleanV2}Raycast-Rbq10-Dreamwaq-v0" --total_envs "$total" \
 --resume_checkpoint "$checkpoint" --resume_sha256 "$sha" --force_ppo_lr 1e-4 --target_iterations "$target" \
 --baseline_gate_waiver "$waiver" --run_name "$run_id" \
 agent.num_steps_per_env=100 agent.algorithm.num_learning_epochs=5 agent.algorithm.num_mini_batches=4 \
 agent.algorithm.schedule=fixed agent.algorithm.min_learning_rate=0.00003 \
 agent.save_interval=100 agent.top5_min_spacing=100
