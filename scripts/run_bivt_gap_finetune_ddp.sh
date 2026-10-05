#!/usr/bin/env bash
# Multi-GPU (torchrun) gap fine-tuning of the BIVT-Ray 17206 teacher, ARM4, RBQ SDK cameras (vendor_new).
#   scripts/run_bivt_gap_finetune_ddp.sh <baseline|intrusion|clean> <smoke|train> [seed]
# Environment: GAP_GPUS (default 0,1,2 -- never the VLLM GPU 3), GAP_TOTAL_ENVS (default 4096, split across ranks),
#   GAP_TARGET_ITERATIONS (default 30000 = total completed updates; 17206 resumes at 17207),
#   GAP_SMOKE_UPDATES (default 3), and for train either GAP_BASELINE_GATE (gate-C record) or
#   GAP_GATE_WAIVER (who waived gate C and why; recorded in params/gap_finetune_manifest.yaml).
#   smoke: real distributed PPO updates on top of 17206 (target 17207+GAP_SMOKE_UPDATES) with a smoke waiver.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant=${1:?reward variant: baseline|intrusion|clean}
mode=${2:?mode: smoke|train}
seed=${3:-42}
case "$variant" in baseline) task=Baseline ;; intrusion) task=Intrusion ;; clean) task=Clean ;; *) echo "bad variant: $variant" >&2; exit 2 ;; esac
checkpoint="$root/checkpoints/teachers/bivt/ray_top1_17206_20261003/teacher/17206_top1.pt"
test -s "$checkpoint"
gpus=${GAP_GPUS:-0,1,2}
case ",$gpus," in *,3,*) echo "GPU 3 belongs to VLLM; refusing" >&2; exit 2 ;; esac
nproc=$(( $(tr -cd , <<<"$gpus" | wc -c) + 1 ))
case "$mode" in
 smoke) total=${GAP_TOTAL_ENVS:-$((nproc * 64))}; target=$((17206 + 1 + ${GAP_SMOKE_UPDATES:-3}))
        gate=(--baseline_gate_waiver "distributed smoke only: ${GAP_SMOKE_UPDATES:-3} updates to verify DDP, resume and camera geometry") ;;
 train) total=${GAP_TOTAL_ENVS:-4096}; target=${GAP_TARGET_ITERATIONS:-30000}
        if [ -n "${GAP_BASELINE_GATE:-}" ]; then gate=(--baseline_gate "$GAP_BASELINE_GATE")
        else gate=(--baseline_gate_waiver "${GAP_GATE_WAIVER:?train needs GAP_BASELINE_GATE or GAP_GATE_WAIVER}"); fi ;;
 *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
cd "$root"
run_id="$(date +%Y-%m-%d_%H-%M-%S)_gap_${variant}_ddp${nproc}_s${seed}_${mode}"
export PYTHONPATH="$root/src" TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$gpus" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/online_top5.json"
export GD_LAB_STOP_FILE="$root/logs/${run_id}.stop"
full_patch() { git diff --binary HEAD; for f in $(git ls-files --others --exclude-standard); do git diff --no-index --binary /dev/null "$f" || true; done; }
export GD_LAB_SOURCE_COMMIT="$(git rev-parse HEAD)"
export GD_LAB_PATCH_SHA256="$(full_patch | sha256sum | cut -d' ' -f1)"
mkdir -p "$root/logs/usd_tmp/$run_id" "$root/logs/launches/$run_id"
git rev-parse HEAD > "$root/logs/launches/$run_id/commit.txt"
full_patch > "$root/logs/launches/$run_id/changes.patch"
sha256sum "$checkpoint" > "$root/logs/launches/$run_id/resume.sha256"
tar --exclude='__pycache__' -czf "$root/logs/launches/$run_id/source.tar.gz" src scripts configs
printf 'Gap DDP run=%s variant=%s mode=%s gpus=%s total_envs=%s target=%s stop_file=%s\n' \
 "$run_id" "$variant" "$mode" "$gpus" "$total" "$target" "$GD_LAB_STOP_FILE"
exec apptainer exec --nv --writable-tmpfs --bind "$root/logs/usd_tmp/$run_id:/tmp/IsaacLab" \
 "${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}" \
 "${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}" -m torch.distributed.run --standalone --nnodes=1 \
 --nproc_per_node="$nproc" scripts/train_bivt.py \
 --headless --distributed --seed "$seed" --logger tensorboard \
 --task "Gd-VrlGapFinetune${task}Raycast-Rbq10-Dreamwaq-v0" --total_envs "$total" \
 --resume_checkpoint "$checkpoint" --force_ppo_lr 1e-4 --target_iterations "$target" "${gate[@]}" --run_name "$run_id" \
 agent.num_steps_per_env=100 agent.algorithm.num_learning_epochs=5 agent.algorithm.num_mini_batches=4 \
 agent.algorithm.schedule=fixed agent.algorithm.min_learning_rate=0.00003 \
 agent.save_interval=100 agent.top5_min_spacing=100
