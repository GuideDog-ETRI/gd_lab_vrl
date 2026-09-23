#!/usr/bin/env bash
# Train one VRL policy across physical GPUs 0, 1, and 2.
set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[1-4]$ || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 ARM(1|2|3|4) MAX_ITERATIONS(positive integer)" >&2
    exit 2
fi

train_arm="$1"
max_iterations="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sif_path="${GD_LAB_SIF:-/data/users/bsseo/gd_lab_isaaclab.sif}"
python_path="${GD_LAB_PYTHON:-/data/users/bsseo/venv/bin/python}"
usd_tmp="$repo_root/logs/usd_tmp/distributed"
run_name="arm${train_arm}_3gpu_$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "$sif_path" ]]; then
    echo "Isaac Lab container not found: $sif_path" >&2
    exit 1
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available" >&2
    exit 1
fi

mkdir -p "$usd_tmp"
cd "$repo_root"
echo "Starting VRL arm $train_arm on GPUs 0,1,2: 4096 environments total, $max_iterations iterations"
echo "Run name: $run_name"
exec env -u PYTHONPATH \
    CUDA_VISIBLE_DEVICES=0,1,2 TRAIN_ARM="$train_arm" OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
    apptainer exec --nv --writable-tmpfs \
    --bind "$usd_tmp:/tmp/IsaacLab" \
    "$sif_path" "$python_path" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=3 \
    scripts/train_vrl.py --distributed --task Gd-Vrl-Rbq10-Dreamwaq-v0 \
    --headless --device cuda:0 --total_envs 4096 --seed 42 \
    --max_iterations "$max_iterations" --logger tensorboard \
    --run_name "$run_name" agent.save_interval=100
