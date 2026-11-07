#!/bin/bash
#SBATCH --job-name=v23_smoke
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# GAST-Narrow v2.3 smoke (whole node, the GPU allocation unit here): 4 GPUs, 192 envs x 16 steps x 5 updates (from 371).
set -euo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export GD_LAB_SIF=$G/gd_lab_isaaclab.sif GD_LAB_PYTHON=$G/venv/bin/python APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
# --nv does not bind the host Vulkan ICD (named nvidia_icd.aarch64.json, absolute lib path): Isaac then finds no GPU device.
# isaacsim aarch64: libgomp and libcarb must be preloaded (static TLS)
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
echo "node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} start=$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cd $G/gd_lab_vrl
export NARROW_GPUS=0,1,2,3
( while sleep 30; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed "s/^/[gpu $(date +%T)] /"; done ) &
# No markers: their USDs are fetched from S3, which hangs on the compute nodes; training does not need them.
time bash scripts/gast_narrow/train_narrow_v23.sh smoke env.commands.base_velocity.debug_vis=false
echo "DONE $(date)"
