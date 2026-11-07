#!/bin/bash
#SBATCH --job-name=v23_scale
#SBATCH --partition=natStr
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --comment="field=robotics;appl=pytorch-ddp"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# GAST-Narrow v2.3 multi-node scaling: SLURM_NNODES x 4 GPUs, PER_GPU envs per GPU, NARROW_UPDATES updates.
set -uo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export GD_LAB_SIF=$G/gd_lab_isaaclab.sif GD_LAB_PYTHON=$G/venv/bin/python APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
# --nv does not bind the host Vulkan ICD (named nvidia_icd.aarch64.json, absolute lib path): Isaac then finds no GPU device.
# isaacsim aarch64: libgomp and libcarb must be preloaded (static TLS)
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
per=${PER_GPU:-1366}
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export NARROW_GPUS=0,1,2,3 NARROW_NCCL_WORKAROUND=0 NARROW_NNODES=$SLURM_NNODES NARROW_RDZV_ENDPOINT=$head:$((20000 + SLURM_JOB_ID % 20000))  # job-specific port (Hangang guide)
export NARROW_TOTAL_ENVS=$((per * 4 * SLURM_NNODES)) NARROW_UPDATES=${NARROW_UPDATES:-10}
export GAST_RUN_ID=scale_${SLURM_JOB_ID}_${SLURM_NNODES}n_${per}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
echo "cpus_on_node=$SLURM_CPUS_ON_NODE nodes=$SLURM_NNODES head=$head per_gpu=$per total=$NARROW_TOTAL_ENVS start=$(date)"
cd $G/gd_lab_vrl
t0=$(date +%s)
# srun gives a step ONE cpu by default: 4 ranks + Isaac on one core ran ~24x slower (GPU 4 %). Give it the node.
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none --kill-on-bad-exit=1 \
  bash scripts/gast_narrow/train_narrow_v23.sh pilot \
  env.commands.base_velocity.debug_vis=false
echo "=== SCALE_DONE nodes=$SLURM_NNODES per_gpu=$per total=$NARROW_TOTAL_ENVS rc=$? wall=$(( $(date +%s) - t0 ))s"
