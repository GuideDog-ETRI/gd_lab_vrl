#!/bin/bash
#SBATCH --job-name=nccl_scale
#SBATCH --partition=natStr
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --comment="field=robotics;appl=pytorch-ddp"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# NCCL at scale without Isaac: Socket over hsn0 (as the training launcher) vs AWS OFI (Hangang guide).
G=/scratch/t515a05/gd_lab
module load apptainer
export APPTAINER_BINDPATH=/scratch
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
run() {
  local label=$1; shift
  echo "===== $label nodes=$SLURM_NNODES $(date +%T)"
  srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none env "$@" timeout 300 \
    apptainer exec --nv $G/gd_lab_isaaclab.sif python3.11 -m torch.distributed.run \
    --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$head:$((29500 + RANDOM % 1000)) \
    --rdzv_id=$label-$SLURM_JOB_ID $G/jobs/nccltest/dist_probe.py 2>&1 | grep -E "PROBE|Error|WARN|Timeout|timed out" | grep -v fuse2fs | sort -u | head -8
  echo "rc=${PIPESTATUS[0]} $(date +%T)"
}
run socket_hsn0 NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn0 GLOO_SOCKET_IFNAME=hsn0 NCCL_PROTO=^LL128
