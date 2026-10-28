#!/bin/bash
#SBATCH --job-name=nccl_ofi
#SBATCH --partition=natStr
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --comment="field=robotics;appl=pytorch-ddp"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# NCCL-OFI (Slingshot CXI) probe in our SIF: broadcast_object_list, small and 16 MB all_reduce, 2000 tiny all_reduces.
G=/scratch/t515a05/gd_lab
module load apptainer
export APPTAINER_BINDPATH=/scratch
source $G/jobs/ofi_env.sh
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "nodes=$SLURM_NNODES start=$(date +%T)"
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none timeout 400 \
  apptainer exec --nv $G/gd_lab_isaaclab.sif python3.11 -m torch.distributed.run \
  --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$head:29511 --rdzv_id=ofi-$SLURM_JOB_ID \
  $G/jobs/nccltest/dist_probe.py > $G/logs/nccl_ofi_${SLURM_JOB_ID}_full.txt 2>&1
grep -E "^PROBE" $G/logs/nccl_ofi_${SLURM_JOB_ID}_full.txt
grep -c "Using network AWS Libfabric" $G/logs/nccl_ofi_${SLURM_JOB_ID}_full.txt | sed "s/^/ranks using AWS Libfabric: /"
grep -E "Error|WARN|Timeout" $G/logs/nccl_ofi_${SLURM_JOB_ID}_full.txt | grep -v fuse2fs | sort | uniq -c | head -5
echo "rc=${PIPESTATUS[0]} end=$(date +%T)"
