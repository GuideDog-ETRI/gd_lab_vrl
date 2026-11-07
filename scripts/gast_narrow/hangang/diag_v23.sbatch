#!/bin/bash
#SBATCH --job-name=v23_diag
#SBATCH --partition=natStr
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --comment="field=robotics;appl=pytorch-ddp"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# GAST-Narrow v2.3 main training from 371: SLURM_NNODES x 4 GPUs, PER_GPU envs per GPU, NARROW_UPDATES updates.
# Stop early with a save: touch $G/gd_lab_vrl/logs/launches/$GAST_RUN_ID/STOP
set -uo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export GD_LAB_SIF=$G/gd_lab_isaaclab.sif GD_LAB_PYTHON=$G/venv/bin/python APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
# --nv does not bind the host Vulkan ICD (named nvidia_icd.aarch64.json, absolute lib path): Isaac then finds no GPU device.
# isaacsim aarch64: libgomp and libcarb must be preloaded (static TLS)
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
per=${PER_GPU:-2048}
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export NARROW_GPUS=0,1,2,3 NARROW_NCCL_WORKAROUND=0 NARROW_NNODES=$SLURM_NNODES NARROW_RDZV_ENDPOINT=$head:$((20000 + SLURM_JOB_ID % 20000))  # job-specific port (Hangang guide)
export NARROW_TOTAL_ENVS=$((per * 4 * SLURM_NNODES)) NARROW_UPDATES=${NARROW_UPDATES:-600}
export GAST_RUN_ID=narrow_v23_371_${SLURM_NNODES}n_${per}x4_${NARROW_UPDATES}_$(date +%Y%m%d)_${SLURM_JOB_ID}
[[ ${DIAG_NCCL:-0} == 1 ]] && source $G/jobs/diag_nccl.env
export NCCL_DEBUG=WARN NARROW_HEARTBEAT=${NARROW_HEARTBEAT:-10} NARROW_HEARTBEAT_ALL=${NARROW_HEARTBEAT_ALL:-0} NARROW_STACK_DUMP_S=${NARROW_STACK_DUMP_S:-240} NARROW_STACK_DUMP_RANKS=${NARROW_STACK_DUMP_RANKS:-0,1,4,255,511}
echo "cpus_on_node=$SLURM_CPUS_ON_NODE nodes=$SLURM_NNODES head=$head per_gpu=$per total=$NARROW_TOTAL_ENVS start=$(date)"
cd $G/gd_lab_vrl
t0=$(date +%s)
# srun gives a step ONE cpu by default: 4 ranks + Isaac on one core ran ~24x slower (GPU 4 %). Give it the node.
# Per-node $HOME on node-local /tmp, seeded with the Kit user data (extension registry, downloaded extensions):
# one shared ~/.local/share/ov is rewritten by every process at start and breaks at 512 processes.
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none --kill-on-bad-exit=1 bash -c '
  h=/tmp/gdhome_$SLURM_JOB_ID; mkdir -p $h/.local/share $h/.cache
  cp -a $HOME/.local/share/ov $h/.local/share/ && cp -a $HOME/.nvidia-omniverse $h/ 2>/dev/null
  cp -a $HOME/.cache/ov $h/.cache/ 2>/dev/null
  NARROW_APPTAINER_HOME=$h exec bash scripts/gast_narrow/train_narrow_v23.sh pilot env.commands.base_velocity.debug_vis=false'
echo "=== TRAIN_DONE nodes=$SLURM_NNODES per_gpu=$per total=$NARROW_TOTAL_ENVS rc=$? wall=$(( $(date +%s) - t0 ))s"
