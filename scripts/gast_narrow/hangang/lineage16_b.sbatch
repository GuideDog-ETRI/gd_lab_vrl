#!/bin/bash
#SBATCH --job-name=lin16_b
#SBATCH --partition=natStr
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=23:50:00
#SBATCH --exclude=node1730,node1395,node1402
#SBATCH --comment="field=robotics;appl=pytorch-ddp"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# Lineage stage B (docs/hangang/WORKORDER_LINEAGE16_20261030.md): the whole GAST-Narrow v2.3 recipe in one run from the
# stage A model (fresh LiDAR encoder, zero-widened first layers), anchor = the same stage A model.
#   STAGE_A=<stage A best .pt> STAGE_A_SHA256=<sha> sbatch lineage16_b.sbatch
# Continue a run (next 24 h job): NARROW_SOURCE=<model_N.pt of this run> NARROW_SOURCE_SHA256=<sha> (PPO restored).
# Stop early with a save: touch $G/gd_lab_vrl/logs/launches/$GAST_RUN_ID/STOP
set -uo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export GD_LAB_SIF=$G/gd_lab_isaaclab.sif GD_LAB_PYTHON=$G/venv/bin/python APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
: "${STAGE_A:?STAGE_A=<stage A best checkpoint>}" "${STAGE_A_SHA256:?STAGE_A_SHA256=<its sha256>}"
per=${PER_GPU:-2048}
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export NARROW_GPUS=0,1,2,3 NARROW_NCCL_WORKAROUND=0 NARROW_NNODES=$SLURM_NNODES NARROW_RDZV_ENDPOINT=$head:$((20000 + SLURM_JOB_ID % 20000))
export NARROW_TOTAL_ENVS=$((per * 4 * SLURM_NNODES)) NARROW_UPDATES=${NARROW_UPDATES:-1200} NARROW_SAVE_EVERY=50
export NARROW_EXPERIMENT=gast/lineage16_b GD_LAB_TOP5_CRITERIA_FILE=$G/gd_lab_vrl/configs/gast_narrow/online_top5_lineage16_b.json
# LINEAGE_RECIPE=v24 (default): the v2.4 task (v2.3 + in-place turns, rear Mid-360, stronger gap nose-up); v23: the v2.3 task.
if [[ ${LINEAGE_RECIPE:-v24} == v24 ]]; then
  export NARROW_TRAIN_PY=scripts/gast_narrow/train_narrow_v24.py NARROW_TASK=Gd-GastNarrowV24-Rbq10-Dreamwaq-v0
fi
export NARROW_SOURCE=${NARROW_SOURCE:-$STAGE_A} NARROW_SOURCE_SHA256=${NARROW_SOURCE_SHA256:-$STAGE_A_SHA256}
export NARROW_ANCHOR_CKPT=$STAGE_A
export GAST_RUN_ID=lineage16_b_${SLURM_NNODES}n_${per}x4_${NARROW_UPDATES}_$(date +%Y%m%d)_${SLURM_JOB_ID}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NARROW_TORCHRUN_ARGS=${NARROW_TORCHRUN_ARGS:-"--log_dir $G/logs/ranklogs_${SLURM_JOB_ID} --redirects 3"}
echo "stage=B cpus_on_node=$SLURM_CPUS_ON_NODE nodes=$SLURM_NNODES head=$head per_gpu=$per total=$NARROW_TOTAL_ENVS source=$NARROW_SOURCE anchor=$STAGE_A start=$(date)"
cd $G/gd_lab_vrl
t0=$(date +%s)
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none --kill-on-bad-exit=1 bash -c '
  h=/tmp/gdhome_$SLURM_JOB_ID; mkdir -p $h/.local/share $h/.cache
  cp -a $HOME/.local/share/ov $h/.local/share/ && cp -a $HOME/.nvidia-omniverse $h/ 2>/dev/null
  cp -a $HOME/.cache/ov $h/.cache/ 2>/dev/null
  NARROW_APPTAINER_HOME=$h exec bash scripts/gast_narrow/train_narrow_v23.sh train env.commands.base_velocity.debug_vis=false'
rc=$?
echo "=== TRAIN_DONE stage=B nodes=$SLURM_NNODES per_gpu=$per total=$NARROW_TOTAL_ENVS rc=$rc wall=$(( $(date +%s) - t0 ))s"
