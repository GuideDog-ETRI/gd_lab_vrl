#!/bin/bash
#SBATCH --job-name=v23_pilot
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --comment="field=robotics;appl=pytorch-ddp"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# GAST-Narrow v2.3 pilot: 4 GPUs, envs per GPU sweep (PILOT_PER_GPU), NARROW_UPDATES updates each; time + memory.
set -uo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export GD_LAB_SIF=$G/gd_lab_isaaclab.sif GD_LAB_PYTHON=$G/venv/bin/python APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
# --nv does not bind the host Vulkan ICD (named nvidia_icd.aarch64.json, absolute lib path): Isaac then finds no GPU device.
# isaacsim aarch64: libgomp and libcarb must be preloaded (static TLS)
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
export NARROW_GPUS=0,1,2,3 NARROW_UPDATES=${NARROW_UPDATES:-20}
echo "node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} start=$(date)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
cd $G/gd_lab_vrl
for per in ${PILOT_PER_GPU:-1366 2048 4096 8192}; do
  total=$((per * 4))
  mem=$G/logs/pilot_${SLURM_JOB_ID}_${per}.mem
  ( while sleep 10; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits; done ) > $mem &
  mon=$!
  echo "=== PILOT per_gpu=$per total=$total start=$(date +%T)"
  t0=$(date +%s)
  NARROW_TOTAL_ENVS=$total GAST_RUN_ID=pilot_${SLURM_JOB_ID}_${per}x4 \
    bash scripts/gast_narrow/train_narrow_v23.sh pilot env.commands.base_velocity.debug_vis=false
  rc=$?
  kill $mon
  peak=$(awk -F', ' '{if($3>m[$1])m[$1]=$3; if($2>u[$1])u[$1]=$2} END{for(i in m) printf "gpu%s %d MiB (util max %d%%) ", i, m[i], u[i]}' $mem)
  echo "=== PILOT_DONE per_gpu=$per total=$total rc=$rc wall=$(( $(date +%s) - t0 ))s peak: $peak"
  [[ $rc -eq 0 ]] || break
done
echo "DONE $(date)"
