#!/bin/bash
#SBATCH --job-name=e2_multi
#SBATCH --partition=natStr
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# Several E2 configs in one job, one node each (the natStr QOS allows 5 jobs per user).
#   E2_CONFIGS="cfg1 cfg2 ..." sbatch -N <count> e2_multi.sbatch
set -uo pipefail
G=/scratch/t515a05/gd_lab
read -ra CFGS <<< "$E2_CONFIGS"
echo "start $(date) nodes=$SLURM_JOB_NODELIST configs=${CFGS[*]}"
for c in "${CFGS[@]}"; do
  srun -N1 -n1 --exclusive --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none \
       --export=ALL,E2_CONFIG=$c --output=$G/logs/${c}_$SLURM_JOB_ID.out bash $G/jobs/e2_full.sbatch &
  sleep 2
done
wait
echo "DONE $(date)"
