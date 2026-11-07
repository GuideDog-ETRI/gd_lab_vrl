#!/bin/bash
#SBATCH --job-name=e2_nose
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# E2 gap nose-up baseline (v2.3 phase 0): 6566 / 465 / 371, gap suite, 4 conditions.
set -uo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
# isaacsim aarch64: libgomp and libcarb must be preloaded (static TLS)
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
cd $G/gd_lab_vrl
PY=$G/hostpy/bin/python  # host python 3.11 + PyYAML (orchestrator only; trials run in the container)
echo "node=$(hostname) orchestrator=$PY start=$(date)"
$PY scripts/gast_narrow/e2/run_e2.py configs/gast_narrow/e2/nose_baseline_gh200.yaml --only gap
echo "DONE $(date)"
