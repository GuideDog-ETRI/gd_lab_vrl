#!/bin/bash
#SBATCH --job-name=gd_build
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --comment="field=robotics;appl=misc"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# aarch64 SIF (wheel path, containers/gd_lab_isaaclab.def unchanged) + host venv on /scratch.
set -euo pipefail
G=/scratch/t515a05/gd_lab
REPO=$G/gd_lab_vrl SIF=$G/gd_lab_isaaclab.sif VENV=$G/venv ISAACLAB=$G/IsaacLab
echo "node=$(hostname) arch=$(uname -m) start=$(date)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
curl -sSI --max-time 15 https://pypi.org/simple/ | head -1 || { echo "NO INTERNET on compute node"; exit 3; }
module load apptainer
export APPTAINER_CACHEDIR=$G/.apptainer_cache
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/gd_build_$SLURM_JOB_ID
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
df -h "$APPTAINER_TMPDIR" | tail -1
if [[ ! -f $SIF ]]; then
  apptainer build --fakeroot "$SIF.tmp" "$REPO/containers/gd_lab_isaaclab.def"
  mv "$SIF.tmp" "$SIF"
fi
ls -lh "$SIF"
X() { apptainer exec --nv --bind /scratch "$SIF" "$@"; }
X python3.11 -c 'import torch,platform;print("torch",torch.__version__,"cuda",torch.version.cuda,platform.machine())'
[[ -x $VENV/bin/python ]] || X python3.11 -m venv --system-site-packages "$VENV"
[[ -d $ISAACLAB ]] || git clone -b v2.3.1 --depth 1 https://github.com/isaac-sim/IsaacLab "$ISAACLAB"
X "$VENV/bin/pip" install -e "$ISAACLAB/source/isaaclab" -e "$ISAACLAB/source/isaaclab_assets" \
  -e "$ISAACLAB/source/isaaclab_rl[rsl-rl]" -e "$ISAACLAB/source/isaaclab_tasks"
X "$VENV/bin/pip" install -e "$REPO" --no-deps
X "$VENV/bin/pip" install pytest
# aarch64: rsl-rl's tensordict (latest) pulls torch 2.14+cu130 into the venv (needs driver >= 580; nodes have 570).
# Keep the image's torch 2.7.0+cu128: drop the venv copies and pin a tensordict that accepts torch 2.7.
printf 'torch==2.7.0+cu128\ntorchvision==0.22.0\n' > $G/pip-constraints-gh200.txt
pk=$(X "$VENV/bin/pip" list --path "$VENV/lib/python3.11/site-packages" | awk 'NR>2{print $1}' | grep -E '^(torch|torchvision|triton|nvidia-)' || true)
[[ -z $pk ]] || X "$VENV/bin/pip" uninstall -y $pk
X "$VENV/bin/pip" install -c $G/pip-constraints-gh200.txt "tensordict<0.15"
# isaacsim aarch64: libgomp and libcarb must be preloaded (static TLS).
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
export OMNI_KIT_ACCEPT_EULA=YES
X "$VENV/bin/python" -c 'import torch,isaacsim,isaaclab,rsl_rl,gd_lab;print("imports ok; cuda avail:",torch.cuda.is_available())'
rm -rf "$APPTAINER_TMPDIR"
echo "DONE $(date)"
