#!/bin/bash
#SBATCH --job-name=lin16_a
#SBATCH --partition=natStr
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=23:50:00
#SBATCH --signal=B:USR1@900
#SBATCH --exclude=node1730,node1395,node1402
#SBATCH --comment="field=robotics;appl=pytorch-ddp"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# Lineage stage A (docs/hangang/WORKORDER_LINEAGE16_20261030.md): backward gap from 3906 (the 6566 recipe,
# task Gd-GastRayIrV21Gap50BackV2), 16 nodes x 4 GPUs x PER_GPU envs, NARROW_UPDATES updates, save every 50.
# Continue a run (next 24 h job): NARROW_SOURCE=<model_N.pt> NARROW_SOURCE_SHA256=<sha> LINEAGE_CONTINUE=1 sbatch ...
#   (weights / CENet / terrain levels restored, new PPO optimizer, ramps not restarted).
# Stop early with a save: touch $G/gd_lab_vrl/logs/launches/$GAST_RUN_ID/STOP
# 24 h limit: Slurm sends USR1 15 min before the limit -> the run saves and stops; this script then resubmits itself
# from the newest model_<n>.pt with the remaining updates (NO_RESUBMIT=1 turns that off). The same happens after a
# crash that left a checkpoint. Check `squeue` for the follow-up job; the run id carries the job id.
set -uo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export GD_LAB_SIF=$G/gd_lab_isaaclab.sif GD_LAB_PYTHON=$G/venv/bin/python APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
per=${PER_GPU:-2048}
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export NARROW_GPUS=0,1,2,3 NARROW_NCCL_WORKAROUND=0 NARROW_NNODES=$SLURM_NNODES NARROW_RDZV_ENDPOINT=$head:$((20000 + SLURM_JOB_ID % 20000))
export NARROW_TOTAL_ENVS=$((per * 4 * SLURM_NNODES)) NARROW_UPDATES=${NARROW_UPDATES:-1200} NARROW_SAVE_EVERY=50
export NARROW_TRAIN_PY=scripts/gast_narrow/train_lineage_a.py NARROW_TASK=Gd-GastRayIrV21Gap50BackV2-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=gast/lineage16_a GD_LAB_TOP5_CRITERIA_FILE=$G/gd_lab_vrl/configs/gast_backgap/online_top5.json
export NARROW_SOURCE=${NARROW_SOURCE:-$G/gd_lab_vrl/checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt}
export NARROW_SOURCE_SHA256=${NARROW_SOURCE_SHA256:-f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98}
# No anchor runner in this task, so nothing flushes a deferred normalizer: per-step all_reduce (64 ranks on OFI).
export GD_LAB_NORMALIZER_DEFER=0
# First job: the 6566 ramps (backward share 200 updates, speed ceiling, handle force) from their task defaults.
[[ ${LINEAGE_CONTINUE:-0} == 1 ]] || export NARROW_KEEP_RAMPS=1
export GAST_RUN_ID=lineage16_a_${SLURM_NNODES}n_${per}x4_${NARROW_UPDATES}_$(date +%Y%m%d)_${SLURM_JOB_ID}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NARROW_TORCHRUN_ARGS=${NARROW_TORCHRUN_ARGS:-"--log_dir $G/logs/ranklogs_${SLURM_JOB_ID} --redirects 3"}
echo "stage=A cpus_on_node=$SLURM_CPUS_ON_NODE nodes=$SLURM_NNODES head=$head per_gpu=$per total=$NARROW_TOTAL_ENVS source=$NARROW_SOURCE start=$(date)"
cd $G/gd_lab_vrl
t0=$(date +%s)
# hydra.output_subdir=null: 64 ranks writing outputs/<date>/<time>/.hydra/ in the same second on /scratch (Lustre)
# killed rank 4 of job 1182064 with OSError EIO (10-30); the Hydra config dump is not needed.
# USR1 15 min before the limit -> STOP file (save + exit). The srun must run in the BACKGROUND: bash runs a trap only
# after the foreground command returns, so with a foreground srun the STOP file was never written (10-31, job
# 1182335: the limit killed the job without a save; the Hangang session wrote STOP by hand).
trap 'echo "USR1: time limit near, saving"; touch logs/launches/$GAST_RUN_ID/STOP' USR1
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none --kill-on-bad-exit=1 bash -c '
  h=/tmp/gdhome_$SLURM_JOB_ID; mkdir -p $h/.local/share $h/.cache
  cp -a $HOME/.local/share/ov $h/.local/share/ && cp -a $HOME/.nvidia-omniverse $h/ 2>/dev/null
  cp -a $HOME/.cache/ov $h/.cache/ 2>/dev/null
  NARROW_APPTAINER_HOME=$h exec bash scripts/gast_narrow/train_narrow_v23.sh train env.commands.base_velocity.debug_vis=false hydra.output_subdir=null' &
sp=$!
wait $sp; rc=$?
while kill -0 $sp 2>/dev/null; do wait $sp; rc=$?; done  # wait returns early when the trap runs; wait again
echo "=== TRAIN_DONE stage=A nodes=$SLURM_NNODES per_gpu=$per total=$NARROW_TOTAL_ENVS rc=$rc wall=$(( $(date +%s) - t0 ))s"
# Resubmit from the newest checkpoint when updates remain (time limit, STOP file, crash with a checkpoint).
# Resubmit hygiene (10-31, job 1187624 failed in 2 min): the checkpoint path must be ABSOLUTE (sha256sum runs in
# $G/jobs), and sbatch passes this job's environment on, so the per-job variables (rank-log dir, ramp flag, source,
# run id, update count, rendezvous) are unset for the child.
ck=$(ls -t $G/gd_lab_vrl/logs/gast/lineage16_a/$GAST_RUN_ID/model_*.pt 2>/dev/null | head -1)
if [[ -n $ck && ${NO_RESUBMIT:-0} != 1 ]]; then
  done_n=$(basename "$ck" .pt | sed 's/model_//'); left=$(( NARROW_UPDATES - done_n ))
  if (( left > 20 )); then
    echo "resubmit: $ck done=$done_n left=$left"
    (cd $G/jobs && env -u NARROW_TORCHRUN_ARGS -u NARROW_KEEP_RAMPS -u NARROW_SOURCE -u NARROW_SOURCE_SHA256 -u GAST_RUN_ID -u NARROW_UPDATES -u NARROW_TOTAL_ENVS -u NARROW_RDZV_ENDPOINT -u NARROW_NNODES NARROW_SOURCE=$ck NARROW_SOURCE_SHA256=$(sha256sum "$ck" | cut -d" " -f1) NARROW_UPDATES=$left \
       LINEAGE_CONTINUE=1 PER_GPU=$per sbatch --parsable -N $SLURM_NNODES lineage16_a.sbatch)
  fi
fi
