#!/bin/bash
#SBATCH --job-name=aprime16
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
# Stage A' (docs/hangang/WORKORDER_STAGE_APRIME_20261101.md): backward gap + gait v3 G4 from 3906 (the 6566 recipe
# + docs/design/GAIT_V3_DESIGN.md section 3.7, task Gd-GastGaitV3Gap), 16 nodes x 4 GPUs x PER_GPU envs,
# NARROW_UPDATES updates, save every 50. The gait weights ramp 0.2 -> 1 over the first 200 updates (first job only).
# Continue a run (next 24 h job): NARROW_SOURCE=<model_N.pt> NARROW_SOURCE_SHA256=<sha> LINEAGE_CONTINUE=1 APRIME_DONE=<n> sbatch ...
#   (weights / CENet / terrain levels restored, new PPO optimizer, ramps not restarted, gait weights at full).
# Mid E2 (unattended): when the run reaches update 300 / 600 / 1000 / 1500 (counted over resubmits), a 1-node E2 job
# (e2_multi.sbatch) is submitted for that model_N.pt with 3906 and 6566 as baselines; results in logs/e2/aprime_*.
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
export NARROW_TOTAL_ENVS=$((per * 4 * SLURM_NNODES)) NARROW_UPDATES=${NARROW_UPDATES:-2000} NARROW_SAVE_EVERY=50
export NARROW_TRAIN_PY=scripts/gast_gait/train_gait_v3.py NARROW_TASK=Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=gast/aprime16 GD_LAB_TOP5_CRITERIA_FILE=$G/gd_lab_vrl/configs/gast_gait/online_top5_g2.json
export NARROW_SOURCE=${NARROW_SOURCE:-$G/gd_lab_vrl/checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt}
export NARROW_SOURCE_SHA256=${NARROW_SOURCE_SHA256:-f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98}
# No anchor runner in this task, so nothing flushes a deferred normalizer: per-step all_reduce (64 ranks on OFI).
export GD_LAB_NORMALIZER_DEFER=0 NARROW_ANCHOR_GAP_NOSE=0
# First job: the 6566 ramps (backward share 200 updates, speed ceiling, handle force) from their task defaults and the
# gait weight ramp (20,000 env steps = 200 updates x 100); a continuation runs everything at full.
if [[ ${LINEAGE_CONTINUE:-0} == 1 ]]; then export GAIT_RAMP_STEPS=0; else export NARROW_KEEP_RAMPS=1 GAIT_RAMP_STEPS=20000; fi
export APRIME_DONE=${APRIME_DONE:-0}  # updates finished by the earlier jobs of this run (mid E2 thresholds)
export GAST_RUN_ID=aprime16_${SLURM_NNODES}n_${per}x4_${NARROW_UPDATES}_$(date +%Y%m%d)_${SLURM_JOB_ID}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NARROW_TORCHRUN_ARGS=${NARROW_TORCHRUN_ARGS:-"--log_dir $G/logs/ranklogs_${SLURM_JOB_ID} --redirects 3"}
echo "stage=Aprime done_before=$APRIME_DONE cpus_on_node=$SLURM_CPUS_ON_NODE nodes=$SLURM_NNODES head=$head per_gpu=$per total=$NARROW_TOTAL_ENVS source=$NARROW_SOURCE start=$(date)"
cd $G/gd_lab_vrl
t0=$(date +%s)
# hydra.output_subdir=null: 64 ranks writing outputs/<date>/<time>/.hydra/ in the same second on /scratch (Lustre)
# killed rank 4 of job 1182064 with OSError EIO (10-30); the Hydra config dump is not needed.
# USR1 15 min before the limit -> STOP file (save + exit). The srun must run in the BACKGROUND: bash runs a trap only
# after the foreground command returns, so with a foreground srun the STOP file was never written (10-31, job
# 1182335: the limit killed the job without a save; the Hangang session wrote STOP by hand).
trap 'echo "USR1: time limit near, saving"; touch logs/launches/$GAST_RUN_ID/STOP' USR1
mid_e2() {  # background: one 1-node E2 per threshold reached in this job (absolute update = APRIME_DONE + local n)
  local run=$G/gd_lab_vrl/logs/gast/aprime16/$GAST_RUN_ID
  for abs in 300 600 1000 1500; do
    local n=$(( abs - APRIME_DONE - 1 ))
    (( n < 0 || n >= NARROW_UPDATES )) && continue
    local ck="" k
    while [[ -z $ck ]]; do  # first saved model at or after the threshold (saves every 50)
      for k in $(seq $n $((n + 60))); do [[ -f $run/model_$k.pt ]] && { ck=$run/model_$k.pt; break; }; done
      [[ -n $ck ]] || { sleep 300; kill -0 $$ 2>/dev/null || return; }
    done
    sleep 120  # let the save finish
    local name=aprime_mid_${SLURM_JOB_ID}_u$abs
    sed -e "s#^name: .*#name: $name#" -e "s#__APRIME__#$ck#" configs/gast_gait/e2/aprime_hangang.yaml \
      > configs/gast_narrow/e2/$name.yaml
    (cd $G/jobs && sbatch --parsable -N 1 --export=ALL,E2_CONFIGS=$name e2_multi.sbatch) && echo "mid E2 u$abs: $ck -> $name"
  done
}
mid_e2 &
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none --kill-on-bad-exit=1 bash -c '
  h=/tmp/gdhome_$SLURM_JOB_ID; mkdir -p $h/.local/share $h/.cache
  cp -a $HOME/.local/share/ov $h/.local/share/ && cp -a $HOME/.nvidia-omniverse $h/ 2>/dev/null
  cp -a $HOME/.cache/ov $h/.cache/ 2>/dev/null
  NARROW_APPTAINER_HOME=$h exec bash scripts/gast_narrow/train_narrow_v23.sh train env.commands.base_velocity.debug_vis=false hydra.output_subdir=null' &
sp=$!
wait $sp; rc=$?
while kill -0 $sp 2>/dev/null; do wait $sp; rc=$?; done  # wait returns early when the trap runs; wait again
echo "=== TRAIN_DONE stage=Aprime nodes=$SLURM_NNODES per_gpu=$per total=$NARROW_TOTAL_ENVS rc=$rc wall=$(( $(date +%s) - t0 ))s"
# Resubmit from the newest checkpoint when updates remain (time limit, STOP file, crash with a checkpoint).
# Resubmit hygiene (10-31, job 1187624 failed in 2 min): the checkpoint path must be ABSOLUTE (sha256sum runs in
# $G/jobs), and sbatch passes this job's environment on, so the per-job variables (rank-log dir, ramp flag, source,
# run id, update count, rendezvous) are unset for the child.
ck=$(ls -t $G/gd_lab_vrl/logs/gast/aprime16/$GAST_RUN_ID/model_*.pt 2>/dev/null | head -1)
if [[ -n $ck && ${NO_RESUBMIT:-0} != 1 ]]; then
  done_n=$(basename "$ck" .pt | sed 's/model_//'); left=$(( NARROW_UPDATES - done_n ))
  if (( left > 20 )); then
    echo "resubmit: $ck done=$done_n left=$left"
    (cd $G/jobs && env -u NARROW_TORCHRUN_ARGS -u NARROW_KEEP_RAMPS -u NARROW_SOURCE -u NARROW_SOURCE_SHA256 -u GAST_RUN_ID -u NARROW_UPDATES -u NARROW_TOTAL_ENVS -u NARROW_RDZV_ENDPOINT -u NARROW_NNODES -u GAIT_RAMP_STEPS -u APRIME_DONE NARROW_SOURCE=$ck NARROW_SOURCE_SHA256=$(sha256sum "$ck" | cut -d" " -f1) NARROW_UPDATES=$left \
       LINEAGE_CONTINUE=1 APRIME_DONE=$(( APRIME_DONE + done_n + 1 )) PER_GPU=$per sbatch --parsable -N $SLURM_NNODES aprime16.sbatch)
  fi
fi
