#!/usr/bin/env bash
# Backward gap stride (gd_lab.gast_backstride): A' continuation on onvlm2 GPUs 0,1,2 (never 3) or the local 5090.
#   STAGE_A=<A' checkpoint> STAGE_A_SHA256=<sha> scripts/gast_backstride/train_backstride_3gpu.sh <smoke|pilot|train>
# Full weights from update 0 (continuation, no ramps); retention anchor toward STAGE_A off the backward gap steps.
set -euo pipefail
root="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"
mode="${1:?usage: train_backstride_3gpu.sh smoke|pilot|train [overrides]}"
: "${STAGE_A:?STAGE_A=<start checkpoint>}" "${STAGE_A_SHA256:?STAGE_A_SHA256=<its sha256>}"
export NARROW_TRAIN_PY=scripts/gast_backstride/train_backstride.py NARROW_TASK=Gd-GastGaitV3BackStride-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=${NARROW_EXPERIMENT:-gast/gait_v3_backstride}
export NARROW_SOURCE=${NARROW_SOURCE:-$STAGE_A} NARROW_SOURCE_SHA256=${NARROW_SOURCE_SHA256:-$STAGE_A_SHA256}
export NARROW_ANCHOR_CKPT=$STAGE_A NARROW_STAND_ANCHOR_WEIGHT=0 NARROW_ANCHOR_GAP_NOSE=0
export NARROW_GPUS=${NARROW_GPUS:-0,1,2} NARROW_UPDATES=${NARROW_UPDATES:-300} NARROW_TOTAL_ENVS=${NARROW_TOTAL_ENVS:-5120}
export GD_LAB_NORMALIZER_DEFER=0 NARROW_KEEP_RAMPS=0 GAIT_RAMP_STEPS=0 GAIT_R20=1
export GD_LAB_TOP5_CRITERIA_FILE="${GD_LAB_TOP5_CRITERIA_FILE:-$root/configs/gast_gait/online_top5_g2.json}"
export GAST_RUN_ID="${GAST_RUN_ID:-backstride_$(date +%Y%m%d_%H%M)_${mode}}"
exec bash "$root/scripts/gast_narrow/train_narrow_v23.sh" "$mode" "${@:2}"
