#!/usr/bin/env bash
# Stage B' + backstride (11-05, gd_lab.gast_backstride.bprime): the B' recipe (fc7d4e9: v25p passages, stand when
# blocked, backward passages, gait v3 G5) + the A18L gap terms (long diagonal trot both ways, no three-point support,
# step floor 0.50 m, landing >= 0.10 m past the far edge, backward gaps <= 0.20 m), from the A18L model.
#   NARROW_SOURCE=<A18L .pt> NARROW_SOURCE_SHA256=<sha> scripts/gast_backstride/train_bprime_stride_3gpu.sh <smoke|pilot|train>
# onvlm2: GPUs 0,1,2 (never 3), 5,120 envs x 300 updates. Anchors (retention off narrow tiles and gap steps, stand)
# target the source model, as B'.
set -euo pipefail
root="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"
mode="${1:?usage: train_bprime_stride_3gpu.sh smoke|pilot|train [overrides]}"
: "${NARROW_SOURCE:?NARROW_SOURCE=<A18L checkpoint>}" "${NARROW_SOURCE_SHA256:?NARROW_SOURCE_SHA256=<its sha256>}"
source "$root/scripts/gast_backstride/a18l_env.sh"
export NARROW_TRAIN_PY=scripts/gast_backstride/train_bprime_stride.py NARROW_TASK=Gd-GastNarrowBPrimeStride-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=${NARROW_EXPERIMENT:-gast/bprime_stride}
export NARROW_GPUS=${NARROW_GPUS:-0,1,2} NARROW_UPDATES=${NARROW_UPDATES:-300} NARROW_TOTAL_ENVS=${NARROW_TOTAL_ENVS:-5120}
export NARROW_TURN=0 NARROW_CLIFF_COLUMNS=0 GAIT_RAMP_STEPS=0 GAIT_R20=${GAIT_R20:-1}
export NARROW_ANCHOR_CKPT=${NARROW_ANCHOR_CKPT:-$NARROW_SOURCE} NARROW_STAND_ANCHOR_CKPT=${NARROW_STAND_ANCHOR_CKPT:-$NARROW_SOURCE}
export NARROW_ANCHOR_GAP_NOSE=0
export GD_LAB_TOP5_CRITERIA_FILE="${GD_LAB_TOP5_CRITERIA_FILE:-$root/configs/gast_narrow/online_top5_lineage16_b_v25p.json}"
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_bprime_stride_${mode}}"
exec bash "$root/scripts/gast_narrow/train_narrow_v23.sh" "$@"
