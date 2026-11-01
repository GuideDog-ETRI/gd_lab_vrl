#!/usr/bin/env bash
# Stage B' (docs/hangang/WORKORDER_STAGE_BPRIME_20261101.md) on onvlm2 GPUs 0,1,2 (never 3): v2.5 passages (v25p: no
# turn / cliff tiles) + gait v3 G5 terms from an A'-type model (task Gd-GastGaitV3Gap).
#   NARROW_SOURCE=<A' .pt> NARROW_SOURCE_SHA256=<sha> scripts/gast_bprime/train_bprime_3gpu.sh <smoke|pilot|train> [overrides]
# Runs train_narrow_v23.sh (multi-GPU launcher, onvlm2 NCCL workaround). Both anchors (retention on non-narrow tiles,
# stand) target the source model itself; no gap_nose_up in B', so the anchor keeps every gap sample.
set -euo pipefail
root="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"
mode="${1:?usage: train_bprime_3gpu.sh smoke|pilot|train [overrides]}"
: "${NARROW_SOURCE:?NARROW_SOURCE=<A-prime checkpoint>}" "${NARROW_SOURCE_SHA256:?NARROW_SOURCE_SHA256=<its sha256>}"
export NARROW_TRAIN_PY=scripts/gast_bprime/train_bprime.py NARROW_TASK=Gd-GastNarrowBPrime-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=gast/bprime
export NARROW_GPUS=${NARROW_GPUS:-0,1,2} NARROW_UPDATES=${NARROW_UPDATES:-300} NARROW_TOTAL_ENVS=${NARROW_TOTAL_ENVS:-5120}
export NARROW_TURN=0 NARROW_CLIFF_COLUMNS=0 GAIT_RAMP_STEPS=0 GAIT_R20=${GAIT_R20:-1}
export NARROW_ANCHOR_CKPT=${NARROW_ANCHOR_CKPT:-$NARROW_SOURCE} NARROW_STAND_ANCHOR_CKPT=${NARROW_STAND_ANCHOR_CKPT:-$NARROW_SOURCE}
export NARROW_ANCHOR_GAP_NOSE=0
export GD_LAB_TOP5_CRITERIA_FILE="${GD_LAB_TOP5_CRITERIA_FILE:-$root/configs/gast_narrow/online_top5_lineage16_b_v25p.json}"
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_bprime_${mode}}"
exec bash "$root/scripts/gast_narrow/train_narrow_v23.sh" "$@"
