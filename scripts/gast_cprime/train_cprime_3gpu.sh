#!/usr/bin/env bash
# Stage C' (docs/design/GAST_NARROW_V26_CLIFF_DESIGN.md) on onvlm2 GPUs 0,1,2 (never 3): B' + the v2.5 cliff decks
# (4 columns, front / back only, 0.3..0.8 m/s) + the support_end actor input, from a B' model (task GastNarrowBPrime).
#   NARROW_SOURCE=<B' .pt> NARROW_SOURCE_SHA256=<sha> scripts/gast_cprime/train_cprime_3gpu.sh <smoke|pilot|train> [overrides]
# Runs train_narrow_v23.sh (multi-GPU launcher, onvlm2 NCCL workaround). Both anchors (retention on non-narrow tiles,
# stand) target the B' source model (NARROW_ANCHOR_CKPT / NARROW_STAND_ANCHOR_CKPT override); cliff rows are narrow
# tiles, so the retention anchor skips them.
set -euo pipefail
root="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"
mode="${1:?usage: train_cprime_3gpu.sh smoke|pilot|train [overrides]}"
: "${NARROW_SOURCE:?NARROW_SOURCE=<B-prime checkpoint>}" "${NARROW_SOURCE_SHA256:?NARROW_SOURCE_SHA256=<its sha256>}"
export NARROW_TRAIN_PY=scripts/gast_cprime/train_cprime.py NARROW_TASK=Gd-GastNarrowCPrime-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=gast/cprime
export NARROW_GPUS=${NARROW_GPUS:-0,1,2} NARROW_UPDATES=${NARROW_UPDATES:-300} NARROW_TOTAL_ENVS=${NARROW_TOTAL_ENVS:-5120}
export NARROW_TURN=0 NARROW_CLIFF_COLUMNS=4 GAIT_RAMP_STEPS=0 GAIT_R20=${GAIT_R20:-1}
export NARROW_ANCHOR_CKPT=${NARROW_ANCHOR_CKPT:-$NARROW_SOURCE} NARROW_STAND_ANCHOR_CKPT=${NARROW_STAND_ANCHOR_CKPT:-$NARROW_SOURCE}
export NARROW_ANCHOR_GAP_NOSE=0
export GD_LAB_TOP5_CRITERIA_FILE="${GD_LAB_TOP5_CRITERIA_FILE:-$root/configs/gast_narrow/online_top5_lineage16_cprime.json}"
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_cprime_${mode}}"
exec bash "$root/scripts/gast_narrow/train_narrow_v23.sh" "$@"
