#!/usr/bin/env bash
# GAST-Narrow v2.5 (cliff stop, turn success redefined) from v24h it653 on onvlm2 GPUs 0,1,2.
#   scripts/gast_narrow/train_narrow_v25_3gpu.sh <smoke|pilot|train> [extra hydra overrides]
# Design: docs/design/GAST_NARROW_V25_DESIGN.md. Runs train_narrow_v23.sh (multi-GPU launcher, onvlm2 NCCL
# workaround, 6566 anchor) with the v2.4 trainer / task: train = NARROW_TOTAL_ENVS (5120) envs x 100 steps x NARROW_UPDATES (800), save 100.
# v2.4b (10-31): 5120 envs (4096 used 30 of 48 GB per GPU; 6144 would not fit), 800 updates.
set -euo pipefail
root="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"
mode="${1:?usage: train_narrow_v25_3gpu.sh smoke|pilot|train [overrides]}"
export NARROW_TRAIN_PY=scripts/gast_narrow/train_narrow_v25.py NARROW_TASK=Gd-GastNarrowV25-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=gast/narrow_v25
export NARROW_SOURCE="${NARROW_SOURCE:-$root/checkpoints/gast_narrow/narrow_v23_gh200_20261029/v23_gh200_it602_top5.pt}"
export NARROW_SOURCE_SHA256=${NARROW_SOURCE_SHA256:-69e7e065ee18cff88e9d7ed0876922374ae830d8ae710b04b80c576d22209c05}
export NARROW_GPUS=${NARROW_GPUS:-0,1,2} NARROW_UPDATES=${NARROW_UPDATES:-800} NARROW_TOTAL_ENVS=${NARROW_TOTAL_ENVS:-5120}
export GD_LAB_TOP5_CRITERIA_FILE="${GD_LAB_TOP5_CRITERIA_FILE:-$root/configs/gast_narrow/online_top5_v25.json}"
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_narrow_v25_653_${mode}}"
exec bash "$root/scripts/gast_narrow/train_narrow_v23.sh" "$@"
