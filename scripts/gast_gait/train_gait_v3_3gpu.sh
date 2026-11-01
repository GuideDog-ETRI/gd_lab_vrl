#!/usr/bin/env bash
# Gait quality v3 (docs/design/GAIT_V3_DESIGN.md) from 3906 on onvlm2 GPUs 0,1,2 (never 3).
#   scripts/gast_gait/train_gait_v3_3gpu.sh <smoke|pilot|train> [extra hydra overrides]
# Runs train_narrow_v23.sh (multi-GPU launcher) with the gait v3 trainer / task: the 6566 recipe (backward gap V2,
# ramps at their task defaults as the 6566 run had) + the v3 posture / touchdown / leg-following terms.
# Stage G0 (a6000): NARROW_UPDATES=200 GAIT_R9=0 GAIT_R10=0 GAIT_R11=0 (posture + touchdown terms only).
# Stage G1 (a6000): NARROW_UPDATES=300, all terms. Switches: GAIT_R<n>=0 (R1, R4..R11), GAIT_R13=1 keeps backward_nose_up.
# Continuation: NARROW_SOURCE=<model_N.pt of this task> NARROW_SOURCE_SHA256=<sha> NARROW_KEEP_RAMPS=0.
set -euo pipefail
root="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"
mode="${1:?usage: train_gait_v3_3gpu.sh smoke|pilot|train [overrides]}"
export NARROW_TRAIN_PY=scripts/gast_gait/train_gait_v3.py NARROW_TASK=Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0
export NARROW_EXPERIMENT=gast/gait_v3
export NARROW_SOURCE="${NARROW_SOURCE:-/home/user/gd_project/gd_lab_vrl/checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt}"
export NARROW_SOURCE_SHA256=${NARROW_SOURCE_SHA256:-f3e7afe53ea8be0f2dd777f331af417aba20b13b5e5be5470ec3d939470f4d98}
export NARROW_GPUS=${NARROW_GPUS:-0,1,2} NARROW_UPDATES=${NARROW_UPDATES:-200} NARROW_TOTAL_ENVS=${NARROW_TOTAL_ENVS:-5120}
# No anchor runner in this task (as lineage stage A): per-step normalizer all_reduce, 6566 ramps from task defaults.
export GD_LAB_NORMALIZER_DEFER=0 NARROW_KEEP_RAMPS=${NARROW_KEEP_RAMPS:-1} NARROW_ANCHOR_GAP_NOSE=0
# G4: gait weights ramp kappa 0.2 -> 1 over 200 updates (GAIT_RAMP_STEPS env steps; 0 = full weights from update 0).
# A continuation (NARROW_KEEP_RAMPS=0) runs with the full weights, like the 6566 ramps.
[[ ${NARROW_KEEP_RAMPS} == 1 ]] || export GAIT_RAMP_STEPS=${GAIT_RAMP_STEPS:-0}
export GAIT_RAMP_STEPS=${GAIT_RAMP_STEPS:-20000}
export GD_LAB_TOP5_CRITERIA_FILE="${GD_LAB_TOP5_CRITERIA_FILE:-$root/configs/gast_gait/online_top5_g2.json}"
tag=""; for n in 1 4 5 6 7 8 9 10 15 16 17 18 19 20; do v="GAIT_R$n"; [[ ${!v:-1} == 0 ]] && tag="${tag}_noR$n"; done
[[ ${GAIT_RAMP_STEPS} == 0 ]] && tag="${tag}_noramp"
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_gait_v3_3906${tag}_${mode}}"
exec bash "$root/scripts/gast_narrow/train_narrow_v23.sh" "$@"
