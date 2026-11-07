#!/usr/bin/env bash
# Backstride A18 (11-05): A17 (A15 wide trot + diagonal trot near slots, backward gaps <= 0.20 m) + crossing landing
# >= 0.10 m past the far edge (-20) + the trot / stride / landing terms on forward gaps too, R17 off both ways.
#   STAGE_A=<a1993 checkpoint> STAGE_A_SHA256=<sha> scripts/gast_backstride/run_a18.sh <smoke|pilot|train>
# onvlm2: GPUs 0,1,2 (never 3), 5,120 envs x 300 updates.
set -euo pipefail
here="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
export BACKSTRIDE_BOTH_DIR=1 BACKSTRIDE_W_LAND_MARGIN=-20 BACKSTRIDE_W_DIAG=-3 BACKSTRIDE_W_STRIDE=-6 BACKSTRIDE_W_CLOSE_LAND=-10
export BACKSTRIDE_BACK_R17=0 BACKSTRIDE_W_SUPPORT=0 BACKSTRIDE_W_ORDER=0 BACKSTRIDE_W_PRINT=0 BACKSTRIDE_BACK_GAP_MAX=0.20
exec bash "$here/train_backstride_3gpu.sh" "$@"
