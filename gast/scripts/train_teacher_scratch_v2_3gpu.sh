#!/usr/bin/env bash
# Experiment 3: GAST teacher from scratch on the v2 objective (Clean gap terms + gap/stair v2 terms),
# Gd-GastScratchV2-Rbq10-Dreamwaq-v0, 3 GPUs (never GPU 3), via the regular from-scratch launcher.
#   gast/scripts/train_teacher_scratch_v2_3gpu.sh smoke|train [extra hydra/CLI overrides]
# The disturbance force ramps over GAST_V2_FORCE_RAMP_STEPS env steps of this process (default 1e6, i.e.
# ~10k updates at 100 steps/env), because a from-scratch policy does not walk stairs early on.
# Reward weights are recorded in the run's params/env.yaml (no warm-start manifest on a scratch run).
set -euo pipefail
here="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
mode="${1:?mode: smoke|train}"
shift
if [ "$mode" = smoke ]; then export GD_LAB_V2_FORCE_RAMP_STEPS=0
else export GD_LAB_V2_FORCE_RAMP_STEPS="${GAST_V2_FORCE_RAMP_STEPS:-1000000}"; fi
exec bash "$here/train_teacher_3gpu.sh" "$mode" --task Gd-GastScratchV2-Rbq10-Dreamwaq-v0 "$@"
