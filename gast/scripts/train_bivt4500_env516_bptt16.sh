#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
teacher="$root/../checkpoints/teachers/bivt/ray_4500_20261001/teacher/model_4500.pt"
test -f "$teacher"
actual=$(sha256sum "$teacher" | cut -d ' ' -f1)
[[ "$actual" == 7131312ffb7ecee17c33a3b52467a572d0028db9b28f829f29812cca2ece0916 ]] || { echo 'Teacher hash mismatch' >&2; exit 1; }
exec bash "$root/scripts/run.sh" student \
  --task Gd-BivtGastStudent-Rbq10-Dreamwaq-Vision-v0 \
  --teacher_checkpoint "$teacher" --num_envs 516 --bptt_steps 16 \
  --iterations "${1:-20000}" --lr .0003 --save_interval 200 \
  --camera_interval_ms 70 100 --camera_delay_ms 0 150 --camera_drop_prob .05 \
  --perception_run_name "${2:-bivt4500_gast_env516_bptt16_20000_20261001}"
