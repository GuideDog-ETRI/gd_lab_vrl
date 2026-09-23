#!/usr/bin/env bash
# Run inside tmux; write a persistent console log and exit status.
set -uo pipefail

train_arm="$1"
max_iterations="$2"
log_file="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

"$repo_root/scripts/train_vrl_3gpu.sh" "$train_arm" "$max_iterations" 2>&1 | tee -a "$log_file"
run_status=${PIPESTATUS[0]}
printf '%s exit=%s\n' "$(date -Is)" "$run_status" | tee "$log_file.exit"
exit "$run_status"
