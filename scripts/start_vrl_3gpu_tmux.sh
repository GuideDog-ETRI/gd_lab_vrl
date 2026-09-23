#!/usr/bin/env bash
# Start a fresh three-GPU VRL run in a detached tmux session.
set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[1-4]$ || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 ARM(1|2|3|4) MAX_ITERATIONS(positive integer)" >&2
    exit 2
fi

train_arm="$1"
max_iterations="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
session_name="vrl_arm${train_arm}_${max_iterations}"
log_file="$repo_root/logs/arm${train_arm}_${max_iterations}_$(date +%Y%m%d_%H%M%S).console.log"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available" >&2
    exit 1
fi
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Training session already exists: $session_name" >&2
    echo "Attach with: tmux attach -t $session_name" >&2
    exit 1
fi
if pgrep -f 'torch.distributed.run.*scripts/train_vrl.py.*--distributed' >/dev/null; then
    echo "A distributed VRL training process is already running; refusing to start a second one." >&2
    tmux list-sessions 2>/dev/null || true
    exit 1
fi

mkdir -p "$repo_root/logs"
printf -v tmux_command '%q ' /bin/bash "$repo_root/scripts/vrl_tmux_worker.sh" "$train_arm" "$max_iterations" "$log_file"
tmux new-session -d -s "$session_name" -c "$repo_root" "$tmux_command"
echo "Started session: $session_name"
echo "Console log: $log_file"
echo "Attach with: tmux attach -t $session_name"
echo "Detach without stopping training: Ctrl-b, then d"
