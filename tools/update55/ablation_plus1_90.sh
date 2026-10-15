#!/usr/bin/env bash
# Prepared only: one independent ablation, no automatic sweep.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$root/run_v3_update55_90.sh" "${1:-small}" "${2:-42}" async
