#!/usr/bin/env bash
# Serve the repository so the viewer can read the RBQ10 URDF and meshes, then open a recording.
#   tools/rl_replay/serve.sh [recording.json, relative to the repo root] [port=8765]
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
port="${2:-8765}"
url="http://127.0.0.1:${port}/tools/rl_replay/viewer.html"
[[ -n "${1:-}" ]] && url="${url}?data=../../${1}"
echo "viewer: $url  (Ctrl+C to stop)"
(sleep 1; command -v xdg-open >/dev/null && xdg-open "$url" >/dev/null 2>&1 || true) &
cd "$repo" && exec python3 -m http.server "$port" --bind 127.0.0.1
