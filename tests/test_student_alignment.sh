#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${GAST_IMAGE:-/home/user/workspace/gd_lab_isaaclab.sif}"
python="${GAST_CONTAINER_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}"
command -v apptainer >/dev/null || { echo "apptainer is required" >&2; exit 2; }
test -f "$image" || { echo "GAST image not found: $image" >&2; exit 2; }
exec apptainer exec --bind "$root:/repo" "$image" "$python" /repo/tests/run_student_alignment_tests.py
