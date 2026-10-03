#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gast_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${gast_root}/.." && pwd)"
image="${GAST_IMAGE:-/home/user/workspace/gd_lab_isaaclab.sif}"
container_python="${GAST_CONTAINER_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}"

command -v apptainer >/dev/null || { echo "apptainer is required" >&2; exit 2; }
test -f "${image}" || { echo "GAST image not found: ${image}" >&2; exit 2; }
exec apptainer exec --bind "${repo_root}:/repo" "${image}" \
  "${container_python}" /repo/gast/scripts/run_unit_tests.py
