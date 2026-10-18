#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "usage: $0 smoke|train" >&2; exit 2; fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_mapper_aug149_1gpu.sh" A2769 "${1:?smoke|train}"
