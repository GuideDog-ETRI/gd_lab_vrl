#!/usr/bin/env bash
# Future authorized training launcher; UPDATE160B must NOT execute this.
set -euo pipefail
: "${GAST_NAV_TRAINING_AUTHORIZED:?Explicit separate training authorization required}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
exec python -B -m torch.distributed.run --standalone --nproc_per_node=3 scripts/gast_nav/train.py \
  --train --distributed --total_envs 4096 --headless "$@"
