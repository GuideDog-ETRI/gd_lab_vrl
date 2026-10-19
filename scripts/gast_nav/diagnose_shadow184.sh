#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
out="$root/logs/gast/nav_v1/${GAST_RUN_ID:?set unique GAST_RUN_ID}"
mkdir -p "$out/runtime/tmp" "$out/runtime/cache" "$out/runtime/mpl" "$out/runtime/usd"
export OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 APPTAINERENV_CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="$root/logs/gast/nav_v1/update162_cuda_tests/runtime/cache" CUDA_CACHE_PATH="$out/runtime/cache/cuda"
export TRAIN_ARM=4 GAST_MATH_SDPA=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$out/runtime/tmp" APPTAINER_TMPDIR="$out/runtime/tmp" XDG_CACHE_HOME="$out/runtime/cache" MPLCONFIGDIR="$out/runtime/mpl"
export NUMBA_CACHE_DIR="$out/runtime/cache/numba" TORCH_HOME="$out/runtime/cache/torch" PYTHONPATH="$root/src:/home/user/workspace/IsaacLab/source/isaaclab"
cp configs/gast_nav/environment.yaml "$out/config.yaml"
python3 - "$root" "$out" <<'MANIFEST183'
from pathlib import Path
import sys,hashlib,json
root,out=map(Path,sys.argv[1:])
files=[]
for folder in ('src/gd_lab/gast_nav','configs/gast_nav','scripts/gast_nav','tests/gast_nav'):
 files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.sh'))
(out/'source_sha.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
MANIFEST183
exec apptainer exec --nv --writable-tmpfs --bind "$out/runtime/usd:/tmp/IsaacLab" /home/user/workspace/gd_lab_isaaclab.sif \
 /home/user/workspace/venv_apptainer/bin/python -B scripts/gast_nav/diagnose_shadow184.py --headless --output "$out" "$@"
