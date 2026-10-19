#!/usr/bin/env bash
# UPDATE162: pilot is the only currently authorized real Nav training scope.
set -euo pipefail
# One diagnostic core owner is selected in train.py; other ranks disable cores.
ulimit -c unlimited
mode="${1:?usage: train_gast_nav_3gpu.sh pilot|train|warmup [train.py options]}"
shift
case "$mode" in
  warmup) mode_flag=--warmup ;;
  pilot) export GAST_NAV_PILOT=1; mode_flag=--pilot ;;
  train) test "${GAST_NAV_TRAINING_AUTHORIZED:-0}" = 1 || { echo 'training authorization required' >&2; exit 2; }; mode_flag=--train ;;
  *) echo "unsupported mode: $mode" >&2; exit 2 ;;
esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
# GAST_NAV_GPUS: visible GPUs for train mode (default 0,1,2; never GPU 3 on onvlm2). Pilot stays 0,1,2.
gpus="${GAST_NAV_GPUS:-0,1,2}"
[[ "$gpus" =~ ^[0-2](,[0-2])*$ ]] || { echo 'GAST_NAV_GPUS must list GPUs 0-2' >&2; exit 2; }
[[ "$mode" = train || "$gpus" = 0,1,2 ]] || { echo 'only train mode may change GPUs' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=$gpus APPTAINERENV_CUDA_VISIBLE_DEVICES=$gpus
export NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# onvlm2 2026-10-19: NCCL P2P and SHM transports fail with illegal memory access even in a bare
# broadcast test; socket transport works. GAST_NAV_NCCL_SOCKET=1 (default) forces it.
if [[ "${GAST_NAV_NCCL_SOCKET:-1}" = 1 ]]; then export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1; fi
export TRAIN_ARM=4 GAST_MATH_SDPA=1 OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$root/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y%m%d_%H%M%S)_nav_${mode}}"
[[ "$GAST_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid run id' >&2; exit 2; }
launch="$root/logs/launches/$GAST_RUN_ID"
mkdir "$launch" -p
# A run id cannot silently overwrite a previous launch.
(set -o noclobber; printf '%s\n' "$mode" > "$launch/LAUNCH_STARTED")
cp "${BASH_SOURCE[0]}" "$launch/launcher.sh"
cp -r configs/gast_nav "$launch/configs"
tar --exclude='__pycache__' -czf "$launch/source.tar.gz" src scripts configs
sha256sum "$launch/source.tar.gz" > "$launch/source.sha256"
sha256sum checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt > "$launch/source_teacher.sha256"
printf '%q ' "$mode" "$@" > "$launch/arguments.txt"
mkdir -p "$launch/runtime/tmp" "$launch/runtime/usd" "$launch/runtime/cache" "$launch/runtime/mpl"
export TMPDIR="$launch/runtime/tmp" APPTAINER_TMPDIR="$launch/runtime/tmp" XDG_CACHE_HOME="$launch/runtime/cache" MPLCONFIGDIR="$launch/runtime/mpl"
export GAST_TRITON_CACHE_ROOT="$root/logs/gast/nav_v1/$GAST_RUN_ID/triton_cache"
export TRITON_CACHE_DIR="$GAST_TRITON_CACHE_ROOT/warmup"
# Run-local CUDA cache stalls NCCL initialization on onvlm2; use driver default.
unset CUDA_CACHE_PATH
export TORCH_HOME="$launch/runtime/cache/torch" NUMBA_CACHE_DIR="$launch/runtime/cache/numba"
mkdir -p "$launch/runtime/home" "$TRITON_CACHE_DIR"
if [[ "${GAST_NAV_MAP_BACKEND:-triton}" != torch ]]; then
# A fresh single-process compile precedes torchrun; bounded and inference-only.
CUDA_VISIBLE_DEVICES=0 APPTAINERENV_CUDA_VISIBLE_DEVICES=0 \
 timeout --signal=TERM --kill-after=5s 30s apptainer exec --nv --writable-tmpfs \
  "${GD_LAB_SIF:?set GD_LAB_SIF}" \
 "${GD_LAB_PYTHON:?set GD_LAB_PYTHON}" -B -m gd_lab.gast_nav.pilot_runner --warmup \
 > "$root/logs/gast/nav_v1/$GAST_RUN_ID/warmup.log" 2>&1
for rank in $(seq 0 $(( $(awk -F, "{print NF}" <<<"$gpus") - 1 ))); do
 mkdir -p "$GAST_TRITON_CACHE_ROOT/$rank"
 cp -a "$GAST_TRITON_CACHE_ROOT/warmup/." "$GAST_TRITON_CACHE_ROOT/$rank/"
done
else
 printf '%s\n' '{"backend":"torch","triton_warmup":"skipped"}' > "$root/logs/gast/nav_v1/$GAST_RUN_ID/warmup.log"
fi
if [[ "$mode" = warmup ]]; then exit 0; fi
nproc=$(awk -F, "{print NF}" <<<"$gpus")
if [[ "${GAST_NAV_SINGLE_RANK_SMOKE:-0}" = 1 ]]; then
 [[ "$mode" = pilot && " $* " = *" --pilot-smoke "* ]] || exit 2
 nproc=1
 export CUDA_VISIBLE_DEVICES=0 APPTAINERENV_CUDA_VISIBLE_DEVICES=0
fi
# Reassert in the very shell that execs Apptainer, with one shared core path.
cd "$root/logs/gast/nav_v1/$GAST_RUN_ID"
ulimit -c unlimited
printf 'core_limit=%s cwd=%s\n' "$(ulimit -c)" "$PWD" > core_setup.txt
exec apptainer exec --nv --writable-tmpfs  --bind "$launch/runtime/usd:/tmp/IsaacLab" \
  "${GD_LAB_SIF:?set GD_LAB_SIF}" "${GD_LAB_PYTHON:?set GD_LAB_PYTHON}" -B \
  -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$nproc" \
  "$root/scripts/gast_nav/train.py" "$mode_flag" --distributed --total_envs 4096 --headless "$@"
