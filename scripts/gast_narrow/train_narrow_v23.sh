#!/usr/bin/env bash
# GAST-Narrow v2.3 from GAST-Narrow v2.2 update 371, one node, GPUs NARROW_GPUS (default: CUDA_VISIBLE_DEVICES or 0).
#   scripts/gast_narrow/train_narrow_v23.sh <smoke|pilot|train> [extra hydra overrides]
# - Start: v2.2 narrowv22_371.pt (00ce7051...) via --warm_start_rayir (train_narrow_v23.py): all weights copied, new PPO,
#   CENet optimizer and terrain levels restored. Terrain 44 columns (tasks_v23.py). Anchor: 6566 (NARROW_ANCHOR_*).
# - Changes (tasks_v23.py): dead end / L-bend columns, safe speed cap, shin clearance points, leg touch penalty.
# - smoke: 192 envs x 16 steps x 5 updates. pilot: NARROW_TOTAL_ENVS (4096) x 100 x NARROW_UPDATES (30, timing).
#   train: NARROW_TOTAL_ENVS (4096) x 100 x NARROW_UPDATES (400).
# - Online Top-5 every update (spacing 1) from 100, v2 score (configs/gast_narrow/online_top5_v2.json, live-reloaded).
# - Stop with a save: touch $GD_LAB_STOP_FILE (default logs/launches/<run>/STOP).
# Paths: NARROW_SOURCE / NARROW_SOURCE_SHA256, GD_LAB_SIF, GD_LAB_PYTHON (defaults below, per host).
set -euo pipefail
script="$(realpath "${BASH_SOURCE[0]}")"
root="$(cd "$(dirname "$script")/../.." && pwd)"
mode="${1:?usage: train_narrow_v23.sh smoke|pilot|train [overrides]}"
shift
gpus="${NARROW_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
[[ "$(hostname -s)" != OnVLM2* || ",$gpus," != *",3,"* ]] || { echo "GPU 3 is not ours on onvlm2" >&2; exit 2; }
nproc=$(awk -F, '{print NF}' <<<"$gpus")
case "$mode" in
  smoke) envs=192; horizon=16; updates=5; save=5 ;;
  pilot) envs=${NARROW_TOTAL_ENVS:-4096}; horizon=100; updates=${NARROW_UPDATES:-30}; save=$updates ;;
  train) envs=${NARROW_TOTAL_ENVS:-4096}; horizon=100; updates=${NARROW_UPDATES:-400}; save=100 ;;
  *) echo "bad mode: $mode" >&2; exit 2 ;;
esac
teacher="${NARROW_SOURCE:-$root/checkpoints/gast_narrow/narrowv22_371.pt}"
sha=${NARROW_SOURCE_SHA256:-00ce70515294ea9033505c40c1e295205e9070856c12c4a0d950dfffe693e677}
test -s "$teacher" || { echo "missing 371: $teacher" >&2; exit 2; }
export NARROW_ANCHOR_CKPT="${NARROW_ANCHOR_CKPT:-$root/checkpoints/gast_narrow/backgapv2_6566.pt}"
export NARROW_ANCHOR_WEIGHT=${NARROW_ANCHOR_WEIGHT:-1.0} NARROW_ANCHOR_EPOCHS=${NARROW_ANCHOR_EPOCHS:-1}
export GD_LAB_NORMALIZER_DEFER=${GD_LAB_NORMALIZER_DEFER:-1}  # normalizer moments reduced once per update (rl/distributed.py)
export NARROW_ANCHOR_GAP_NOSE=${NARROW_ANCHOR_GAP_NOSE:-1}  # anchor skips the samples gap_nose_up flags (gap tile, nose-up > limit)
test -s "$NARROW_ANCHOR_CKPT" || { echo "missing 6566 anchor: $NARROW_ANCHOR_CKPT" >&2; exit 2; }
if [[ -z "${GD_LAB_SIF:-}" ]]; then
  for c in /data/users/bsseo/gd_lab_isaaclab.sif "$HOME/workspace/gd_lab_isaaclab.sif"; do [[ -f $c ]] && GD_LAB_SIF=$c && break; done
fi
if [[ -z "${GD_LAB_PYTHON:-}" ]]; then
  for c in /data/users/bsseo/venv/bin/python "$HOME/workspace/venv/bin/python" "$HOME/workspace/venv_apptainer/bin/python"; do
    [[ -L $c || -e $c ]] && GD_LAB_PYTHON=$c && break
  done
fi
cd "$root"
export PYTHONPATH="$root/src" TRAIN_ARM=4 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$gpus" APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpus" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# onvlm2 (2026-10-19): NCCL P2P/SHM transports hit illegal memory access; socket transport on loopback works.
# 2026-10-26 (after the repair): even then, small collectives (1-element all_reduce) with the default Tree
# algorithm hit an illegal memory access; 4 MB ones and NCCL_ALGO=Ring are fine (logs/probe on onvlm2).
# NARROW_NCCL_WORKAROUND=0 (GH200 nodes): leave NCCL on its default transports (P2P/NVLink, SHM, Tree).
if [[ ${NARROW_NCCL_WORKAROUND:-1} == 1 ]]; then
  export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME="${NARROW_NCCL_IFNAME:-lo}"
  export NCCL_ALGO=${NCCL_ALGO:-Ring}
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GAST_MATH_SDPA=${GAST_MATH_SDPA:-1} GAST_NAN_DIAG=0
# No expandable_segments: the 96 GB cards do not need it (it was added for the 32 GB 90.111 card).
unset PYTORCH_CUDA_ALLOC_CONF
# 6566 (and so 1748) finished its ramps (backward share, speed ceiling, disturbance force): do not restart them.
export GD_LAB_BACKGAP_RAMP_STEPS=${GD_LAB_BACKGAP_RAMP_STEPS:-0}
export GD_LAB_V21_SPEED_RAMP_STEPS=${GD_LAB_V21_SPEED_RAMP_STEPS:-0} GD_LAB_V2_FORCE_RAMP_STEPS=${GD_LAB_V2_FORCE_RAMP_STEPS:-0}
export GAST_RUN_ID="${GAST_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_narrow_v23_371_${envs}x${horizon}_${updates}_${mode}}"
[[ "$GAST_RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid run id' >&2; exit 2; }
export GAST_VERIFY_SYNC_EVERY=100
export GD_LAB_TOP5_CRITERIA_FILE="$root/configs/gast_narrow/online_top5_v2.json"
launch="$root/logs/launches/$GAST_RUN_ID"
export GD_LAB_STOP_FILE="${GD_LAB_STOP_FILE:-$launch/STOP}"
# Multi-node (one launcher per node, e.g. srun --ntasks-per-node=1): NARROW_NNODES > 1 with NARROW_RDZV_ENDPOINT
# (host:port of node 0) and the same GAST_RUN_ID on every node; NARROW_NODE_RANK (default SLURM_NODEID) 0 writes
# the launch artefacts. NCCL between nodes: TCP sockets on NARROW_NCCL_NET_IF (default hsn0, Slingshot).
nnodes=${NARROW_NNODES:-1}
node_rank=${NARROW_NODE_RANK:-${SLURM_NODEID:-0}}
if (( nnodes > 1 )); then
  [[ -n ${NARROW_RDZV_ENDPOINT:-} ]] || { echo "NARROW_NNODES>1 needs NARROW_RDZV_ENDPOINT" >&2; exit 2; }
  # Static rendezvous (node rank from Slurm, store on node 0): the elastic c10d rendezvous dropped agents at 128
  # nodes ("failed to recv, got 0 bytes", 2026-10-28). NARROW_RDZV=c10d brings it back.
  if [[ ${NARROW_RDZV:-static} == c10d ]]; then
    dist=(--nnodes="$nnodes" --rdzv_backend=c10d --rdzv_endpoint="$NARROW_RDZV_ENDPOINT" --rdzv_id="$GAST_RUN_ID")
  else
    dist=(--nnodes="$nnodes" --node_rank="$node_rank" --master_addr="${NARROW_RDZV_ENDPOINT%:*}"
          --master_port="${NARROW_RDZV_ENDPOINT##*:}")
  fi
  export NCCL_SOCKET_IFNAME="${NARROW_NCCL_NET_IF:-hsn0}" GLOO_SOCKET_IFNAME="${NARROW_NCCL_NET_IF:-hsn0}"
  if [[ ${NARROW_NCCL_TRANSPORT:-ofi} == ofi ]]; then
    # KISTI Hangang: NCCL over Slingshot CXI with the host AWS-OFI plugin (modules aws-ofi-nccl/1.20.0 +
    # libfabric/2.3.1; what `kisti-container --nccl ofi` binds). Our SIF lacks hwloc / libnl-3 / libpciaccess: bound
    # from the host. TCP sockets on hsn0 stalled 512-rank runs (small all_reduces, 2026-10-28).
    ofi_plugin=/apps/library/aws-ofi-nccl/1.20.0/aarch64 ofi_fabric=/opt/cray/libfabric/2.3.1
    ofi_binds="$ofi_plugin:$ofi_plugin:ro,$ofi_fabric:$ofi_fabric:ro"
    for l in libcxi.so.1 libjson-c.so.5 libhugetlbfs.so liblustreapi.so.1 libmunge.so.2 libhwloc.so.15 \
             libnl-3.so.200 libpciaccess.so.0; do
      ofi_binds="$ofi_binds,$(readlink -f /usr/lib64/$l):/host/usr/lib64/$l:ro"
    done
    export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+$APPTAINER_BINDPATH,}$ofi_binds"
    export APPTAINERENV_LD_LIBRARY_PATH="$ofi_plugin/lib:$ofi_fabric/lib64:/host/usr/lib64"
    export NCCL_NET_PLUGIN=libnccl-net-ofi.so NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0
    export FI_CXI_RDZV_GET_MIN=0 FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_RX_MATCH_MODE=hybrid
  else
    export NCCL_NET=${NCCL_NET:-Socket}
  fi
  # KISTI Hangang guide (NCCL / Slingshot): LL128 gave wrong inter-node AllReduce results there; required off.
  export NCCL_PROTO=${NCCL_PROTO:-^LL128}
else
  dist=(--standalone --nnodes=1)
fi
# The robot URDF -> USD conversion writes into /tmp/IsaacLab: one directory per node, or nodes read each other's
# half-written USD ("No contact sensors added to the prim ... no rigid bodies").
usd_tmp="$root/logs/usd_tmp"
(( nnodes > 1 )) && usd_tmp="$usd_tmp/$GAST_RUN_ID/node$node_rank"
mkdir -p "$usd_tmp" "$launch"
if (( node_rank == 0 )); then
  (set -o noclobber; printf '%s\n' "$mode" > "$launch/LAUNCH_STARTED")
  sha256sum "$teacher" > "$launch/source_teacher.sha256"
  cp "$GD_LAB_TOP5_CRITERIA_FILE" "$launch/online_top5.json"
  cp "$script" "$launch/launcher.sh"
  tar --exclude='__pycache__' -czf "$launch/source.tar.gz" -C "$root" src scripts configs
  cp "$root/GIT_COMMIT" "$launch/git_head.txt" 2>/dev/null || git -C "$root" rev-parse HEAD > "$launch/git_head.txt" 2>/dev/null || true
fi
printf 'narrow v2.3 run=%s mode=%s node=%s/%s gpus=%s nproc=%s envs=%s horizon=%s updates=%s sif=%s python=%s\n' \
  "$GAST_RUN_ID" "$mode" "$node_rank" "$nnodes" "$gpus" "$nproc" "$envs" "$horizon" "$updates" "$GD_LAB_SIF" "$GD_LAB_PYTHON"
# NARROW_APPTAINER_HOME: a per-node copy of the Kit user data mounted as $HOME. With one shared ~/.local/share/ov,
# 512 processes rewrite the extension registry index at once and some read it half-written (JSONDecodeError ->
# "No module named isaacsim.asset", 2026-10-28, 128 nodes).
# Kit console at Error: every env of every rank logs USD "Unresolved reference" warnings (2 GB in 13 min on 128 nodes).
home_opt=()
[[ -n ${NARROW_APPTAINER_HOME:-} ]] && home_opt=(--home "$NARROW_APPTAINER_HOME:$HOME")
exec apptainer exec --nv --writable-tmpfs "${home_opt[@]}" --bind "$usd_tmp:/tmp/IsaacLab" "$GD_LAB_SIF" \
 "$GD_LAB_PYTHON" -m torch.distributed.run "${dist[@]}" ${NARROW_TORCHRUN_ARGS:-} \
 --nproc_per_node="$nproc" scripts/gast_narrow/train_narrow_v23.py --headless --distributed --total_envs "$envs" --seed 42 \
 --kit_args="${NARROW_KIT_ARGS:---/log/outputStreamLevel=Error}" \
 --task Gd-GastNarrowV23-Rbq10-Dreamwaq-v0 --warm_start_rayir "$teacher" --warm_start_sha256 "$sha" \
 --policy_lr 1e-4 --terrain_lr 3e-4 \
 --logger tensorboard --experiment_name gast/narrow_v23 --max_iterations "$updates" \
 agent.num_steps_per_env="$horizon" agent.algorithm.num_mini_batches=4 agent.algorithm.num_learning_epochs=5 \
 agent.algorithm.schedule=fixed agent.save_interval="$save" agent.top5_min_spacing=1 "$@"
