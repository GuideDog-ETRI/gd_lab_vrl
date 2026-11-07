#!/bin/bash
# Move the running v2.3 64-node training to 128 nodes once natStr can start 128 nodes now; fall back to 64 nodes
# (from the same checkpoint) if the 128-node run does not reach its first update in time.
#   bash switch_to_128.sh <64-node job id> <total updates>   (runs on the login node, logs to stdout)
set -uo pipefail
G=/scratch/t515a05/gd_lab
REPO=$G/gd_lab_vrl
JOBS=$G/jobs
JOB=$1 TOTAL=$2
POLL_S=${POLL_S:-300}            # how often natStr is checked
START_WAIT_S=${START_WAIT_S:-1800}  # 128-node job must start within this
FIRST_UPDATE_S=${FIRST_UPDATE_S:-2700}  # ... and log its first update within this after starting
log() { echo "[switch $(date +%m-%d_%H:%M:%S)] $*"; }
run_id() { basename "$(ls -d "$REPO"/logs/launches/*_"$1" 2>/dev/null | head -1)"; }  # GAST_RUN_ID ends in the job id
alive() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
can_start_128() {  # sbatch --test-only: would a 128-node job start within 5 minutes?
  local at
  at=$(cd $JOBS && sbatch --test-only -N 128 train_v23_main.sbatch 2>&1 | grep -o 'to start at [^ ]*' | awk '{print $4}')
  [[ -n $at ]] && (( $(date -d "$at" +%s) <= $(date +%s) + 300 ))
}
latest_saved() {  # newest model_<n>.pt of a run directory, and n
  ls -t "$REPO/logs/gast/narrow_v23/$1"/model_*.pt 2>/dev/null | head -1
}
submit() {  # nodes, checkpoint, updates -> job id
  local sha; sha=$(sha256sum "$2" | cut -d' ' -f1)
  (cd $JOBS && NARROW_SOURCE="$2" NARROW_SOURCE_SHA256=$sha NARROW_UPDATES=$3 \
     NARROW_STACK_DUMP_S=600 NARROW_STACK_DUMP_RANKS=0,1,511 sbatch --parsable -N "$1" train_v23_main.sbatch)
}

RUN=$(run_id "$JOB")
log "watching job $JOB ($RUN), total $TOTAL updates"
until can_start_128; do
  alive "$JOB" || { log "job $JOB ended before 128 nodes were free; nothing to do"; exit 0; }
  sleep "$POLL_S"
done
log "128 nodes available: stopping $JOB with a save"
touch "$REPO/logs/launches/$RUN/STOP"
while alive "$JOB"; do sleep 20; done
CK=$(latest_saved "$RUN"); DONE=$(basename "$CK" .pt | sed 's/model_//')
LEFT=$(( TOTAL - DONE ))
log "saved $CK (update $DONE); $LEFT updates left"
(( LEFT > 0 )) || { log "nothing left to train"; exit 0; }

J128=$(submit 128 "$CK" "$LEFT"); log "submitted 128-node job $J128"
t=0; until [[ $(squeue -h -j "$J128" -o %T) == RUNNING ]] || (( t > START_WAIT_S )); do sleep 30; t=$((t+30)); done
ok=0
if [[ $(squeue -h -j "$J128" -o %T) == RUNNING ]]; then
  OUT=$G/logs/v23_main_$J128.out; t=0
  until (( t > FIRST_UPDATE_S )); do
    tail -c 5000000 "$OUT" 2>/dev/null | grep -a -q "Learning iteration" && { ok=1; break; }
    alive "$J128" || break
    sleep 60; t=$((t+60))
  done
fi
if (( ok )); then log "128-node job $J128 is training"; exit 0; fi
log "128-node job $J128 did not reach an update (start ${START_WAIT_S}s / first update ${FIRST_UPDATE_S}s): back to 64"
scancel "$J128"; sleep 30
J64=$(submit 64 "$CK" "$LEFT"); log "submitted 64-node job $J64 from $CK ($LEFT updates)"
