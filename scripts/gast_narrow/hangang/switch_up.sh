#!/bin/bash
# Stop a running v2.3 job with a save and continue it on more nodes; fall back to FALLBACK_NODES (same checkpoint)
# when the bigger job does not start within START_WAIT_S or logs no iteration within FIRST_UPDATE_S of starting.
#   bash switch_up.sh <job id> <updates of that job> <node spec, e.g. 200-237> <fallback nodes>
set -uo pipefail
G=/scratch/t515a05/gd_lab; REPO=$G/gd_lab_vrl; JOBS=$G/jobs
JOB=$1 TOTAL=$2 NODES=$3 FALLBACK=$4
START_WAIT_S=${START_WAIT_S:-1200} FIRST_UPDATE_S=${FIRST_UPDATE_S:-1800}
log() { echo "[switch $(date +%m-%d_%H:%M:%S)] $*"; }
alive() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
RUN=$(basename "$(ls -d "$REPO"/logs/launches/*_"$JOB" | head -1)")
submit() {  # nodes, checkpoint, updates -> job id
  local sha; sha=$(sha256sum "$2" | cut -d' ' -f1)
  (cd $JOBS && NARROW_SOURCE="$2" NARROW_SOURCE_SHA256=$sha NARROW_UPDATES=$3 sbatch --parsable -N "$1" train_v23_main.sbatch)
}
first_iteration() { grep -a -l "Iteration time" $G/logs/ranklogs_$1/*/attempt_0/*/stdout.log 2>/dev/null | head -1; }
log "stopping $JOB ($RUN) with a save"
touch "$REPO/logs/launches/$RUN/STOP"
while alive "$JOB"; do sleep 15; done
CK=$(ls -t "$REPO/logs/gast/narrow_v23/$RUN"/model_*.pt | head -1); DONE=$(basename "$CK" .pt | sed 's/model_//')
LEFT=$(( TOTAL - DONE )); log "saved $CK (iteration $DONE of $TOTAL); $LEFT left"
NEW=$(submit "$NODES" "$CK" "$LEFT"); log "submitted $NODES-node job $NEW"
t=0; until [[ $(squeue -h -j "$NEW" -o %T) == RUNNING ]] || ! alive "$NEW" || (( t > START_WAIT_S )); do sleep 20; t=$((t+20)); done
log "job $NEW: $(squeue -h -j "$NEW" -o '%T %D nodes')"
ok=0
if [[ $(squeue -h -j "$NEW" -o %T) == RUNNING ]]; then
  t=0; until (( t > FIRST_UPDATE_S )); do f=$(first_iteration "$NEW"); [[ -n $f ]] && { ok=1; break; }; alive "$NEW" || break; sleep 30; t=$((t+30)); done
fi
if (( ok )); then log "job $NEW is training: $(grep -a 'Iteration time' "$f" | head -2 | tr -s ' ' | tr '\n' ';')"; log "rank log $f"; exit 0; fi
log "job $NEW did not reach an iteration: back to $FALLBACK nodes"; scancel "$NEW"; sleep 30
FB=$(submit "$FALLBACK" "$CK" "$LEFT"); log "submitted $FALLBACK-node fallback $FB from $CK ($LEFT iterations)"
