#!/bin/bash
# Overnight watchdog for a v2.3 training job: every POLL_S checks progress; a hang (no new iteration line for
# HANG_S) or a failed job is resubmitted from the newest checkpoint of the run (up to MAX_RESTARTS), excluding the
# nodes given in EXCLUDE. On a clean finish it submits the full E2 (6566, 371 and the run's Top-5) on one node.
#   bash watchdog_v23.sh <job id> <iterations of that job> <node spec> <exclude list>
set -uo pipefail
G=/scratch/t515a05/gd_lab; REPO=$G/gd_lab_vrl; JOBS=$G/jobs
JOB=$1 TOTAL=$2 NODES=$3 EXCLUDE=$4
POLL_S=${POLL_S:-300} HANG_S=${HANG_S:-1200} MAX_RESTARTS=${MAX_RESTARTS:-2}
restarts=0
log() { echo "[watchdog $(date +%m-%d_%H:%M:%S)] $*"; }
alive() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
run_of() { basename "$(ls -d "$REPO"/logs/launches/*_"$1" 2>/dev/null | head -1)"; }
rank_log() { grep -a -l "Learning iteration" $G/logs/ranklogs_$1/*/attempt_0/*/stdout.log 2>/dev/null | head -1; }
submit() {  # nodes, checkpoint, iterations -> job id
  local sha; sha=$(sha256sum "$2" | cut -d' ' -f1)
  (cd $JOBS && NARROW_SOURCE="$2" NARROW_SOURCE_SHA256=$sha NARROW_UPDATES=$3 \
     sbatch --parsable -N "$1" --exclude="$EXCLUDE" train_v23_main.sbatch)
}
start_e2() {  # run dir name -> E2 job with 6566, 371 and the run's Top-5
  local run=$1 cfg=$REPO/configs/gast_narrow/e2/v23_final_gh200.yaml
  local top; top=$(ls "$REPO/logs/gast/narrow_v23/$run/best_top5/"*_top*.pt 2>/dev/null | sort -t_ -k2)
  if [[ -z $top ]]; then  # no Top-5 (online gates not met): the periodic checkpoints, every 200 + the last
    top=$(ls "$REPO/logs/gast/narrow_v23/$run"/model_*.pt | sort -V | awk -F'model_' '{n=$2+0} n>0 && n%200==0' ; \
          ls "$REPO/logs/gast/narrow_v23/$run"/model_*.pt | sort -V | tail -1)
    top=$(echo "$top" | sort -u -V | tail -6)
    log "no Top-5 in $run; E2 on periodic checkpoints: $(echo $top | xargs -n1 basename | tr '\n' ' ')"
  fi
  python3 - "$REPO/configs/gast_narrow/e2/v23_check_gh200.yaml" "$cfg" "$REPO" $top <<'EOF'
import re, sys, os
src, dst, repo, tops = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
s = open(src).read().replace("name: v23_check_gh200", "name: v23_final_gh200")
lines = ["models:                       # 6566 (gap reference), 371 (v2.3 start), v2.3 final Top-5",
         "  b6566: checkpoints/gast_narrow/backgapv2_6566.pt",
         "  v22u371: checkpoints/gast_narrow/narrowv22_371.pt"]
for t in tops:
    it = re.match(r"(?:model_)?(\d+)", os.path.basename(t)).group(1)
    lines.append(f"  v23u{it}: {os.path.relpath(t, repo)}")
s = re.sub(r"models:.*?\n\n", "\n".join(lines) + "\n\n", s, count=1, flags=re.S)
open(dst, "w").write(s)
EOF
  local e2; e2=$(cd $JOBS && E2_CONFIG=v23_final_gh200 sbatch --parsable e2_full.sbatch)
  log "E2 submitted: job $e2 (config $cfg)"
}

log "watching job $JOB ($TOTAL iterations, nodes $NODES, exclude $EXCLUDE)"
last_line="" last_change=$(date +%s)
while true; do
  RUN=$(run_of "$JOB")
  if ! alive "$JOB"; then
    state=$(sacct -j "$JOB" -X -n -o State | head -1 | tr -d ' ')
    f=$(rank_log "$JOB")
    if [[ -n $f ]] && grep -a -q "GAST_COMPLETE" "$f"; then
      log "job $JOB finished ($state): $(grep -a 'Learning iteration' "$f" | tail -1 | tr -s ' ')"
      start_e2 "$RUN"; exit 0
    fi
    reason="job ended without completing ($state)"
  else
    f=$(rank_log "$JOB")
    line=$([[ -n $f ]] && grep -a "Learning iteration" "$f" | tail -1 | tr -s ' ')
    now=$(date +%s)
    if [[ -n $line && $line != "$last_line" ]]; then last_line=$line; last_change=$now
      (( 10#$(date +%M) < POLL_S / 60 )) && log "progress: $line | $(grep -a 'ETA' "$f" | tail -1 | tr -s ' ')"
      sleep "$POLL_S"; continue
    fi
    (( now - last_change < HANG_S )) && { sleep "$POLL_S"; continue; }
    reason="no new iteration for $(( (now - last_change) / 60 )) min (last: ${last_line:-none})"
    scancel "$JOB"; sleep 30
  fi
  log "PROBLEM: $reason"
  (( restarts < MAX_RESTARTS )) || { log "restart limit reached; leaving it stopped"; exit 1; }
  CK=$(ls -t "$REPO/logs/gast/narrow_v23/$RUN"/model_*.pt 2>/dev/null | head -1)
  [[ -n $CK ]] || { log "no checkpoint in $RUN; cannot resume"; exit 1; }
  DONE=$(basename "$CK" .pt | sed 's/model_//'); LEFT=$(( TOTAL - DONE ))
  (( LEFT > 0 )) || { log "nothing left"; exit 0; }
  restarts=$((restarts + 1))
  JOB=$(submit "$NODES" "$CK" "$LEFT"); TOTAL=$LEFT
  log "restart $restarts: job $JOB from $CK ($LEFT iterations left, nodes $NODES)"
  last_line="" last_change=$(date +%s); sleep "$POLL_S"
done
