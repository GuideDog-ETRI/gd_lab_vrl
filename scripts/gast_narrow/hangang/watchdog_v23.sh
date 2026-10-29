#!/bin/bash
# Overnight watchdog for a v2.3 training job: every POLL_S checks progress; a hang (no new iteration line for
# HANG_S) or a failed job is resubmitted from the newest checkpoint of the run (up to MAX_RESTARTS), excluding the
# nodes given in EXCLUDE. On a clean finish it submits the full E2 (6566, 371 and the run's Top-5) on one node.
# A node whose srun task fails to launch ("launch failed", e.g. "Error configuring interconnect") is added to EXCLUDE
# and the job is resubmitted at once from the same checkpoint (the other nodes would wait for it forever).
#   bash watchdog_v23.sh <job id> <iterations of that job> <node spec> <exclude list> <checkpoint the job started from>
set -uo pipefail
G=/scratch/t515a05/gd_lab; REPO=$G/gd_lab_vrl; JOBS=$G/jobs
JOB=$1 TOTAL=$2 NODES=$3 EXCLUDE=$4 CK=${5:-}
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
    bad=$(grep -a -o "task [0-9]* launch failed" $G/logs/v23_main_$JOB.out 2>/dev/null | head -1 | awk '{print $2}')
    if [[ -n $bad && -n $CK ]]; then
      host=$(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)" | sed -n "$((bad + 1))p")
      log "PROBLEM: job $JOB task $bad on $host failed to launch: $(grep -a 'launch failed' $G/logs/v23_main_$JOB.out | head -1)"
      scancel "$JOB"; sleep 20; EXCLUDE="$EXCLUDE,$host"
      JOB=$(submit "$NODES" "$CK" "$TOTAL"); log "restart (launch failure): job $JOB from $CK, exclude $EXCLUDE"
      last_line="" last_change=$(date +%s); sleep "$POLL_S"; continue
    fi
    # Straggler (needs NARROW_HEARTBEAT_ALL=1): a rank that never logged a rollout step while the other ranks of its node
    # reached update-start, STRAGGLER_S after start -> its node is excluded and the job resubmitted.
    if [[ -n $CK && -z $(rank_log "$JOB") ]] && (( $(date +%s) - last_change > ${STRAGGLER_S:-480} )); then
      host=""
      for d in $G/logs/ranklogs_$JOB/*/attempt_0; do
        for lr in 0 1 2 3; do
          [[ -f $d/$lr/stdout.log ]] || continue
          if ! grep -a -q "rollout step" "$d/$lr/stdout.log" && grep -a -q "update-start" $d/*/stdout.log; then
            host=$(grep -a -h -o "rank [0-9]* node[0-9]*" $d/*/stdout.log | head -1 | awk '{print $3}'); break 2
          fi
        done
      done
      if [[ -n $host ]]; then
        log "PROBLEM: job $JOB has a rank on $host that never finished a rollout step (others are waiting)"
        scancel "$JOB"; sleep 20; EXCLUDE="$EXCLUDE,$host"
        JOB=$(submit "$NODES" "$CK" "$TOTAL"); log "restart (stalled rank): job $JOB from $CK, exclude $EXCLUDE"
        last_line="" last_change=$(date +%s); sleep "$POLL_S"; continue
      fi
    fi
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
  NEWCK=$(ls -t "$REPO/logs/gast/narrow_v23/$RUN"/model_*.pt "$REPO/logs/gast/narrow_v23/$RUN"/best_top5/*_top*.pt 2>/dev/null \
          | awk -F/ '{n=$NF; sub(/^model_/,"",n); sub(/_top[0-9]\.pt$|\.pt$/,"",n); print n+0, $0}' | sort -n | tail -1)
  if [[ -n $NEWCK ]]; then DONE=${NEWCK%% *}; CK=${NEWCK#* }
  elif [[ -n ${CK:-} ]]; then DONE=0; log "no checkpoint in $RUN; reusing $CK"
  else log "no checkpoint in $RUN; cannot resume"; exit 1; fi
  LEFT=$(( TOTAL - DONE ))
  (( LEFT > 0 )) || { log "nothing left"; exit 0; }
  restarts=$((restarts + 1))
  JOB=$(submit "$NODES" "$CK" "$LEFT"); TOTAL=$LEFT
  log "restart $restarts: job $JOB from $CK ($LEFT iterations left, nodes $NODES)"
  last_line="" last_change=$(date +%s); sleep "$POLL_S"
done
