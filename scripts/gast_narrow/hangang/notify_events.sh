#!/bin/bash
# Exit (and so wake the monitoring session, which reports to the user) on the next important training event:
# a watchdog PROBLEM / restart / finish / E2 line, the watchdog stopping, or the running v2.3 job passing the next
# multiple of EVERY iterations (progress report).
#   bash notify_events.sh [EVERY]
G=/scratch/t515a05/gd_lab; LOG=$G/logs/watchdog_v23.log
EVERY=${1:-100}
start=$(wc -l < "$LOG")
iteration() {  # current iteration of the running v23_main job (-1 if none yet)
  local j f; j=$(squeue -u "$USER" -h -n v23_main -o %i | head -1)
  f=$(grep -a -l "Learning iteration" $G/logs/ranklogs_$j/*/attempt_0/*/stdout.log 2>/dev/null | head -1)
  [[ -n $f ]] && grep -a "Learning iteration" "$f" | tail -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/' || echo -1
}
next=$(( ($(iteration) / EVERY + 1) * EVERY ))
while true; do
  new=$(tail -n +$((start + 1)) "$LOG" | grep -E "PROBLEM|restart|finished|E2 submitted|restart limit|cannot resume|nothing left")
  if [[ -n $new ]]; then echo "EVENT:"; echo "$new"; exit 0; fi
  if ! pgrep -f "watchdog_v23.sh" >/dev/null; then echo "EVENT: watchdog is not running"; tail -3 "$LOG"; exit 0; fi
  it=$(iteration)
  if (( it >= next )); then echo "PROGRESS REPORT: iteration $it"; exit 0; fi
  sleep 60
done
