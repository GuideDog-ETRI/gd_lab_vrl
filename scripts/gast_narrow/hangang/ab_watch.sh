#!/bin/bash
# Watch the four 8-node PPO-setting runs (logs/ab_jobs.txt): exit early on a launch failure or a run ending without
# GAST_COMPLETE, else when all four finished; prints a side-by-side summary either way.
G=/scratch/t515a05/gd_lab
eval "$(sed 's/ /\n/g' $G/logs/ab_jobs.txt)"   # a=... b=... c=... d=...
declare -A J=([a]=$a [b]=$b [c]=$c [d]=$d)
rank_log() { grep -a -l "Learning iteration" $G/logs/ranklogs_$1/*/attempt_0/*/stdout.log 2>/dev/null | head -1; }
summary() {
  for k in a b c d; do
    j=${J[$k]}; f=$(rank_log $j); st=$(sacct -j $j -X -n -o State | head -1 | tr -d ' ')
    [[ -z $f ]] && { echo "($k) job $j $st: no iteration yet"; continue; }
    it=$(grep -a "Learning iteration" $f | tail -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
    t=$(grep -a "Computation:" $f | tail -10 | sed -E 's/.*collection: ([0-9.]+)s, learning ([0-9.]+)s.*/\1 \2/' | awk '{c+=$1;l+=$2;n++} END{printf "%.1f+%.1f", c/n, l/n}')
    v() { grep -a "$1" $f | tail -5 | awk -v F="$2" '{s+=$NF;n++} END{printf F, s/n}'; }
    echo "($k) job $j $st it=$it time=${t}s reward=$(v 'Mean reward:' %.2f) len=$(v 'Mean episode length:' %.0f)" \
         "contact=$(v 'Episode_Termination/base_contact:' %.3f) pass=$(v 'Episode_Reward/narrow_success:' %.3f)" \
         "hold=$(v 'Episode_Reward/narrow_hold:' %.3f) gapclean=$(v 'Episode_Reward/platform_gap_clean:' %.4f)" \
         "std=$(v 'Mean action noise std:' %.3f)"
  done
}
reported_start=0; [[ ${1:-} == end ]] && reported_start=1
while true; do
  for k in a b c d; do
    j=${J[$k]}
    if grep -a -q "launch failed" $G/logs/v23_ab_${k}_$j.out 2>/dev/null; then
      echo "EVENT: ($k) job $j launch failure: $(grep -a 'launch failed' $G/logs/v23_ab_${k}_$j.out | head -1)"; summary; exit 0; fi
    if ! squeue -h -j $j | grep -q . && ! grep -a -q GAST_COMPLETE $(rank_log $j) 2>/dev/null; then
      echo "EVENT: ($k) job $j ended without completing"; summary; exit 0; fi
  done
  if (( ! reported_start )); then
    n=0; for k in a b c d; do [[ -n $(rank_log ${J[$k]}) ]] && n=$((n+1)); done
    if (( n == 4 )); then echo "EVENT: all four runs reached their first iteration"; summary; exit 0; fi
  fi
  running=0; for k in a b c d; do squeue -h -j ${J[$k]} | grep -q . && running=1; done
  (( running )) || { echo "EVENT: all four runs finished"; summary; exit 0; }
  sleep 60
done
