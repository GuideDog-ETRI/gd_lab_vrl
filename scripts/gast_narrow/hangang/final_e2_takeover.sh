#!/bin/bash
# Wait for the watchdog's single-node final E2, cancel it and submit the split version instead.
G=/scratch/t515a05/gd_lab
until grep -q "E2 submitted: job" $G/logs/watchdog_v23.log; do
  sleep 60
done
j=$(grep "E2 submitted: job" $G/logs/watchdog_v23.log | tail -1 | sed -E 's/.*job ([0-9]+).*/\1/')
scancel "$j"; echo "cancelled watchdog E2 job $j"
run=$(basename "$(ls -d $G/gd_lab_vrl/logs/gast/narrow_v23/*_1173553 | head -1)")
bash $G/jobs/final_e2_split.sh "$run"
