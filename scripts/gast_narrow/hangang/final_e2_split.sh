#!/bin/bash
# Final v2.3 E2 split one model per node (512 envs per cell, all conditions: ~1 h per node instead of 8+ h on one).
# Models: 6566, 371, the run's Top-5 and every periodic checkpoint (model_100 ... last).
#   bash final_e2_split.sh <run dir name> [exclude list]
set -uo pipefail
G=/scratch/t515a05/gd_lab; REPO=$G/gd_lab_vrl; RUN=$1; EXCLUDE=${2:-node1730,node1395,node1402}
D=$REPO/logs/gast/narrow_v23/$RUN; CFG=$REPO/configs/gast_narrow/e2
declare -A M=([b6566]=checkpoints/gast_narrow/backgapv2_6566.pt [v22u371]=checkpoints/gast_narrow/narrowv22_371.pt)
for t in "$D"/best_top5/*_top*.pt; do [[ -e $t ]] && M[top$(basename "$t" .pt)]=${t#$REPO/}; done
for t in "$D"/model_*.pt; do n=$(basename "$t" .pt); n=${n#model_}; (( n > 0 )) && M[v23it$n]=${t#$REPO/}; done
for k in "${!M[@]}"; do
  name=v23_final_$k
  python3 - "$CFG/v23_check_gh200.yaml" "$CFG/$name.yaml" "$name" "$k" "${M[$k]}" <<'PY'
import re, sys
src, dst, name, key, path = sys.argv[1:]
s = open(src).read().replace("name: v23_check_gh200", "name: " + name)
s = re.sub(r"models:.*?\n\n", f"models:                       # final v2.3 E2, one model per node\n  {key}: {path}\n\n", s, count=1, flags=re.S)
open(dst, "w").write(s)
PY
  j=$(cd $G/jobs && sbatch --parsable --job-name=$name --export=ALL,E2_CONFIG=$name --exclude="$EXCLUDE" e2_full.sbatch)
  echo "$k ${M[$k]} job $j"
done | tee $G/logs/final_e2_jobs.txt
