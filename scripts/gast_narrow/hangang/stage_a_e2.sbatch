#!/bin/bash
#SBATCH --job-name=stageA_e2
#SBATCH --partition=natStr
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --comment="field=robotics;appl=pytorch"
#SBATCH --output=/scratch/t515a05/gd_lab/logs/%x_%j.out
# Stage A E2 + STAGE_A selection, unattended (docs/hangang/WORKORDER_STAGE_B_V25_20261031.md section 0):
#   sbatch --dependency=afterany:<last stage A job> stage_a_e2.sbatch
# 1. Brings the repository to origin/gh200 (fetched beforehand; fast-forward only) now that stage A is done.
# 2. Candidates: every best_top5/*.pt of the stage A runs + the newest model_*.pt of the newest run; 6566 = baseline.
#    They are split over the job's nodes (one E2 config per node, gap + stairs, nominal, 512 envs per cell).
# 3. Merges the trial directories into one E2 output, summarises, and runs select_stage_a.py, which writes
#    $G/logs/e2/stage_a_<jobid>/STAGE_A.txt (name, path, sha256) and stage_a_selection.md. Nothing else is started:
#    stage B is submitted after the user's check.
set -uo pipefail
G=/scratch/t515a05/gd_lab
module load apptainer
export APPTAINER_BINDPATH=/scratch,$G/nvidia_icd_gh200.json:/usr/share/vulkan/icd.d/nvidia_icd.json
export APPTAINERENV_LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:/usr/local/lib/python3.11/dist-packages/isaacsim/kit/libcarb.so
PY=$G/hostpy/bin/python
cd $G/gd_lab_vrl
echo "start $(date) nodes=$SLURM_JOB_NODELIST head=$(git log --oneline -1)"
timeout -k 10 60 git merge --ff-only origin/gh200 && echo "code: $(git log --oneline -1)"
OUT=logs/e2/stage_a_$SLURM_JOB_ID
mkdir -p $OUT/cfg
mapfile -t CANDS < <(ls logs/gast/lineage16_a/*/best_top5/*_top*.pt 2>/dev/null)
last_run=$(ls -td logs/gast/lineage16_a/*/ | head -1)
last=$(ls -t ${last_run}model_*.pt 2>/dev/null | head -1)
[[ -n $last ]] && CANDS+=("$last")
echo "candidates (${#CANDS[@]}):"; printf '  %s\n' "${CANDS[@]}"
(( ${#CANDS[@]} )) || { echo "no candidates"; exit 1; }
$PY - "$OUT" "$SLURM_NNODES" "${CANDS[@]}" <<'EOF'
import sys, yaml, re
from pathlib import Path
out, nodes, cands = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
base = yaml.safe_load(Path("configs/gast_narrow/e2/v23_check_gh200.yaml").read_text())
for k in ("narrow", "dead_end", "terrain"):
    base["suites"].get(k, {})["enabled"] = False
base["conditions"] = {"nominal": dict(base["conditions"]["nominal"], suites=["gap", "stairs"])}
base["report"] = {"baseline": "b6566", "max_drop_pt": 1.0}
names = {}
for p in cands:
    run = Path(p).parts[-3] if "best_top5" in p else Path(p).parts[-2]
    job = re.findall(r"_(\d+)$", run)
    tag = Path(p).stem.split("_")[0]
    names[f"a{job[0] if job else run[-6:]}_{tag}"] = p
items = sorted(names.items())
chunks = [items[i::nodes] for i in range(nodes)]
for i, chunk in enumerate(chunks):
    cfg = dict(base, name=f"stage_a_part{i}", out_root=str(out / "parts"),
               models=dict([("b6566", "checkpoints/gast_narrow/backgapv2_6566.pt")] + chunk))
    (out / "cfg" / f"part{i}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
print("configs:", [len(c) for c in chunks])
EOF
for i in $(seq 0 $((SLURM_NNODES - 1))); do
  srun -N1 -n1 --exclusive --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none --output=$G/logs/stageA_e2_part${i}_$SLURM_JOB_ID.out \
       $PY scripts/gast_narrow/e2/run_e2.py $OUT/cfg/part$i.yaml &
  sleep 2
done
wait
# One output: config with every model, trial directories side by side (b6566 is evaluated on each node; keep one).
mkdir -p $OUT/merged/trials
$PY - "$OUT" <<'EOF'
import sys, yaml
from pathlib import Path
out = Path(sys.argv[1])
cfgs = [yaml.safe_load(p.read_text()) for p in sorted((out / "cfg").glob("part*.yaml"))]
merged = dict(cfgs[0], name="stage_a", models={k: v for c in cfgs for k, v in c["models"].items()})
(out / "merged" / "config.yaml").write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
EOF
for d in $OUT/parts/*/trials/*__*; do
  n=$(basename $d); [[ -e $OUT/merged/trials/$n ]] || ln -s $(realpath $d) $OUT/merged/trials/$n
done
$PY scripts/gast_narrow/e2/summarize_e2.py $OUT/merged
$PY scripts/gast_narrow/e2/select_stage_a.py $OUT/merged --baseline b6566 --tol 1.0
echo "DONE $(date) out=$OUT/merged"
