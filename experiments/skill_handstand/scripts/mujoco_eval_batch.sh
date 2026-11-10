#!/bin/bash
# mujoco_eval_batch.sh <checkpoint> <scenario> <condim 3|6> <n seeds>  -> one JSON line per seed (summary fields only)
set -u
ck=$1; sc=$2; cd_=$3; n=$4
S=${HS_PYLIB:?set HS_PYLIB to a python path with mujoco, imageio and pyopengl}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
extra=""; [ "$cd_" = 3 ] && extra="--foot_condim 3"
for seed in $(seq 1 $n); do
  timeout -k 5 900 apptainer exec /home/user/workspace/gd_lab_isaaclab.sif env PYTHONPATH=$S CUDA_VISIBLE_DEVICES= \
    /home/user/workspace/venv_apptainer/bin/python $root/scripts/mujoco_check.py --checkpoint $ck --scenario $sc --seed $seed $extra \
    --mjcf /home/user/gd_project/gd_rbq10_deploy_vrl/simulation/terrains/gap_pit65/rbq_payload.xml \
    --mesh_root /home/user/gd_project/RBQ_vendor_new/RBQ-nightly/resources 2>/dev/null | grep '"scenario"' | \
    python3 -c "import sys,json; d=json.loads(sys.stdin.read()); d.pop('joints',None); d['seed']=$seed; d['condim']=$cd_; print(json.dumps(d))" &
  (( seed % 5 == 0 )) && wait
done; wait
