#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export STUDENT_RESUME="$PWD/logs/vision_rbq10_dreamwaq/arm_4/cvtt7761_rvld_20000_20261001/perception_18600.pt"
bash scripts/train_cvtt7761_students_5090.sh rvld 20000 64 cvtt7761_rvld_20000_resume18600_20261001 > logs/cvtt7761_rvld_20000_resume18600_20261001.console.log 2>&1
unset STUDENT_RESUME
bash scripts/train_cvtt7761_students_5090.sh gavd 20000 64 cvtt7761_gavd_20000_20261001 > logs/cvtt7761_gavd_20000_20261001.console.log 2>&1
