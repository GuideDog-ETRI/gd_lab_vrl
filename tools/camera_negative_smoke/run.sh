#!/usr/bin/env bash
# Entry-path negative smoke for the camera-contract checks (needs Isaac/GPU; nothing trains past 3 captures).
#   tools/camera_negative_smoke/run.sh [teacher.pt]
# Real entry points, --student_resume / --student_checkpoint on fixtures that MUST be refused (missing contract,
# other calibration) and on a matching control that MUST get past the check:
#   distill_student.py (RVLD, GAVD) resume, gast train_student.py resume, gast train_student_live.py resume,
#   play_student.py (RVLD) evaluation.
# Runs use the legacy calibration because the 17206 teacher was trained with it (GD_LAB_ALLOW_LEGACY_CAMERA=1);
# GD_LAB_ALLOW_MISSING_CAMERA_CONTRACT is deliberately unset.
# Verdict per case: refusal = non-zero exit AND the exact refusal message; control = exit 0 AND the exact
# success message. Exit status = number of failed cases (0-99); setup errors use 100+ so they never look like
# a case count: 100 = bad teacher path, 101 = fixtures could not be built, 102 = evaluation teacher copy failed.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
teacher=$(realpath "${1:-$root/checkpoints/teachers/bivt/ray_top1_17206_20261003/teacher/17206_top1.pt}") || exit 100
test -s "$teacher" || { echo "missing teacher $teacher" >&2; exit 100; }
image=${GD_LAB_SIF:-/home/user/workspace/gd_lab_isaaclab.sif}
python=${GD_LAB_PYTHON:-/home/user/workspace/venv_apptainer/bin/python}
out="$root/logs/camera_negative_smoke/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
export TRAIN_ARM=4 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 GD_LAB_ALLOW_LEGACY_CAMERA=1
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}"
unset GD_LAB_ALLOW_MISSING_CAMERA_CONTRACT
in_container() { timeout 1200 apptainer exec --nv --writable-tmpfs "$image" "$python" "$@"; }

(cd "$root" && PYTHONPATH="$root/src" in_container tools/camera_negative_smoke/make_fixtures.py main "$teacher" "$out/fixtures") \
  || { echo "FAIL fixtures (main)" >&2; exit 101; }
(cd "$root" && PYTHONPATH="$root/src" in_container tools/camera_negative_smoke/make_fixtures.py gast "$teacher" "$out/fixtures") \
  || { echo "FAIL fixtures (gast)" >&2; exit 101; }
for f in rvld gavd gast; do for c in missing mismatch match; do
  test -s "$out/fixtures/${f}_${c}.pt" || { echo "FAIL fixture ${f}_${c}.pt missing" >&2; exit 101; }
done; done

# play_student.py finds its teacher as <LOG_ROOT>/<experiment>/<run>/<checkpoint>: build an isolated one.
play_root="$out/logroot"
mkdir -p "$play_root/camsmoke/teacher" && cp "$teacher" "$play_root/camsmoke/teacher/model_17206.pt" \
  && cmp -s "$teacher" "$play_root/camsmoke/teacher/model_17206.pt" || { echo "FAIL evaluation teacher copy" >&2; exit 102; }

failures=0
expect() {  # name kind(refuse|pass) exact_message command...
  local name=$1 kind=$2 message=$3 status ok=0; shift 3
  "$@" > "$out/$name.log" 2>&1; status=$?   # "$@" may be a shell function; the timeout lives in in_container
  if grep -Fq -- "$message" "$out/$name.log"; then
    if [ "$kind" = pass ] && [ "$status" -eq 0 ]; then ok=1; fi
    if [ "$kind" = refuse ] && [ "$status" -ne 0 ]; then ok=1; fi
  fi
  if [ "$ok" -eq 1 ]; then
    echo "PASS $name (exit $status)"
  else
    echo "FAIL $name (exit $status, want $kind + '$message', see $out/$name.log)"; failures=$((failures + 1))
  fi
}
MISSING="checkpoint has no camera contract, so its calibration cannot be verified for"
DIFFERS="checkpoint camera contract differs from the one required for"
GAST_REFUSED="Resume camera calibration/timing contract differs from this run"
RESUMED="[INFO] Student/optimizer resumed at 1;"
common=(--headless --device cuda:0 --num_envs 2 --iterations 3 --save_interval 1000 --teacher_checkpoint "$teacher")

distill() { (cd "$root" && PYTHONPATH="$root/src" in_container scripts/distill_student.py --task Gd-VrlRayStudent-Rbq10-Dreamwaq-Vision-v0 \
               "${common[@]}" --student_arch "$2" --student_resume "$out/fixtures/$1.pt" --perception_run_name "camsmoke_$1" \
               env.camera_profile=vendor_legacy); }
for arch in rvld:cnn_gru gavd:grid_attention_v1; do
  n=${arch%%:*} a=${arch#*:}
  expect "distill_${n}_missing"  refuse "$MISSING resume" distill "${n}_missing"  "$a"
  expect "distill_${n}_mismatch" refuse "$DIFFERS resume" distill "${n}_mismatch" "$a"
  expect "distill_${n}_match"    pass   "$RESUMED"        distill "${n}_match"    "$a"
done

gast() { (cd "$root" && PYTHONPATH="$root/src" in_container "scripts/gast/$1" --task Gd-BivtGastStudent-Rbq10-Dreamwaq-Vision-v0 \
            "${common[@]}" --student_resume "$out/fixtures/gast_$2.pt" --perception_run_name "camsmoke_${1%.py}_$2" \
            env.camera_profile=vendor_legacy); }
for script in train_student.py train_student_live.py; do
  expect "${script%.py}_missing"  refuse "$GAST_REFUSED" gast "$script" missing
  expect "${script%.py}_mismatch" refuse "$GAST_REFUSED" gast "$script" mismatch
  expect "${script%.py}_match"    pass   "$RESUMED"      gast "$script" match
done

play() { (cd "$root" && GD_LAB_LOG_ROOT="$play_root" PYTHONPATH="$root/src" in_container scripts/play_student.py \
            --task Gd-VrlRayStudent-Rbq10-Dreamwaq-Vision-v0 --headless --device cuda:0 --num_envs 2 --max_steps 5 \
            --experiment_name camsmoke --load_run teacher --checkpoint model_17206.pt \
            --student_checkpoint "$out/fixtures/rvld_$1.pt" env.camera_profile=vendor_legacy); }
expect play_rvld_missing  refuse "$MISSING evaluation" play missing
expect play_rvld_mismatch refuse "$DIFFERS evaluation" play mismatch
expect play_rvld_match    pass   "[INFO] Student checkpoint:" play match

echo "logs: $out  failures=$failures"
[ "$failures" -gt 99 ] && failures=99
exit "$failures"
