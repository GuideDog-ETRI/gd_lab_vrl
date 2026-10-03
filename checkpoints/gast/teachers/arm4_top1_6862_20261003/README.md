# GAST teacher checkpoint: iteration 6862

This directory contains the **GAST teacher** checkpoint selected as the current online Top-1 at packaging time. It is provided for a separate GAST student-distillation experiment; it is not a student model and is not a deployment-ready ONNX/package.

## Contents

- `teacher/6862_top1.pt` — teacher checkpoint copied from the GAST run's `best_top5/`.
- `params/agent.yaml`, `params/env.yaml` — exact run parameters captured for the teacher run.
- `selection/leaderboard.json` — leaderboard snapshot containing the selected entry and contemporaneous ranking.
- `selection/online_top5.json` — selection gates captured when the run launched.
- `source/` — launch-time source archive, tracked diff, launch script, and source commit identifier.
- `metadata.json`, `SHA256SUMS.txt` — provenance and integrity information.

## Selection record (online proxy, not held-out deployment evaluation)

- Iteration: `6862`; online score: `0.8865376111457486`.
- Mean terrain level: `9.4703`; platform-gap mean level: `9.6111`.
- Base-contact termination: `0.9549%`; platform-gap termination: `0%`; stairs termination: `1.7974%`.
- The contemporaneous Top-5 gate record reports all gates passing; reference iteration `6703`.
- GAST diagnostics recorded reconstruction loss `0.15763`, blackout fraction `0.19553`, valid-cell fraction `0.78035`, and noise strength `1.0`.

These are online selection-window statistics. The score is explicitly marked `score_is_online_proxy`; it must not be presented as a held-out evaluation result.

## Distillation

Use the GAST-specific student pipeline and this checkpoint as the frozen teacher, e.g. from the repository's `gast/` working directory:

```bash
./scripts/train_student_3gpu.sh --teacher_checkpoint /path/to/6862_top1.pt
```

Check the current `gast/scripts/train_student.py` CLI and the GAST student README before launching: environment count, iteration target, sensor/rendering requirements, GPU allocation, and output path should be set for the destination machine. Do **not** substitute the generic BIVT distillation script: this checkpoint belongs to the GAST teacher/student path. Student training uses rendered camera observations, so it still incurs simulator camera-rendering cost. Student training and deployment/export validation are not included in this package.

## Provenance caveat

The source snapshot is the exact launch-time snapshot recorded with the run: base commit `036e2846648f5a4cb16bda8a53dbbea91d325815` plus `source/tracked_changes.patch` and `source/source.tar.gz` (the latter also captures launch-time source files that were not tracked by Git). `params/` are copied from that same run. Consult `metadata.json` for the source run path and SHA-256 of the teacher checkpoint.
