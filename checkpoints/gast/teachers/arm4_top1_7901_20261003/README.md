# GAST teacher checkpoint: iteration 7901

This is the **GAST teacher** checkpoint selected as online Top-1 when packaged. It is intended as the frozen teacher for a separate GAST student-distillation run. It is not a student checkpoint or deployment-ready model.

## Package contents

- `teacher/7901_top1.pt` — teacher checkpoint.
- `params/agent.yaml`, `params/env.yaml` — run-specific configurations.
- `selection/leaderboard.json` — Top-5/leaderboard snapshot at packaging time.
- `selection/online_top5.json` — selection thresholds captured for the run.
- `source/` — source commit, launch-time source archive/diff, and launch script.
- `metadata.json`, `SHA256SUMS.txt` — provenance and integrity manifest.

## Selection record

- Iteration `7901`; online score `0.8883521135164075`.
- Mean terrain level `9.1147`; base-contact termination `1.6667%`; platform-gap termination `0%`; stairs termination `3.2051%`.
- Recorded gates passed; reference iteration `6862`.
- GAST diagnostics: reconstruction `0.17582`, blackout fraction `0.18092`, valid-cell fraction `0.79454`, noise strength `1.0`.

The score and rates are online selection-window statistics, not a held-out evaluation or deployment benchmark. Ranking can change as training continues; the bundled leaderboard records the snapshot at packaging time.

## Student distillation

Use the GAST-specific pipeline from the repository's `gast/` directory, for example:

```bash
./scripts/run.sh student --teacher_checkpoint /path/to/7901_top1.pt
```

Before running on the destination host, review `gast/scripts/train_student.py` and the GAST student documentation, and set environment count, training budget, GPU allocation, rendering/sensor settings, and output path appropriately. Do not use the generic BIVT distillation pipeline for this checkpoint. Student training uses rendered camera observations and therefore still has camera-rendering cost. This package does not include a trained student or deployment/export validation.

## Source provenance

The teacher run was launched from base commit `036e2846648f5a4cb16bda8a53dbbea91d325815`; the accompanying launch-time tracked diff and `source.tar.gz` preserve its changes and source snapshot. The exact run and checkpoint SHA-256 are recorded in `metadata.json`.
