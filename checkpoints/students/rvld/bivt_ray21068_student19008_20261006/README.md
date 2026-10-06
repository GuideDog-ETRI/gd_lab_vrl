# BIVT-Ray 21068 → RVLD student, iteration 19008

This package preserves the user-selected RVLD `cnn_gru` checkpoint at capture iteration 19008 from run `rvld21068_512env_20k_20261006_080741`. The run completed 20000 capture iterations; the final checkpoint is not part of this package. This checkpoint was not selected by held-out validation or a student Top-1 leaderboard.

- Student: `student/perception_19008.pt` (SHA256 `e5ede9b6369e9395152fca0a713f6af0f24426ff3a6c2030e10c359f84fe3727`)
- Frozen teacher: `checkpoints/teachers/bivt/ray_gap_clean_vendor_new_top1_21068_20261005/teacher/21068_top1.pt` (SHA256 `fa397d22a949e24f74312b5270b5954a4ddb15c7451ea85ba8f64dbeeb7c9501`), already tracked in this branch
- Training: 512 environments, 20000 capture target, BPTT 16, seed 42, learning rate 0.0003, 1000-iteration save interval, hidden size 64
- Camera: `vendor_new`, four 80×45 depth plus `ir_proxy` views, 70–100 ms capture interval, 0–150 ms delay, frame-set drop probability 0.05

`metadata.json` records the teacher reference, hashes, training parameters, and supervision contracts. `camera_contract.json`, `params/`, and `source/` preserve calibration and run provenance. Verify the package with `sha256sum -c SHA256SUMS.txt` from this directory.

The checkpoint contains training and optimizer state. It is not a standalone deployment export, and no held-out terrain, gap, stair, MuJoCo, or hardware evaluation is claimed here. The checkpoint's embedded `teacher_checkpoint` is the original server-local path; use the repository teacher path above on another machine.
