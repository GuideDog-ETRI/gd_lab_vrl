# Arm4 BIVT-Ray-17206 → GAST student, capture 20000

This package contains the final GAST student checkpoint from
`gast/logs/gast/arm4/gast17206_student_512env_20k_fromscratch_20261004/`
and the exact BIVT-Ray teacher used for distillation. The teacher is duplicated
here intentionally so this student package remains self-contained.

- Student: `student/perception_20000.pt` (iteration 20000)
- Student architecture: `gast_spatiotemporal_v1`, hidden dimension 6116, terrain latent dimension 32
- Teacher: `teacher/model_17206_top1.pt`
- Teacher SHA256: `a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d`
- Student SHA256: `d64f3788d1f319e1b125ed3922d5bb360d42497b87a2bb98635afeb5692f36f9`
- Training: 512 envs, 20000 capture attempts, BPTT 16, seed 42, LR 0.0003, save interval 1000
- Camera: `vendor_legacy`, four 80×45 depth + `ir_proxy` views; 70–100 ms interval,
  0–150 ms delay, 0.05 frame-set drop probability

`params/` and `teacher/params/` preserve the student and teacher environment
contracts. `selection/` preserves the run's Top-5 criteria and leaderboard.
The checkpoint contains student/optimizer state; it is not a standalone policy
export and does not certify gap/stair walking performance. `ir_proxy` is not a
physical infrared sensor simulation. Verify `SHA256SUMS.txt` before use.
