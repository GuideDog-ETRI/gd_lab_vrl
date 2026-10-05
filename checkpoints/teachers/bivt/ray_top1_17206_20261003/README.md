# BIVT-Ray teacher 17206 Top-1

Frozen Arm4 RBQ10 DreamWaQ BIVT-Ray teacher checkpoint selected at iteration
17206 from the source run below. This package preserves the teacher checkpoint,
its environment/agent configuration, and the source leaderboard for reuse by
camera-student distillation.

- Source run: `logs/vision_rbq10_dreamwaq/arm_4/2026-10-03_18-52-19_dwb_v3_6_21_b1_18_ray_20k_1024_resume2000r1_20261003/`
- Checkpoint: `teacher/17206_top1.pt`
- SHA256: `a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d`
- Variant: `bivt_ray_occlusion_v2`; 32-dimensional terrain-latent actor contract
- Camera profile recorded by the source environment: `vendor_legacy`

The checkpoint is a training artifact, not evidence of student or deployment
performance. See `SHA256SUMS.txt` before use.
