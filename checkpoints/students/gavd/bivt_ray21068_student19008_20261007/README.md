# BIVT-Ray 21068 to GAVD student, iteration 19008

This package preserves the GAVD `grid_attention_v1` checkpoint selected from the completed gavd21068_512env_20k_20261006_170023 run (20,000 capture iterations, 512 environments).

- Student: `student/perception_19008.pt`, SHA256 `975bea9794a2aadc504fe29153da7705a19ae63cf2f166ff3f9b64dc7663dafc`
- Frozen teacher: `checkpoints/teachers/bivt/ray_gap_clean_vendor_new_top1_21068_20261005/teacher/21068_top1.pt`, SHA256 `fa397d22a949e24f74312b5270b5954a4ddb15c7451ea85ba8f64dbeeb7c9501` (already tracked in the repository; not duplicated here)
- Camera profile: `vendor_new`; four 80 x 45 depth plus IR proxy views. See `camera_contract.json` for intrinsics and mount poses.
- Supervision: capture-time teacher action, visible valid finite teacher geometry, teacher ray-mask visibility, alignment contract v2.
- Training: 512 environments, 20,000 capture iterations, BPTT 16, LR 0.0003, seed 42, warmup 1000, ramp 4000, save interval 1000. Camera interval 70-100 ms, delay 0-150 ms, drop probability 0.05.
- Selection: minimum training proxy score 0.19830153 among 15 saved candidates at or after 6,000 iterations. For each candidate, average TensorBoard values over the preceding 1,000 capture iterations (start exclusive, end inclusive), then calculate latent MSE + 0.5 x hazard MSE + action MSE + 0.5 x spatial loss. See `selection.json` for the full ranked list, sample counts, and means.

`params/student/` and `params/teacher/` retain agent and environment configuration. `training_args.json`, `settings.txt`, `git_sha`, and `source/` retain training and source provenance. The checkpoint includes optimizer state. Its embedded teacher path is local to the training server; use the repository teacher path above on a deployment server.

This training proxy ranking is **not** a deployment walking-performance validation. No held-out terrain, gap, stair, MuJoCo, or hardware evaluation is claimed. Verify integrity with `sha256sum -c SHA256SUMS.txt` from this package directory.
