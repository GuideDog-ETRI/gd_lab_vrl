# BIVT-Ray teacher 21068 Top-1 — clean gap fine-tune

This package contains the BIVT-Ray RBQ10 DreamWaQ training checkpoint selected
as Top-1 at iteration 21068, along with the source run's agent/environment
configuration, fine-tuning manifest, and a leaderboard snapshot captured when
this package was assembled. The snapshot includes the iteration-21068 entry;
it is not a claim that the live leaderboard had this exact state at selection
time.

- Checkpoint: `teacher/21068_top1.pt`
- SHA256: `fa397d22a949e24f74312b5270b5954a4ddb15c7451ea85ba8f64dbeeb7c9501`
- Source run: `logs/vision_rbq10_dreamwaq/arm_4/2026-10-05_22-54-55_2026-10-05_22-54-39_gap_clean_ddp3_s42_train/`
- Fine-tune task: `Gd-VrlGapFinetuneCleanRaycast-Rbq10-Dreamwaq-v0`
- Source camera profile: `vendor_new`; the resumed/base teacher was recorded as
  `vendor_legacy` in the manifest. This camera-profile change is part of the
  run provenance.
- Base teacher: iteration 17206, SHA256
  `a70adbda5b73925f5936eaef27ee3b33686d9f7489486afdda9bf40ddb0c326d`
- Distributed setup: 4096 total environments, 3 ranks. The launch used seed 42;
  the run manifest records rank-local seed 44.
- Top-1 online-proxy score in the snapshot: `0.8652681642427711`; clean gap
  crossing rate `0.9562962962962962`; base contact rate `0.03339371980676328`.

The checkpoint is a PyTorch training artifact, not an exported deployment
