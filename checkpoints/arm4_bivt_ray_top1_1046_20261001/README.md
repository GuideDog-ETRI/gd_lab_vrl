# BIVT-Ray iteration 1046 Top-1 candidate

This package contains the current online Top-1 candidate from the ongoing BIVT-Ray run, not its final 20,000-iteration checkpoint. Its score is an online proxy and is not a deployment evaluation result.

- Candidate: `teacher/1046_top1.pt`
- Online proxy score: `0.9449391933161341`
- Source run: `logs/vision_rbq10_dreamwaq/arm_4/2026-09-30_23-23-51_arm4_bs_2r_to20k`
- Parameters: `params/agent.yaml`, `params/env.yaml`
- Ranking and source details: `leaderboard.json`, `source_1046.json`
- Deployment validation: not performed

The final model should be packaged separately after the run reaches its configured target and the final checkpoint is verified.
