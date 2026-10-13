# GAST v2.1 teacher Top-5 (2026-10-13)

This package preserves the five ranked teacher checkpoints from the completed
30,000-iteration GAST GapClean v2.1 run. The `.pt` files were pushed once (a71b669) to
transfer them from the multi-GPU server and then removed from Git (weights stay outside Git);
copy them between servers and check them with SHA256SUMS.txt. Configuration and provenance stay in Git.
The original run files remain untouched.

| Rank | Checkpoint | Online score | Clean gap | Mean terrain level |
| ---: | --- | ---: | ---: | ---: |
| 1 | `teacher/26123_top1.pt` | 0.87019 | 95.6% | 9.16 |
| 2 | `teacher/24906_top2.pt` | 0.86705 | 95.8% | 9.34 |
| 3 | `teacher/27869_top3.pt` | 0.86638 | 95.3% | 9.21 |
| 4 | `teacher/18277_top4.pt` | 0.86503 | 94.6% | 9.42 |
| 5 | `teacher/20825_top5.pt` | 0.86490 | 93.1% | 9.28 |

These scores are online selection proxies from different episode samples, not
fixed-seed walking evaluations. The complete criteria and sample counts are in
`leaderboard.json`; `metadata.json` gives the five checkpoint hashes and key
metrics. Verify all copied files with `sha256sum -c SHA256SUMS.txt` from this
directory before use.

The saved `params/agent.yaml`, `params/env.yaml`, and
`params/gast_warm_start.yaml` are the exact run configurations. The
`provenance/` directory contains the launch-time Git commit, tracked-code patch,
untracked diagnostic module, launcher, and online Top-5 criteria. The training
worktree had uncommitted source changes; a clean checkout of current `main` is
not asserted to reproduce these weights without that recorded source snapshot.
The agent configuration still contains training resume fields and must not be
treated as a deployment command.

These are PyTorch **training checkpoints**, not ONNX/TorchScript robot policy
exports. No deployment conversion, simulator replay, or robot-control validation
is claimed by this package. It does not include the 30,000-iteration final model
or student weights.
