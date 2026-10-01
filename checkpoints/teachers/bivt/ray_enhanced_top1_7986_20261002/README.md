# Enhanced BIVT-Ray Top-1: iteration 7986

Teacher for downstream camera-student distillation, not a distilled student.
Model: `teacher/7986_top1.pt`. Full training checkpoint; parameters in `params/`.
Source implementation: commit `8bc0ac57f84d7a0cbf92447a4c9f0a61cb2956de`, branch `vrl_models`.
See `docs/teachers/bivt.md` for enhanced visibility geometry and validation tradeoffs.

The model was trained with rendering-free terrain rays, leg capsules, trunk box,
foot spheres, conservative edge rejection and camera transport timing.
Online Top-1 score is not a held-out or deployment evaluation result.
The saved leaderboard is a snapshot; later rankings can change.

For distillation use the repository camera-enabled student task and
`scripts/distill_student.py --teacher_checkpoint <absolute-path>/teacher/7986_top1.pt`.
This is an argument fragment, not a complete launch command. Do not use the
ray-only teacher task as the image-producing student environment.
Verify teacher observation/encoder compatibility before a full distillation run.
Teacher capture interval is 70-100ms, delivery latency 0-50ms, packet drop 5%.
The existing 5090 student uses 0-150ms latency: this difference must be explicitly
resolved for the new experiment. Do not alter a currently running experiment.

`metadata.json` records source, model hash and exact online selection metrics.
Verify transferred files using `sha256sum -c SHA256SUMS.txt` from this directory.
