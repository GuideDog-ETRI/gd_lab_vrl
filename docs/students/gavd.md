# Top-1 6987 attention student (2026-09-30)

Teacher: `checkpoints/teachers/cvtt/arm4_top5_20260930/6987_top1.pt`.
Online leaderboard score 0.8737814273; this is not a held-out ranking.
SHA256 `c153e0d3d18ce023b6b1dba4651cf9e01d3eb38b62a651c8ebeb2bb5f3236edd`.
The single-GPU launcher checks every supplied checksum, copies the teacher and
leaderboard to a unique run directory, and never alters teacher weights or deployment.
Existing user file `scripts/compare_arm4_stand.py` is unrelated and untouched.

## Architecture and objectives

`grid_attention_v1` is opt-in; the legacy CNN-GRU remains the default.
Four shared CNN maps retain 9x16 tokens per view. Camera identity, calibrated
camera-to-trunk rays, optical depth, and near/far validity embed into 32-D tokens.
187 learned queries with 11x17 robot-grid coordinates cross-attend to all views.
A 63-D recurrent memory generates the existing 32-D terrain latent.
This is learned grid alignment, not exact gravity-aligned reprojection: no current
robot pose is provided to this encoder. Far/no-return rays are retained because
discarding them could discard gap evidence. There is no hard sky crop or body mask.

Training loss is latent MSE + action MSE + 0.5 spatial loss + configured scalar
hazard loss (default 0.5). The actor is frozen, but action gradients flow through
it into the student. Images, actor base input, teacher action, terrain target and
episode IDs are all captured together before transport delay. Reset/reordered
packets are excluded with the existing transport mask.

Spatial labels supervise visible height, visibility, signed forward-axis edges
(4cm threshold), and a local planar-support proxy. Positive edges receive 5x
weight. Invisible geometry is not filled with fictitious labels. The support
proxy is not a guarantee of load-bearing or kinematically reachable footholds;
edge labels do not semantically distinguish every gap from a down-step.

The teacher drives initially. After 1,000 capture attempts the probability of
student actions ramps to 1 over 4,000 attempts. This generates student-visited
states with frozen-teacher supervision, rather than training only on teacher paths.
Environments lacking a delivered frame use teacher actions until one arrives.
Memory/latent readiness reset on episode termination. Teacher parameters and
normalizers remain in eval mode with gradients disabled.

Transport: 70–100ms intervals, 0–150ms delay, 5% whole-packet drops. Existing
camera noise/ghost augmentation remains enabled; independently 8% of views per
delivered sample are replaced by far depth and black IR. Loss metrics include
action error, spatial loss, student rollout fraction and delivered frame age.

## Important deployment contract

Shapes remain frames `[1,4,2,45,80]`, hidden `[1,64]`, latent `[1,32]`.
**Hidden slot 63 is input frame age in seconds, clipped to [0,1], not recurrent
memory.** Set this slot before inference; output slot 63 is zero. Slots 0–62
carry recurrent state. New checkpoints and export sidecars record this contract.
The current production runtime is NOT modified. Do not replace its student
without implementing the age-slot contract and verifying the paired 6987 actor.
An old actor with a different latent basis is not a valid substitute.

## Run and inspect

```bash
bash scripts/train_gavd_5090.sh 64 64 attention_smoke --student_warmup 0 --student_ramp 8
bash scripts/train_gavd_5090.sh 20000 64 arm4_teacher6987_attention_20k
```

Names must be unique. The local launcher uses the existing 5090 Apptainer image
and Python environment; it does not edit or execute the three-GPU launcher.
Outputs are under `logs/vision_rbq10_dreamwaq/arm_4/<run>/` with atomic
`perception_<capture>.pt` checkpoints every 200 attempts. Checkpoints contain
architecture, camera contract, frozen teacher SHA, optimizer and CLI settings.
Optimizer state is preserved for future recovery; automatic resume is not implemented.
Capture count is not PPO iteration count or optimizer update count (BPTT=8).

Export selection is architecture-aware in `scripts/export_student.py`.
Always use a fresh output directory and the corresponding frozen teacher actor.
No exported model is installed into the deployment repository by training.

## Acceptance criteria

Tests cover unknown-depth finite outputs, signed visible-edge supervision,
masked geometry gradients, frozen-actor/student gradient isolation, recurrent
BPTT/reset, checkpoint round-trip, legacy shape compatibility, TorchScript and
ONNX numerical parity. Run the camera transport and camera geometry tests too.
Simulation smoke must load 6987 strictly, optimize, save and terminate normally.
Low training loss alone does not establish walking quality: compare this student
against the legacy student with identical teacher, seeds, camera conditions and
terrain, including zero command, stairs, gaps and delayed/dropped frames, before
any deployment replacement. This task does not alter the live deployed model.

## Verified execution

- Targeted pytest: 39 passed, including ONNX parity, plus 11 existing VRL pipeline
  tests passed (50 total); ruff passed.
- `arm4_attention6987_smoke64_20260930`: 64 environments, 64 captures,
  24.19s training time excluding startup, normal exit 0. Final-window latent MSE
  0.01784, hazard MSE 0.00559; 3 dropped packets. This short run forced the student
  rollout ramp immediately to exercise that path, not to measure locomotion quality.
- Its `perception_64.pt` passed strict reload, finite-weight checks and frozen
  teacher SHA verification. The earlier 4-env run saved correctly but its shell
  exited 127 after a launcher edit; use the subsequent normal 64-env run as the
  acceptance smoke. The launcher now execs the simulator process.
- Isaac reported existing foot visual USD unresolved-reference warnings and NGX
  initialization warnings; neither prevented the smoke. These are not a claim
  that rendering is physically identical to the deployed MuJoCo images.
- Main run: `arm4_teacher6987_attention_20k_20260930`, 64 envs, 20,000 captures,
  tmux `vrl-attention6987`; console at `logs/<run>.console.log`.
  Training launch is not evidence that the full run has completed or improved
  deployment locomotion. The legacy deployed model remains unchanged.
