# GAST - Arm4 isolated experiment

Gap-Aware Spatiotemporal Teacher-Student Learning. External gap signals are not
used. This directory snapshots the existing gd_lab source, configs and robot
assets; scripts select only this directory's src through PYTHONPATH. Existing
CVTT/BIVT/RVLD/GAVD code outside gast is unchanged. No Git push is performed.

## Architecture

- Teacher: camera-free 11x17 height+validity; pose-warped history at ages
  approximately 0, .1, .2, .4, .8, 1.6, 3.2, 4.8 s. CNN 8/16 -> 64, temporal
  4-head attention -> GRU64 -> latent32. History is stored in PPO observations.
- Denoising decoder: latent32 -> 128 -> 187x6 (height, valid, gap, up, down,
  support). Joint PPO and 0.1-weight reconstruction auxiliary update.
- Existing CENet implementation and architecture are unchanged. Teacher trains
  from scratch; student freezes the whole teacher including CENet and Actor.
- Student: four 80x45 Depth+IR-proxy images -> shared CNN -> calibrated grid
  cross-attention -> short temporal attention + pose-warped cell GRU32 -> latent32.
  BPTT64 captures covers roughly 5 s. Student pose uses simulator ground truth in
  this first implementation; deploy must supply estimated pose and evaluate drift.
- Auxiliary grid reconstruction and learned quality gate. Missing images and
  age >= .3 s give exactly zero terrain contribution. Blackout trains the new
  policy's blind route; this is not numerically identical to an old DWB actor.
- Gap labels use known gap terrain families and pit-floor geometry. No-return
  depth is not treated as a semantic gap. Support is a local geometric proxy,
  not proof that a foothold is dynamically safe.
- Arm4: physics .005 s, policy .01 s; hip Kp123.39 / knee Kp127.77, Kd2.4.
  Existing platform-gap all-feet crossing reward and terrain curriculum retained.

## Run

From this directory, `scripts/run.sh teacher --num_envs 256 --max_iterations 20000`
trains a new teacher. For the student, use `scripts/run.sh student --num_envs 16
--iterations 20000 --bptt_steps 64 --lr .0003 --teacher_checkpoint /absolute/model.pt`.
`python3 scripts/pipeline.py` defaults to SMOKE ONLY: teacher3 -> student16 ->
student resume24. `python3 scripts/pipeline.py --train` explicitly selects the
20,000 + 20,000 sequence. It never retries a failed stage. State:
logs/current_gast_run.json; logs and models remain under gast/logs. No production
training has been started on this machine; user moved training to another server.

On the training server set GAST_IMAGE and GAST_CONTAINER_PYTHON for Apptainer,
or GAST_PYTHON to a working native IsaacLab Python executable. GAST_GPU selects
one GPU (default 0). This first pipeline is single-GPU, even on a multi-GPU host.
The server must provide IsaacLab / Isaac Sim and the matching rsl_rl dependency;
these large installed runtimes are not bundled. Run unit tests and default smoke
on the destination before considering full training.

Teacher iterations are PPO updates (100 control steps/env/update by default).
Student iterations are camera capture attempts, not PPO iterations. Report
seconds/update, seconds/capture and wall time separately. A short smoke test
establishes execution only, not gap or stair competence. Existing online Top-5
is a rollout proxy, not held-out validation.

## Design document

The bundled PDF and Markdown at docs/rbq10_learning_methods_20260930.*
include a GAST architecture page and design
appendix. References: Miki et al. 2022 (arXiv:2201.08117), Extreme Parkour 2023
(2309.14341), MSTA (2409.03332), foot-position maps (2604.02744).

## Implementation limits

Teacher noise currently includes height jitter, episode bias, cell dropout and
whole-scan blackout. Arbitrary spatial calibration error and realistic pose drift
remain follow-up tests. Camera IR is rendered luminance, not physical IR. The
student has a new recurrent-state contract and is not supported by old GAVD ONNX
export/deployment scripts. No robot deployment is performed by this pipeline.

## Local validation - 2026-10-01

- 4 CPU tests passed: memory translation, temporal gradient/order, unknown-depth
  label masking, missing/stale exact-zero gate and student backward.
- Teacher smoke: 32 env, horizon16, 3 PPO updates, model_2.pt saved.
- Teacher benchmark: 256 env, horizon100, 6 updates, model_5.pt saved;
  60.70 s total learning time; steady iterations approximately 8.9-9.1 s while
  another student trained on the GPU. GPU total memory approached 31.2/32.6 GB.
  Re-benchmark batch/memory on the destination; do not assume this is its ETA.
- Student smoke: 4 env, BPTT8, 16 captures, 15 delivered packets, saved at8/16;
  9.87 s learning time. This does not measure the production 16-env/BPTT64 case.
- CENet source SHA256 (same as original):
  81757443cd6edf78171d16edc9ec7cdbee45682af31a3055f98cba89aca2f441.
- No 20,000-update teacher or student has been launched. Destination smoke also
  checks student resume; local optimizer-resume and production BPTT64 simulation
  remain unverified. Smoke checkpoints are intentionally not versioned.

## OnVLM2 three-GPU teacher (2026-10-01)

The single-GPU validation above is historical. The OnVLM2 launcher is
`bash scripts/train_teacher_3gpu.sh smoke|benchmark|train` and uses GPUs 0,1,2
only; GPU3's existing vLLM service is not touched. Production runs inside tmux
session `gast_teacher_3gpu`, independently of the SSH client.

- Teacher only, from scratch; Arm4 seed42; 30,000 PPO updates.
- Total 4096 environments split 1366/1365/1365; horizon100, epochs5,
  minibatches4; 200Hz physics/100Hz control; original eleven terrains including gap.
- Ordinary checkpoints every1000 iterations (and initial/final saves).
- Top5 eligibility checked EVERY iteration starting at iteration5000, with
  spacing1 (adjacent iterations allowed), using `configs/online_top5.json`.
  Mean terrain >=9, gap >=8, base/gap/stairs base-contact termination <=9%,
  gap-success regression <=2 percentage points and base-contact deterioration
  <=1 percentage point versus current Top1. Existing weighted online score is
  unchanged. This is rollout ranking, not separate held-out evaluation.
- GAST reconstruction, blackout fraction, valid-cell fraction and noise strength
  are diagnostics only; they do not introduce extra admission gates.
- CENet optimizer gradients/skip decisions and terrain reconstruction gradients
  are synchronized, as are observation normalization and AdaBoot state.
  Smoke runs compare model and optimizer hashes after EVERY update; production
  checks every100 updates. This changes the historical CENet source hash above.
- Large symmetry-augmented batches exceed a CUDA attention launch limit.
  Independent encoder samples are executed in chunks with activation
  recomputation, preserving the four PPO minibatches and optimizer-step count.
- Rank0 owns configuration/metadata/checkpoint writes. Checkpoint publication is
  atomic; loss diagnostics and curriculum means are retained. Completed-run
  metadata is not written when an early-stop checkpoint was requested.
- To stop safely, send SIGUSR1 to a VERIFIED teacher worker PID (not torchrun).
  All ranks finish the current update, rank0 saves, then all workers exit.
  Do not use SIGKILL or assume tmux alone can save on a power failure.
- Each launch stores source/config snapshots in `logs/launches/<run-id>/`.
  Raw upstream console metrics are rank0 unless explicitly marked global;
  Top5 gathers completed episodes across all three ranks. `GAST_PROGRESS`
  reports exact global timesteps and an ETA in seconds without 24-hour wrapping.

Validation commands: `python tests/test_gast.py`, `python tests/test_top5_schedule.py`,
and `python -m torch.distributed.run --standalone --nproc_per_node=3 tests/test_distributed_sync.py`.
