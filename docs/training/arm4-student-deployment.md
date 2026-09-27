# Arm4 teacher 3700 -> camera student -> MuJoCo

Teacher checkpoint:
`logs/vision_rbq10_dreamwaq/arm_4/2026-09-26_17-03-13_arm4_3gpu_top5_resume_model3000/model_3700.pt`.

## Timing and gains

Set `TRAIN_ARM=4` for both `scripts/train_perception.py` and
`scripts/play_student.py`. These scripts now apply the experiment defaults
before explicit Hydra overrides. Play keeps its fixed command/curriculum settings.
Arm4 uses a 0.01 s policy step, hip/thigh Kp 123.39, knee Kp 127.77, Kd 2.4.
The camera interval remains 0.08 s: eight Arm4 policy steps.
Student checkpoints record the actual 0.01 s / 8-step contract. Old student
checkpoints with a 0.02 s / 4-step contract will fail the play contract check.

In `gd_rbq10_deploy_vrl`, the VRL backend uses a 100 Hz actor (five 2 ms reference
ticks), Kd 2.4 and a fresh-frame student. This backend is now configured for
Arm4; older 50 Hz VRL actors need their own matching backend configuration.
The student polls every 5 ms, processes a set only when all four depth and IR
channels have new receive versions, and holds the latent between updates.
Receive timestamps must be within 50 ms of each other; this does not establish
source capture synchronization. No frames older than 250 ms are admitted.
An old latent expires after 250 ms and the existing actor path uses zero latent;
there is no new automatic stop mechanism.

The nominal contract remains 12.5 Hz. Distillation now samples camera capture
intervals of 70--100 ms (7--10 Arm4 steps, 10--14.3 Hz), transport delays of
0--50 ms, and drops 5% of complete four-camera sets. These are configurable
initial robustness settings, not measurements of the target device. Capture and
transport timing are shared across the simulation batch; each robot's episode
identity and GRU state are handled independently.

Images, latent targets, hazard labels and episode identities are frozen together
at capture. The student advances only when a packet is delivered. Pre-reset
packets and packets older than the last delivered capture are discarded per
robot. BPTT spans received frames; missing frames do not advance the GRU. The
transport distribution and seed are saved in student checkpoints. The renderer
runs at policy boundaries to support arbitrary scheduled capture ticks, so this
costs more rendering than the original fixed 80 ms path.

This teaches perception with irregular/missing observations; matching a delayed
image to its capture-time latent does not train prediction of the present terrain
or prove that the actor tolerates old latents. Use the same transport options in
`play_student.py` to assess the complete control loop. Its default is an
80 ms, zero-delay, no-drop baseline. Actual hardware capture synchronization,
delay and loss distributions still need measurement.

## Distillation (choose an available physical GPU)

On this server the detached launcher is:

```bash
./scripts/start_arm4_student_tmux.sh 0 20000 64
tmux attach -t vrl_arm4_student3700
```

Arguments are physical GPU index, capture attempts, environment count. It refuses
duplicate student jobs, uses a dedicated USD temp directory, writes a persistent
console log and exit status, and saves every 200 captures. Detach with Ctrl-b then
d. The current run name is recorded in `logs/arm4_student3700_latest_run.txt`.
GPU 0 can share with the teacher at this measured size, but compute contention
can slow both jobs. Do not repeat the launcher while a student is running.

Run inside the established Apptainer environment; do not start a second job on
a GPU without checking its free memory. Replace `GPU_INDEX` below with the chosen GPU.

```bash
cd /data/users/bsseo/gd_lab_vrl
mkdir -p logs/usd_tmp/student_arm4
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=GPU_INDEX TRAIN_ARM=4 \
  OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  apptainer exec --nv --writable-tmpfs \
  --bind "$PWD/logs/usd_tmp/student_arm4:/tmp/IsaacLab" \
  /data/users/bsseo/gd_lab_isaaclab.sif /data/users/bsseo/venv/bin/python \
  scripts/train_perception.py --headless --device cuda:0 --num_envs 64 \
  --load_run 2026-09-26_17-03-13_arm4_3gpu_top5_resume_model3000 \
  --checkpoint model_3700.pt --iterations 20000 --save_interval 200 \
  --bptt_steps 8 --perception_run_name arm4_teacher3700_student \
  --camera_interval_ms 70 100 --camera_delay_ms 0 50 --camera_drop_prob 0.05
```

These are initial env-count/training-length choices, not validated capacity or
convergence claims. Iterations count camera ticks. Confirm the log reports
`step_dt=0.01`, nominal eight-step contract, sampled interval steps `(7, 10)`,
delay steps `(0, 5)`, and the exact teacher path.
Student output is under
`logs/vision_rbq10_dreamwaq/arm_4/arm4_teacher3700_student/`.

Use the same container wrapper and `TRAIN_ARM=4` for `play_student.py` with the
same `--load_run`, `--checkpoint`, and a `--student_checkpoint` path. Compare
teacher and student under matching fixed terrain and command conditions.
For a temporal stress evaluation append
`--camera_interval_ms 70 100 --camera_delay_ms 0 50 --camera_drop_prob 0.05`.
To recover fixed timing in distillation use
`--camera_interval_ms 80 80 --camera_delay_ms 0 0 --camera_drop_prob 0`.
`--iterations` counts capture attempts (including drops), not optimizer updates.
`transport/updates`, `transport/delay_ms`, and `transport/dropped` report delivery
behavior. A window with no deliveries skips the optimizer update.

## Export and transfer

Using the training Python environment, export the teacher with
`scripts/export_vrl.py CHECKPOINT --out exported/arm4_teacher3700` and the chosen
student with `scripts/export_student_vrl.py STUDENT_CHECKPOINT --actor-onnx
exported/arm4_teacher3700/policy_vrl.onnx`.
Copy both `policy_vrl.onnx` and `policy_vrl_student.onnx` to the target deployment
repo at `resources/policy/vrl/arm4_teacher3700/`.
The current exporters do not embed a complete timing/gain/calibration contract;
retain this document and the student checkpoint metadata alongside the files.

On the target PC, after configuring the simulator dependencies and building:

```bash
RBQ_WALK=ours \
RBQ_POLICY_FILE=vrl/arm4_teacher3700/policy_vrl.onnx \
RBQ_SIM_VISION=1 bash scripts/run_sim_vrl.sh
```

The deployment repository defaults to `vendor`, so specify `RBQ_WALK=ours`.

## Verification and resource measurement (2026-09-27)

- CPU regression tests: 33 passed (`test_camera_transport.py`, `test_vrl_camera_fixes.py`).
- Python entry points/modules: compileall passed.
- Deployment: both changed C++ translation units passed `g++ -std=c++17 -fsyntax-only`
  with actual vendored SDK/ONNX Runtime and extracted Eigen/OpenCV headers. Full
  CMake/Pilot linking could not run on this host (CMake and development dependencies
  are not installed); MuJoCo validation remains for the target PC.
- Isaac Lab student distillation from teacher 3700: 4 envs / 24 capture attempts
  completed, checkpoints 8/16/24 saved; two packet drops occurred.
- Isaac Lab student playback: 4 envs / 160 policy steps with 70--100 ms capture,
  0--50 ms delay, 5% packet drops completed. This only verifies execution, not
  convergence or deployable locomotion quality.
- Capacity probe: 64 envs / 32 capture attempts completed on RTX PRO 6000 Blackwell
  Server Edition 96 GB, concurrently with the teacher. Sampled peak process GPU
  memory: 7716 MiB (1 s sampling); peak process RSS: 16476844 KiB. Training loop:
  14.16 s; whole process including startup/shutdown: 74.53 s. A short sample is
  not a long-run capacity guarantee and does not establish RTX 5090 performance.
  Measurements: `logs/student_timing_capacity64_resources.json`.

For a different PC, copy this modified repository (including assets), the teacher
3700 checkpoint and its `params/agent.yaml` and `params/env.yaml`, and recreate
the compatible Isaac Lab/Python environment. The deployment repository alone
does not provide the simulator needed for distillation. Use the repository
revision containing `camera_transport.py` and the Arm4 student launcher.
