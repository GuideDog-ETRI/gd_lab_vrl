# VRL arm2 on one RTX 5090: machine handoff

## Source

- Repository: `https://github.com/GuideDog-ETRI/gd_lab_vrl`
- Pull the current `main` branch before VRL arm2 training. It includes the
  verified camera cadence change.
- The change adjusts camera capture from a fixed 4 control steps to the number
  of steps that keeps the physical capture interval at 0.08 s. Arm2 and arm4
  control at 0.01 s (100 Hz), so they capture every 8 control steps. The scene
  renderer runs every 16 physics steps at 0.005 s per physics step. Arm1 and
  arm3 keep their existing 4-step cadence.

## What was verified on the source machine

- Environment: Apptainer/Isaac Lab 0.48.0, Torch 2.7.0+cu128, RSL-RL 3.0.1.
- `tests/test_vrl_camera_fixes.py`: 19 passed.
- VRL arm4 with 3 RTX PRO 6000 Blackwell GPUs: one distributed PPO iteration
  completed with 1,365 environments per GPU (4,095 total, 409,500 transitions).
- Rank 0 produced one `model_0.pt`; saved config had `sim.dt=0.005`,
  `decimation=2`, `render_interval=16`.
- That one iteration took 68.41 s after initialization on the three-GPU
  source machine. This timing does not predict RTX 5090 single-GPU speed.
- Arm2 and the RTX 5090 machine have **not** been run or benchmarked here.

## One-GPU arm2 smoke command on the other machine

Use that machine's paths for `SIF` and `PYTHON`. The Python environment must
have the repository installed editable and the matching Isaac Lab/RSL-RL
dependencies. Run from the repository root. `--num_envs` is the total count
on this one GPU. Start with 1,024 as an unverified memory-conscious setting;
adjust after observing actual GPU memory and iteration time.

```bash
cd /path/to/gd_lab_vrl
mkdir -p logs/usd_tmp/arm2_5090
SIF=/path/to/gd_lab_isaaclab.sif
PYTHON=/path/to/venv/bin/python
env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 TRAIN_ARM=2 OMNI_KIT_ACCEPT_EULA=YES \
  apptainer exec --nv --writable-tmpfs \
  --bind "$PWD/logs/usd_tmp/arm2_5090:/tmp/IsaacLab" \
  "$SIF" "$PYTHON" scripts/train_vrl.py \
  --task Gd-Vrl-Rbq10-Dreamwaq-v0 --headless --device cuda:0 \
  --num_envs 1024 --seed 42 --max_iterations 1 --logger tensorboard \
  --run_name arm2_5090_smoke agent.save_interval=1
```

Check the resulting `logs/vision_rbq10_dreamwaq/arm_2/.../params/env.yaml`
for `decimation: 2`, `dt: 0.005`, and `render_interval: 16`, and check that
`model_0.pt` exists. For the long run, choose an iteration count from the
measured time per iteration on the 5090, then change `--max_iterations` and
`--run_name`. Do not add `--distributed` or `torch.distributed.run` for one GPU.

The source-machine 3-GPU smoke log is
`logs/arm4_3gpu_1365_final_smoke_console.log`; its checkpoint is under
`logs/vision_rbq10_dreamwaq/arm_4/2026-09-23_14-15-31_arm4_3gpu_1365_final_smoke/`.
