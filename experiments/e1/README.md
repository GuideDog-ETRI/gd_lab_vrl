# E1 closed-loop evaluation (UPDATE121–137)

Isaac Lab E1: 32 envs × 30 s, vx 1.0 m/s, camera transport 100 ms / 0–150 ms delay / 5% drop, ARM4 gains.
Teachers only (no student). Run from anywhere; paths resolve to the repo root.

- `run_trial128.sh <arm> <seed> <envs> <out.json> [seconds]` — one trial. Env: `U130_TEACHER` (teacher .pt; its
  package must contain `params/env.yaml`), `U129_FAMILY` (`gap | stairs_inv | pyramid_stairs`),
  `U130_SENSOR` (`lidar | cam`), `U136_IR` (`native | zero`), `U136_CAM_FILTER=1` (camfilter136), `U136_CAM_OFF`
  (diagnostic: zero sensor121 noise rates), `GD_LAB_REPLAY_LOCK`, `GAST_USD_TMP`.
- Arms: `G` ground-truth map, `GV` GT masked to LiDAR-visible cells, `v25or` = `L0S` + `U130_DUAL=2 U130_DDEEP=0.7`
  (deployed LiDAR structure processor, tools/rl_replay/structure130.py).
- `e1_136.sh jobs.txt <slots_per_gpu>` — queue driver; `E1_GPUS="0 1"`, `GAST_IMAGE`, `GAST_CONTAINER_PYTHON`.
  Job line: `name family arm teacher_pt seed ir_mode [sensor] [camoff]`. Outputs go to `<repo>/outputs/<name>/`.
- Labels: `*_stdown` = `pyramid_stairs_inv` (climbing out of a pit), `*_stup` = `pyramid_stairs`.
- Results of the Ray+IR teachers: `checkpoints/teachers/gast/rayir_top5_20261016/e1/`.
