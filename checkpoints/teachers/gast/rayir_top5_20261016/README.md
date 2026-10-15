# Ray+IR GAST teachers, Top-5 of three runs (2026-10-14 … 10-16)

Pick a model by need; weights are in Git (force-added, ~10.8 MB each).

| run | start (lineage root) | terrain mix | Top-1 | E1 LiDAR v25or gap | E1 camera+filter gap | E1 stairs (v25or) |
|---|---|---|---|---|---|---|
| `2769_from2000` | GAST v2.1 2000 ← 21068 ← 17206 ← RBQ deploy blind `d_v3.6.21_b1_18` (ONNX) | normal | 2769 | 96/97 = 99.0% | 81/86 = 94.2% | 98.8% / 100% |
| `2836_gap50_from26123` | GAST v2.1 26123 (same lineage) | gap 50% | 2836 | 75/86 = 87.2% | 80/92 = 87.0% | 98.2% / 99.4% |
| `3906_gap50_from_dwb38000` | DWB-38000: our ARM4 blind DreamWaQ trained from scratch (onvlm2, 4096 env, 0→38000) | gap 50% | 3906 | 90/93 = 96.8% | 76/79 = 96.2% | 100% / 100% |

E1 = Isaac closed loop, 32 env × 30 s, seeds 42–45 (paired initial states), vx 1.0; stairs = pit climb-out / pyramid up.
Full tables (G, GV, v25or, camera, IR = 0) and per-trial result.json: `e1/` (`e1/SUMMARY.md`, `e1/summary.csv`).

- Code: `src/gd_lab/gast/rayir.py`, `scripts/gast/train_rayir_*.sh`, E1 `experiments/e1/`, `tools/rl_replay/{e1_lidar126,lidar126,structure130,cam130,camfilter136,rayir_e1}.py`.
- Evaluation needs the `params/env.yaml` next to the weights (camera-profile check).
- Simulation uses ground-truth pose; on the robot the pose comes from camel_pilot `/odom` (odom→base_link, 50 Hz).
- Camera path: spikes (0.6%) + min-z reducer made flat cells "deep"; `U136_CAM_FILTER=1` (camfilter136) fixes it.
