# Ray+IR teacher run 2769_from2000

- Host: onvlm2 GPU0 (1 GPU); 2026-10-14 21:43 → 10-15 17:11; 1024 envs × 100 steps × 5000 PPO updates (5.12e8 env steps), seed 42.
- Start: GAST v2.1 model_2000 (db500e96…)
- Task: Gd-GastRayIrV21-Rbq10-Dreamwaq-v0 (10 terrain families, normal mix). Input: raycast camera-visible cells + NoisyTerrain noise + per-cell IR proxy (gd_lab/gast/rayir.py, IR_GAIN .57, matched to the noen rosbag).
- Init: --warm_start_gast: all 69 tensors, terrain_encoder.cnn.0 widened 2→3 (IR weights 0); CENet optimizer + terrain levels restored; new PPO optimizer
- Launcher (repo layout): `scripts/gast/train_rayir_from2000_1gpu.sh train`
- Files: `best_top5/*.pt` + `leaderboard.json` (online Top-5, criteria configs/gast/online_top5_rayir.json), `params/` (env.yaml, agent.yaml, gast_warm_start.yaml), `completion.json`.

| checkpoint | online score | G (gap clean) | B (1-contact) | mean terrain level | sha256 |
|---|---|---|---|---|---|
| 2769_top1.pt | 0.890 | 1.00 | 1.00 | 9.20 | `65da772b8b79` |
| 969_top2.pt | 0.886 | 1.00 | 0.94 | 10.00 | `1a009d5c37df` |
| 4271_top3.pt | 0.869 | 1.00 | 0.98 | 9.00 | `cefa0529f49a` |
| 4884_top4.pt | 0.861 | 0.91 | 1.00 | 9.40 | `2d04e6b9cec9` |
| 2149_top5.pt | 0.858 | 0.94 | 1.00 | 9.46 | `4276c9039566` |

Only the Top-1 was evaluated in E1; see ../e1/SUMMARY.md. The online score is a training proxy, not an evaluation.
