# Ray+IR teacher run 2836_gap50_from26123

- Host: local 5090 (1 GPU); 2026-10-15 07:54 → 10-16 00:25; 1024 envs × 100 steps × 5000 PPO updates (5.12e8 env steps), seed 42.
- Start: GAST v2.1 26123 Top-1 (113c2b88…)
- Task: Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0 (20 columns, platform_gap ×10 = 50%). Input: raycast camera-visible cells + NoisyTerrain noise + per-cell IR proxy (gd_lab/gast/rayir.py, IR_GAIN .57, matched to the noen rosbag).
- Init: --warm_start_gast as above
- Launcher (repo layout): `scripts/gast/train_rayir_gap50_1gpu.sh gast train`
- Files: `best_top5/*.pt` + `leaderboard.json` (online Top-5, criteria configs/gast/online_top5_rayir.json), `params/` (env.yaml, agent.yaml, gast_warm_start.yaml), `completion.json`.

| checkpoint | online score | G (gap clean) | B (1-contact) | mean terrain level | sha256 |
|---|---|---|---|---|---|
| 2836_top1.pt | 0.891 | 1.00 | 1.00 | 9.85 | `d40c79e8ac1e` |
| 3381_top2.pt | 0.871 | 0.94 | 1.00 | 9.51 | `c2cecc0ad1d5` |
| 4281_top3.pt | 0.867 | 0.94 | 0.97 | 10.43 | `ba4a1c4c0f53` |
| 626_top4.pt | 0.867 | 0.90 | 1.00 | 9.30 | `e5c951cc0ead` |
| 1249_top5.pt | 0.866 | 0.93 | 0.98 | 10.56 | `e3f226cffe17` |

Only the Top-1 was evaluated in E1; see ../e1/SUMMARY.md. The online score is a training proxy, not an evaluation.
