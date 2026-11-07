# Ray+IR teacher run 3906_gap50_from_dwb38000

- Host: 90.111 5090 (1 GPU); 2026-10-15 07:54 → 10-16 00:17; 1024 envs × 100 steps × 5000 PPO updates (5.12e8 env steps), seed 42.
- Start: DWB-38000 blind ARM4 (c6cfe178…, checkpoints/teachers/blind/arm4_38000)
- Task: Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0. Input: raycast camera-visible cells + NoisyTerrain noise + per-cell IR proxy (gd_lab/gast/rayir.py, IR_GAIN .57, matched to the noen rosbag).
- Init: --warm_start_dwb: 45 locomotion tensors copied, 24 terrain tensors fresh, latent head and actor latent columns zero; new PPO/CENet optimizers; default terrain levels
- Launcher (repo layout): `scripts/gast/train_rayir_gap50_1gpu.sh dwb train`
- Files: `best_top5/*.pt` + `leaderboard.json` (online Top-5, criteria configs/gast/online_top5_rayir.json), `params/` (env.yaml, agent.yaml, gast_warm_start.yaml), `completion.json`.

| checkpoint | online score | G (gap clean) | B (1-contact) | mean terrain level | sha256 |
|---|---|---|---|---|---|
| 3906_top1.pt | 0.890 | 1.00 | 1.00 | 9.33 | `f3e7afe53ea8` |
| 4538_top2.pt | 0.854 | 0.89 | 1.00 | 9.56 | `9162b30908ae` |
| 2142_top3.pt | 0.847 | 0.92 | 0.96 | 9.91 | `5823d0b13bff` |
| 2944_top4.pt | 0.829 | 0.85 | 0.98 | 9.14 | `cd9d98a63919` |
| 1630_top5.pt | 0.816 | 0.82 | 0.99 | 9.34 | `4a450b73de1b` |

Only the Top-1 was evaluated in E1; see ../e1/SUMMARY.md. The online score is a training proxy, not an evaluation.
