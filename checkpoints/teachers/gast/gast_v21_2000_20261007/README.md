# GAST v2.1 teacher, update 2000

- Source: 10.77.32.231 `gast-v21-20261007` (commit 4585f46), run `2026-10-07_11-02-05_gast_gapclean_from_bivt_train`, `model_2000.pt`
- Task `Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0`, warm start from BIVT-Ray Clean 21068, 3 GPU, 4096 env
- Not an online Top-1 (the board held 883 from before the disturbance/noise ramps finished); chosen as the 2,000-update milestone
- MuJoCo oracle evaluation 2026-10-07 (0.65 m gap course, 20 cm stairs): gap 17/18 completed, 21068 stair falls (pull 200 N, descend push 250 N) both survived
- Weights are not in git: `sha256sum -c SHA256SUMS.txt` after copying `teacher/model_2000.pt`
