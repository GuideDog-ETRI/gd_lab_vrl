B′ + backstride stage 2 (KISTI Hangang)
job id      : 1228779 (lin16_b, natStr, 16 nodes x 4 GH200, 2048 envs/GPU = 131072)
commit      : abc0fab (gh200; working tree clean at start, see git_status_at_start.txt)
recipe      : LINEAGE_RECIPE=bprime_stride STRIDE_STAGE=2 NARROW_UPDATES=1000
source      : bprime_stride_v22_299.pt (sha 1062d254...ddf08), anchor backstride_a18l_299.pt (sha a1493d04...60b8e)
run dir     : /scratch/t515a05/gd_lab/gd_lab_vrl/logs/gast/lineage16_bprime_stride_s2/lineage16_b_16n_2048x4_1000_20261106_1228779
start / end : 2026-11-06 20:18:02 / 2026-11-07 15:06:34 KST (COMPLETED, wall 18:48:32)
last update : 999 (completed_updates 1000), NaN 0, Traceback 0
Top-5 at end (online score, best_top5/leaderboard.json):
  top1: update 846  score 0.8986
  top2: update 521  score 0.8949
  top3: update 927  score 0.8944
  top4: update 861  score 0.8943
  top5: update 519  score 0.8936
files: top<rank>_<update>.pt, model_999.pt (final), model_100/200/300/400/500.pt, params/, leaderboard.json,
       completion.json, events.out.tfevents.*, SHA256SUMS.txt (all .pt)
