# 학습 계보와 단계별 선정 모델 (2026-11-07 정리)

git에는 **단계별 선정 모델만** 둔다(`<단계>/<번호_특징_서버>/`). 그 밖의 Top-5·탐색 모델은 로컬 `legacy/`에만 있고 git 밖이다
(.gitignore `*.pt`).
폴더 이름 = `<계보 번호>_<특징>_<학습 서버>`. 이름만 보고 계보를 읽도록 했다: 번호 n은 번호 n−1의 모델에서 이어 학습했다(⓪→①→②→③→④→⑤→⑥, 한 줄 계보; ⑥의 앵커만 ③). 이터레이션 번호와 모델 이름(DWB-38000, 3906, a1993, A18L, bps_cont 299, v22 299, it861)은 폴더 이름에 넣지 않고 아래 표와 각 폴더의 파일 이름에 남긴다. 서버: `hangang`(KISTI GH200), `a6000`(onvlm2 RTX PRO 6000), `5090`(90.111 RTX 5090), `5090_local`(로컬 RTX 5090). 각 모델 폴더에 `params/`와 `SHA256SUMS`가 있고, `sha256sum -c SHA256SUMS`로 확인한다.

| 단계 | 위치 | 모델 (sha256 앞 8자리) | 어디서 / 규모 | 레서피와 특징 | 코드 커밋 |
|---|---|---|---|---|---|
| ⓪ blind | `00_blind/0_blind_walk_a6000/teacher/model_38000.pt` | DWB-38000 (c6cfe178) | A6000(onvlm2) 1 GPU, 4,096 env | 블라인드 기본 보행 | — |
| ① Ray+IR | `01_ray_ir/1_rayir_terrain_gap50_5090/best_top5/3906_top1.pt` | 3906 (f3e7afe5) | 90.111, 1,024 env × 5,000 | Ray+IR 지형 입력 교사, gap 50 | — |
| ② gap | `02_gap/2_gap_fwd_back_quiet_gait_hangang/aprime_g5_it1993_job1203105_top3.pt` | A′ a1993 (73a872fc) | 한강 131,072 env × 2,000 | gait v3 G5(조용한 걸음, DOG_GAIT_REFERENCE) + 6566 후진 갭 + R20 | fe00896 |
| ③ 후진 보폭 | `02_gap/3_back_gap_long_diag_trot_5090_local/backstride_a18l_299.pt` | A18L (a1493d04) | 로컬 5090, 1,024 env × 300 | 보폭 하한 0.50 m(다리 길이 기준), 대각 트롯, 간격 0.30, 착지 여유 0.10, R17 끔, 후진 갭 ≤ 0.20 m | 881655a |
| ④ B′ 기초 | `03_narrow/4_narrow_base_a6000/bprime_base_a6000_it299.pt` | bps_cont 299 (9e7afa4c) | A6000 5,120 env × 600 (bps_fin 300 + cont 300) | STRIDE_STAGE=2(B′ 통로 + ③ 갭 항), 서기 씨앗 1~2.5 % | c0ef6b1 |
| ⑤ B′ 1단계 | `03_narrow/5_narrow_stand_first_a6000/bprime_stride_v22_299.pt` | v22 (1062d254) | A6000 5,120 env × 300 | STRIDE_STAGE=1: margin 끔, 정지 벌점 1.5 s / −1 → 서기 학습 | d7f1055 |
| ⑥ B′ 2단계 **최종** | `03_narrow/6_narrow_margin_final_hangang/bprime_stride_s2_hangang_it861.pt` | 한강 it861 (3fc9a125) | 한강 131,072 env × 1,000, job 1228779 | STRIDE_STAGE=2(B′ 그대로), 앵커 A18L. 11-07 E2로 Top-5·최종 중 선정 | abc0fab |
| 낭떠러지 | `04_cliff/` | — | to-do (T13) | — | aa160ba, 61fdbf1 |
| 제자리 회전 | `05_rotation/` | — | to-do (T13) | — | — |

- ⑥ 선정 근거: `03_narrow/6_narrow_margin_final_hangang/e2_final_candidates_summary.md` (A6000 E2, seed 42). 후진 갭 L11 v0.3 91.7 %(최종 999는 85.4),
  갭 건넘 98.3, 계단 95.7, 좁은 통로 95.8, 굽은 통로 94.5, 막힌 통로 서기 100 / 침범 0.
- 서기 앵커는 이 계보에서 표본 0개였다(NARROW_TURN=0). 서기는 ⑤의 보상으로 배웠다.
- 레서피 문서: `docs/recipes/GD_VRL_TRAINING_RECIPES.md`, 결과: `docs/recipes/GD_VRL_RESULTS_SHEET.md`.
- 이전 계보(6566 → 1748 → 1111 → 465 → 371 → 602 → 653 …)와 A′ Top-5 나머지는 `legacy/gast_narrow/`(로컬). 옛 설정 파일은 이 경로를 가리킨다.
