# 한강 작업 지시: Stage A′ (후진 갭 + 보행 v3 G4, 3906 출발) — 2026-11-01

## 1. 목적

3906에서 출발해 후진 갭(6566 레시피)과 보행 품질(GAIT_V3_DESIGN.md §3.7 G4)을 함께 학습한다. 다음 단계는 B′(통로) → C1(낭떠러지) → C2(제자리 회전)이고, 모두 이 모델에서 출발한다.

| 근거 (a6000, 300 update, E2 넘어짐 %) | 계단 오르기 | nose 오르기 | 후진 갭 | 내리막 경사 | 전진 갭 머리 20° 초과 |
|---|---|---|---|---|---|
| 3906 (출발점) | — | 17.2 | 70–100 | 3.1 | — |
| G3 (보행 항목 전부, 처음부터 최대) | 5.2 | 20.3 | 7.0 | 12.5 | 6.6 |
| G-null (보행 항목 없음) | 1.6 | 1.6 | 1.4 | 3.1 | 42 |
| 6566 (기준, 6,566 update) | 1.3 | 4.7 | 0.4 | 0 | 13.5 |

G4 = G3 + 후진 갭에서 발 속도 감점 끄기(R19) + 갭 지지 −1.0 + 보행 가중치 ramp(0.2→1, 200 update) + 경사에서 보행 항목 다시 켜기. a6000에서 같은 레시피 300 update 확인이 이 작업과 함께 돈다(11-01 20:40경 E2). **a6000 G4가 기준에 못 미치면 사용자가 이 작업을 대기 중 취소(scancel)한다.**

## 2. 실행

| 항목 | 값 |
|---|---|
| 코드 | `origin/gh200`(d90e7cd 이후, fast-forward) |
| 스크립트 | `scripts/gast_gait/hangang/aprime16.sbatch` (`$G/jobs/`에 복사해 제출) |
| 규모 | 16노드 × GPU 4 × 2,048 env = 131,072 env, 100 step/update, **2,000 update**, 50 update마다 저장 |
| 출발 | 3906_top1.pt (sha f3e7afe5…4d98), 새 PPO 옵티마이저, 지형 단계 이어받음 |
| 과제 | `Gd-GastGaitV3Gap-Rbq10-Dreamwaq-v0` (6566 과제 + 보행 v3 G4 항목) |
| Top-5 | 매 update 비교, `configs/gast_gait/online_top5_g2.json` (갭 종료율 ≤ 0.35 등) |
| 시간 | update당 약 1분 예상 → 약 34 h = job 2개 (23:50 제한, USR1로 저장 후 자동 재제출, 재제출 job은 ramp 없이 전체 가중치) |
| SRU | 16노드 × 약 34 h ≈ 544 노드·시간 ≈ 65만 SRU-초 (노드 1시간 = 1,200 SRU-초) |

제출(한강 세션, 1회):
```
timeout -k 10 120 git -C $G/gd_lab_vrl fetch origin gh200
timeout -k 10 60 git -C $G/gd_lab_vrl merge --ff-only origin/gh200     # 다른 학습이 이 작업본에서 돌고 있지 않을 때만
cp $G/gd_lab_vrl/scripts/gast_gait/hangang/aprime16.sbatch $G/jobs/
cd $G/jobs && timeout -k 10 60 sbatch --test-only aprime16.sbatch       # 시작 예상 시각을 먼저 보고
cd $G/jobs && timeout -k 10 60 sbatch aprime16.sbatch                    # 사용자 승인 후
```

시작 확인(첫 update 로그, 보고 1회): `stage=Aprime done_before=0`, `Curriculum/gait_weight_ramp/kappa` ≈ 0.2에서 증가, `Episode_Reward/gap_support` 존재, `lead_foot_stall` 없음, NaN 없음.

## 3. 중간 E2 (무인)

`aprime16.sbatch`가 update 300 / 600 / 1,000 / 1,500에 도달하면 1노드 E2(`e2_multi.sbatch`, 설정 `configs/gast_gait/e2/aprime_hangang.yaml` 복사본)를 스스로 제출한다. 기준 모델 3906, 6566. 결과: `logs/e2/aprime_mid_<job>_u<N>/summary.md`.

| 판정 (넘어짐 %) | 계속 | 멈추고 보고 |
|---|---|---|
| 계단 오르기 L9 / nose 오르기 | 3906 이하 | 3906보다 2 %p 넘게 나쁨 |
| 후진 갭 평균 | 직전 E2보다 나아지거나 같음 | 직전보다 2 %p 넘게 나빠짐 |
| 내리막 경사 | ≤ 6.2 (32 env 중 2) | 그 이상이 두 번 연속 |

멈출 때: `touch $G/gd_lab_vrl/logs/launches/<GAST_RUN_ID>/STOP` (저장 후 종료) → 다음 job이 이미 제출됐으면 `scancel`. 멈춤 결정은 사용자가 한다.

## 4. 최종 판정과 선택

- 계단 오르기·내리기 6566 대비 1 %p 이내, 후진 갭 넘어짐 ≤ 1 %, 내리막 경사 ≤ 3.1 %, 전진 갭 머리 20° 초과 ≤ 약 10 %.
- 후보: 모든 `best_top5/*.pt` + 마지막 `model_*.pt`. 선택된 모델은 로컬 MuJoCo(평지 머리 들림·착지·발 높이, 후진 갭)로 다시 확인한 뒤 B′ 출발점이 된다.

## 5. 운용 규칙

- 원격 명령은 모두 `timeout -k 10 120` 이하.
- 한강 세션은 대기(모니터·루프 없음). 확인은 요청이 있을 때만.
- 한강에서는 push하지 않는다.
