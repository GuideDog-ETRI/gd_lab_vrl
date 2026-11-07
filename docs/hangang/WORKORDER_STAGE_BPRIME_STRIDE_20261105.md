# 한강 작업 지시: Stage B′ + 백스트라이드(A18L) — 좁은 통로 + 갭 대각 트롯, A18L 모델 출발 — 2026-11-05

**상태: 제출(사용자 결정 11-05 "한강에 바로 제출하고, 같은 시각에 A6000에서도 +300 it 이어 학습").** A6000 이어 학습
(bps model_299 → +300, 약 2.5 h)에서 막힌 통로·막다른 길 서기가 오르지 않으면 이 작업을 취소한다(사용자 판단).

앞 문서: `WORKORDER_STAGE_BPRIME_20261101.md`(B′ 레시피·충돌 점검·E2·운용 규칙 — 이 문서는 바뀐 것만 적는다).

## 1. 코드와 출발 모델

| 항목 | 값 |
|---|---|
| 코드 | origin/gh200 = c0ef6b1 위 이 문서 커밋(B′ fc7d4e9 + `gd_lab.gast_backstride.bprime`) |
| 과제 | `Gd-GastNarrowBPrimeStride-Rbq10-Dreamwaq-v0`, 트레이너 `scripts/gast_backstride/train_bprime_stride.py` |
| 출발·앵커(STAGE_A) | `checkpoints/gast_narrow/backstride_20261105/backstride_a18l_299.pt`, sha256 `a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e` (A18L, a1993 출발 300 it, 과제 `Gd-GastGaitV3BackStride`) |
| 갭 스위치 | `scripts/gast_backstride/a18l_env.sh`(sbatch가 source): 전·후진 R17 끔, 대각 트롯 −3, 걸음 하한 0.50 m / −6, 같은 쪽 착지 간격 −10, 건너편 끝 0.10 m 지나 착지 −20, 후진 갭 ≤ 0.20 m |
| 규모 | 16노드 × 4 GPU × 2,048 = 131,072 env, 2,000 update, 저장 50마다, 24 h 한도 재제출 체인(sbatch 그대로) |
| 실험 이름 | `gast/lineage16_bprime_stride`, Top-5 `online_top5_lineage16_b_v25p.json` |

## 2. A6000 사전 검증 (11-05, 5,120 env × 300, A18L 출발, E2 nominal)

| 문제 정의 | 결과 | 반영 방향 |
|---|---|---|
| 갭 건넘 / 넘어짐 | 96.6 / 3.4 %(A18L 97.2 / 2.7), clean 68.5 %(A18L 58.2) | 유지 |
| 후진 갭 L11 0.3 m/s | 넘어짐 10.4 %, 머리 20° 초과 10.4 %(A18L 8.3 / 2.1) | 감시 항목 |
| 계단 통과 / 넘어짐 | 95.2 / 0.8 % | 유지 |
| 통로 통과 | 77.0 %(bpfin 81.6), 무접촉 68.8 % | 오르는 중 |
| 막힌 통로 / 막다른 길 서기 | 0 / 0 %(bpfin 0 / 9.4) | B′ 첫 300 update와 같은 상태. B′는 다음 300 update에 88~98 %. A6000 이어 학습으로 확인 |
| MuJoCo | 6코스 완주, 통로 코스 5/5 구간 | — |

## 3. 판정 (최종 E2 `configs/gast_narrow/e2/stage_bprime_stride_hangang.yaml`)

B′ 문서 §1 그대로, 갭 기준만 A18L(stageA):
- 통로: 602 − 1 %p 이내, 막다른 길 서기 ≥ 90 %, 막힌 통로 침범 ≤ 5 %, 후진 통로 ≥ 전진 − 5 %p.
- 갭: 건넘 ≥ A18L − 1 %p, 후진 0.3 m/s 정지 없음, 대각 트롯·3발 접지·걸음(갭 리플레이)이 A18L 수준. 실전 기준 갭 통과 ≥ 90 %.
- 계단·내리막 경사: A18L − 1 %p 이내.

## 4. 제출 명령 (한강 세션, 모든 원격 명령 `timeout -k 10 120` 이하)

```bash
G=/scratch/t515a05/gd_lab
squeue -u $USER                                        # 이 작업본에서 도는 학습·E2가 없을 것
timeout -k 10 120 git -C $G/gd_lab_vrl fetch origin gh200
timeout -k 10 60  git -C $G/gd_lab_vrl merge --ff-only origin/gh200
git -C $G/gd_lab_vrl log --oneline -1                  # 이 문서 커밋
S=$G/gd_lab_vrl/checkpoints/gast_narrow/backstride_20261105/backstride_a18l_299.pt
sha256sum $S                                           # a1493d04...60b8e
cp $G/gd_lab_vrl/scripts/gast_narrow/hangang/lineage16_b.sbatch $G/jobs/
cd $G/jobs && LINEAGE_RECIPE=bprime_stride STAGE_A=$S STAGE_A_SHA256=a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e \
  timeout -k 10 60 sbatch --test-only lineage16_b.sbatch
cd $G/jobs && LINEAGE_RECIPE=bprime_stride STAGE_A=$S STAGE_A_SHA256=a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e \
  timeout -k 10 60 sbatch lineage16_b.sbatch
```

시작 확인(첫 update 로그 뒤 1회 보고): `source_kind: backstride`, actor 235→253 확장, `params/env.yaml`에
`BPrimeNarrowBackGapVelocityCommand`, `gap_support: null`, `back_gap_stride` 하한 0.5, 보존 앵커 비율 로그, NaN 없음.

## 5. 감시·중간 E2

B′ 문서 §7·§9 그대로: 중간 E2 update 400 / 800 / 1,200 / 1,600(요청 시), 두 번 연속 평탄하면 STOP(사용자 결정). 감시
항목: 막힌 통로·막다른 길 서기, 후진 갭 L11 0.3 넘어짐·머리 들림, 대각 트롯, 갭·계단 유지. 한강 세션은 대기(모니터·루프
없음), 보고는 요청 시 1회, push하지 않는다.
