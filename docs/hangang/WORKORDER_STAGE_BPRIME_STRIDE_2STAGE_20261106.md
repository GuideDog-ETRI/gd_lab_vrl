# 한강 작업 지시: Stage B′ + 백스트라이드 2단계 (1단계 서기 → 2단계 margin) — 2026-11-06

**상태: 사용자 결정 11-06 "대안 A로 가자".** 앞 문서 `WORKORDER_STAGE_BPRIME_STRIDE_20261105.md`의 한 번에 하는 레시피
(job 1222681)는 취소. 이 문서는 바뀐 것만 적는다.

**수정 11-06 19:00 (사용자 결정): 1단계 job 1227288 취소, 2단계를 A6000 1단계 결과에서 바로 시작.**
A18L 출발 1단계는 135 update까지 막힌 통로·막다른 길 서기 0.0 %, 서기 보상 ≈ 0, 막다른 길 지형 레벨 0이었다. A6000에서도 A18L 출발
런(bps_fin)은 300 update 동안 막다른 길 레벨 0이었고, 서기가 1~2.5 % 남은 모델(bps_cont)에서 출발한 1단계(v22)만 레벨이 0 → 8.7로
올랐다. 1단계는 서기가 조금이라도 있는 모델에서만 작동한다. 그래서 1단계는 A6000 v22 결과를 쓰고, 한강은 **2단계만** 돌린다:
`NARROW_SOURCE` = `checkpoints/gast_narrow/bprime_stride_v22_20261106/bprime_stride_v22_299.pt`
(sha256 `1062d254f0b27d3981b7b0a08cdc2a2ff314038b7f852b7f8a15245f3d7ddf08`). A6000에서 같은 경로(v22 → 2단계 300 it)의 결과는
it236 통과 92.0 %, 서기 99.4 %, 갭 건넘 96.2 %, 계단 95.1 %. 아래 §2·§4의 1단계 행은 기록으로만 남긴다.

## 1. 왜 2단계인가

| 문제 정의 | 결과 | 반영 방향 |
|---|---|---|
| fc7d4e9 레시피(margin 50/50 + 정지 벌점 1.0 s / −2)를 처음부터 켬 | A6000 cont 300 it: 막힌 통로 서기 1–2.5 %. 리플레이: 통과 가능한 통로에서 서 있으면 −38(B′ 정지 벌점), 막힘 확률 86 % 이상이어야 서는 쪽이 이득 | 서기를 배우기 전에 margin·강한 정지 벌점을 켜지 않는다 |
| 1단계: margin 끔 + v2.5 정지 벌점(1.5 s / −1), 300 it | 막힌 통로 서기 92.2 %, 막다른 길 93.8 / 90.6 % | 서기를 먼저 배움 |
| 2단계: margin 켬(B′ 그대로), 300 it | 서기 99.1 %, 막다른 길 100 / 96.9 %, 통과 88.9 %(it236 92.0), 굽은 통로 75.0 / 65.6 %, 계단 95.1 %, 갭 96 % | B′ 최종과 같은 수준 |
| 후진 25 cm 갭(L14) | 학습 범위 밖(후진 ≤ 0.20 m). 1·2단계 모두 넘어짐 17–21 %(A18L 6 %). 0.26 m로 넓혀 학습(bg26) → 후진 L14 27 %, 후진 L11 19 %로 악화 | 판정에서 제외. 후진 25 cm는 gdm 운용 규칙(넓은 갭 앞에서는 돌아서 전진) |

## 2. 코드와 출발 모델

| 항목 | 값 |
|---|---|
| 코드 | origin/gh200 = 이 문서 커밋. `BPRIME_STRIDE_STAGE`(gd_lab.gast_backstride.bprime), sbatch `STRIDE_STAGE=1|2` |
| STAGE_A (1단계 출발, 두 단계 모두 앵커·서기 앵커) | `checkpoints/gast_narrow/backstride_20261105/backstride_a18l_299.pt`, sha256 `a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e` |
| 1단계 | `STRIDE_STAGE=1`: margin 끔(`BPRIME_MARGIN_BALANCE=0`), 정지 벌점 `NarrowStallPenaltyV25` −1.0. 나머지(A18L 갭 항, 후진 갭 ≤ 0.20 m, 앵커)는 앞 문서와 같음. 800 update |
| 2단계 | `STRIDE_STAGE=2`, `NARROW_SOURCE` = **A6000 1단계 v22 model_299**(위 경로, PPO 상태 복원, source_kind `bprime_stride_continue`). B′ 그대로. 1,000 update |
| 규모 | 16노드 × 4 GPU × 2,048 = 131,072 env, 저장 50마다. 약 52 update/h → 1단계 약 15.5 h, 2단계 약 19.5 h(24 h 한도 안) |
| 실험 이름 | `gast/lineage16_bprime_stride_s1`, `gast/lineage16_bprime_stride_s2` |
| 운용 | **자동 재제출 없음**(sbatch 기본 `NO_RESUBMIT=1`), `NCCL_PROTO=^LL128`(sbatch 기본), sbatch·scancel은 사용자 확인 후 |

## 3. 판정 (E2 `configs/gast_narrow/e2/stage_bprime_stride_hangang.yaml`)

- 1단계 끝(또는 400 update 중간 E2): 막힌 통로 서기 ≥ 90 %, 막다른 길 ≥ 90 %, 갭·계단 A18L − 1 %p 이내. 통과하면 2단계 제출.
- 2단계: 앞 문서 §3 그대로. **갭 평균과 판정에서 `gap/backward/L14/*` 칸은 뺀다**(후진 25 cm는 학습 범위 밖, 기록만).
- 실전 기준: 갭 통과 ≥ 90 %, 계단 보행이 무너지지 않을 것.

## 4. 제출 명령 (한강 세션, 원격 명령 `timeout -k 10 120` 이하)

```bash
G=/scratch/t515a05/gd_lab
squeue -u $USER                                        # 이 작업본에서 도는 학습·E2가 없을 것
timeout -k 10 120 git -C $G/gd_lab_vrl fetch origin gh200
timeout -k 10 60  git -C $G/gd_lab_vrl merge --ff-only origin/gh200
git -C $G/gd_lab_vrl log --oneline -1                  # 이 문서 커밋
S=$G/gd_lab_vrl/checkpoints/gast_narrow/backstride_20261105/backstride_a18l_299.pt
sha256sum $S                                           # a1493d04...60b8e
cp $G/gd_lab_vrl/scripts/gast_narrow/hangang/lineage16_b.sbatch $G/jobs/
# 1단계 (사용자 확인 후)
cd $G/jobs && LINEAGE_RECIPE=bprime_stride STRIDE_STAGE=1 NARROW_UPDATES=800 STAGE_A=$S \
  STAGE_A_SHA256=a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e timeout -k 10 60 sbatch --test-only lineage16_b.sbatch
cd $G/jobs && LINEAGE_RECIPE=bprime_stride STRIDE_STAGE=1 NARROW_UPDATES=800 STAGE_A=$S \
  STAGE_A_SHA256=a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e timeout -k 10 60 sbatch lineage16_b.sbatch
# 2단계 (사용자 확인 후): C = A6000 1단계 결과 v22 model_299
C=$G/gd_lab_vrl/checkpoints/gast_narrow/bprime_stride_v22_20261106/bprime_stride_v22_299.pt
sha256sum $C                                           # 1062d254...ddf08
cd $G/jobs && LINEAGE_RECIPE=bprime_stride STRIDE_STAGE=2 NARROW_UPDATES=1000 STAGE_A=$S \
  STAGE_A_SHA256=a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e \
  NARROW_SOURCE=$C NARROW_SOURCE_SHA256=1062d254f0b27d3981b7b0a08cdc2a2ff314038b7f852b7f8a15245f3d7ddf08 timeout -k 10 60 sbatch --test-only lineage16_b.sbatch
cd $G/jobs && LINEAGE_RECIPE=bprime_stride STRIDE_STAGE=2 NARROW_UPDATES=1000 STAGE_A=$S \
  STAGE_A_SHA256=a1493d04a97afde7f415c026c1e34bf902f4edbd080357764aeb999468a60b8e \
  NARROW_SOURCE=$C NARROW_SOURCE_SHA256=1062d254f0b27d3981b7b0a08cdc2a2ff314038b7f852b7f8a15245f3d7ddf08 timeout -k 10 60 sbatch lineage16_b.sbatch
```

시작 확인(첫 update 로그 뒤 1회 보고): `source_kind`(1단계 `backstride`, 2단계 `bprime_stride_continue`),
`params/env.yaml`의 `narrow_stall.func`(1단계 `tasks_v25:NarrowStallPenaltyV25` / −1.0, 2단계 `BPrimeStallPenalty` / −2.0),
환경 변수 `BPRIME_MARGIN_BALANCE`(1단계 0, 2단계 1), 보존 앵커 비율 로그, NaN 없음.

## 5. 감시·중간 산출

- (1단계 취소로 해당 없음) 1단계 **model_100.pt**가 나오면 `model_100.pt` + `params/` 를 tar로 묶어 사용자에게 전달(업로드) — MuJoCo 갭·계단·좁은
  통로 녹화로 걸음걸이 경향을 본다. 학습은 계속 돈다.
- 2단계 model_100.pt가 나오면 같은 방식으로 묶어 전달(MuJoCo 막다른 길 서기·갭·계단 확인). 중간 E2: 2단계 300 update(요청 시). 한강 세션은 대기(모니터·루프 없음), 보고는 요청 시 1회, push하지 않는다.
