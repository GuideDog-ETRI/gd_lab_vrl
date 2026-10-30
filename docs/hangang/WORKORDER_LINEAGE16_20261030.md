# 작업서: 100 Hz 보행정책 계보 재학습 (한강 16노드, 2단계)

2026-10-30 · B.S.Seo (ETRI) · 브랜치 `gh200`

## 목적

3906 → 6566 → 1748 → 1111 → 465 → 371 → v2.3으로 6단계에 걸쳐 만든 기능을 **2단계 학습**으로 다시 만든다.
결과물은 v2.3(it602)과 같은 입력·출력 구조의 정책 하나다(MuJoCo 배포 경로 그대로 사용).

| 단계 | 출발 | 학습 내용 | 앵커 | 기본 update |
|---|---|---|---|---|
| A. 후진 갭 | 3906 (`f3e7afe5…`) | 6566 레시피 그대로: 과제 `Gd-GastRayIrV21Gap50BackV2` (후진 명령 0.4→0.7, 이동 방향 기준 clean, 후진 앞쪽 들림 벌점) | 없음 | 1,500 |
| B. 통로 전체 | A의 E2 최고 모델 | v2.3 레시피 전체를 한 번에: LiDAR 지도·측면 여유 입력, 막히면 서기, 지형 44열, 통과 보너스·과속·멈춤 벌점, 안전 속도, 정강이 측정점, 갭 머리 들림 벌점 | A 모델 (통로가 아닌 곳, 머리 들림 표본 제외) | 1,500 |

B는 A 모델(LiDAR 없음)에서 시작한다. LiDAR 인코더는 새로 만들고, actor/critic 첫 층의 새 입력 열은 0으로 넣는다
(시작 시점 정책 = A 모델). 1748을 6566에서 만들 때와 같은 방식이다(`train_narrow_v23.py`).

## 자원과 다른 사용자 배려

- **16노드 × GPU 4장 = 64장** (natStr 522노드의 약 3%). 학습 작업은 한 번에 하나만 띄운다.
- GPU당 env 2,048 → 전체 131,072 env. iteration당 시간 예상 약 50~55 s (1노드 46.5 s, 118노드 59~64 s 실측 사이 값, 16노드 실측 아님).
- 작업당 24 h 제한: 1,500 update ≈ 21~23 h로 작업 하나에 들어가도록 잡았다. 넘치면 이어 학습(아래).
- 시작 실패·멈춤 노드 3대는 미리 제외(`--exclude=node1730,node1395,node1402`).
- E2는 학습이 끝난 뒤 1~2노드로 돌린다(`e2_multi.sbatch`).

## iteration(update) 수를 정한 근거

| 근거 | 내용 |
|---|---|
| update당 정책 변화가 작음 | lr 1e-4 고정, iteration당 경사 단계 20회(5 epoch × 4 mini-batch)로 데이터 양과 무관. 데이터를 늘려도 필요한 update 수는 비례해 줄지 않는다 |
| 그래도 데이터가 많으면 update 효율이 오름 | 16노드는 6566 학습(1,024 env)의 128배, v2.2(4,096 env)의 32배. 잡음이 줄어 같은 update에서 더 안정적으로 나아간다 |
| A 참고 | 6566은 1,024 env에서 8,000 update. 16노드에서 얼마나 줄어드는지는 측정 전 → 1,500으로 잡고 300·600·1,000·1,500에서 E2 |
| B 참고 | 기존 계보는 통로 기능에 v1~v2.3 합계 약 6,700 update(작은 규모 위주). v2.3 128노드는 이어 학습 614 update 중 약 100 update에서 막다른 길 서기 92.9% 도달 → 1,500으로 잡고 500·1,000·1,500에서 E2 |
| 연장 기준 | 마지막 두 E2 사이에 주요 지표(후진 갭, 막다른 길 서기, 열린 통로 통과)가 2%p 넘게 오르고 있으면 500 update 연장 |

## 학습 설정

| 항목 | 값 | 근거 |
|---|---|---|
| PPO | 5 epoch × 4 mini-batch, schedule fixed, policy lr 1e-4, terrain lr 3e-4, clip 0.2, γ 0.99, λ 0.95, 엔트로피 0.01, horizon 100 | 8노드 A/B 4종 중 가장 좋음 (scale notes 8절) |
| 저장 | 50 update마다 (`NARROW_SAVE_EVERY=50`), Top-5는 매 update | 정지 붕괴 구간을 피해 E2로 고를 후보 확보 |
| 정규화 통계 | A: `GD_LAB_NORMALIZER_DEFER=0` (필수), B: 1 | 지연 동기화는 앵커 runner만 flush한다. A 과제에는 앵커가 없어 1이면 정규화가 갱신되지 않음 (`train_lineage_a.py`가 막음) |
| 램프 | A 첫 작업: 과제 기본값 (후진 비율 200 update, 속도 상한, 손잡이 외력) — 6566 학습과 같음. A 이어 학습·B: 0 | `NARROW_KEEP_RAMPS=1` / `LINEAGE_CONTINUE=1` |
| Top-5 기준 A | `configs/gast_backgap/online_top5.json` (6566 학습과 같음) | |
| Top-5 기준 B | `configs/gast_narrow/online_top5_lineage16_b.json`: 막힌 통로 정지 ≥ 0.85, 침범 ≤ 0.05 (새 기준), 갭 레벨 ≥ 8.5, 갭 넘어짐 ≤ 0.145, 계단 넘어짐 ≤ 0.078 | 128노드 run에서 정지 회복 중 모델이 Top-5에 들어감. 갭 레벨은 env 수에 따라 평형이 달라짐 |

Top-5 기준은 128노드 값에서 출발한 임시값이다. 16노드는 표본이 적어 흔들림이 크므로, **B 200 update 시점에
E2로 확인한 기준점 + 흔들림 2 SD로 다시 정한다** (파일은 실행 중 다시 읽힘).

## 실행 순서

```bash
cd /scratch/t515a05/gd_lab/gd_lab_vrl && git fetch origin && git checkout gh200 && git pull --ff-only
cd scripts/gast_narrow/hangang

# A: 3906에서 후진 갭 (16노드, 1,500 update)
sbatch lineage16_a.sbatch
# A가 24 h 안에 안 끝나면: 마지막 저장본에서 이어 학습 (PPO 옵티마이저는 새로, 램프 재시작 없음)
NARROW_SOURCE=<.../model_N.pt> NARROW_SOURCE_SHA256=$(sha256sum <.../model_N.pt> | cut -d' ' -f1) \
  NARROW_UPDATES=<남은 수> LINEAGE_CONTINUE=1 sbatch lineage16_a.sbatch

# A E2 (300·600·1,000·1,500 저장본 + 6566 비교) → 후진 갭·전진 갭·계단이 6566 이상인 최고 모델 선택

# B: A 최고 모델에서 통로 전체 (16노드, 1,500 update)
STAGE_A=<A 최고 .pt> STAGE_A_SHA256=<sha> sbatch lineage16_b.sbatch
# 이어 학습: STAGE_A/STAGE_A_SHA256는 그대로, NARROW_SOURCE=<B의 model_N.pt> NARROW_SOURCE_SHA256=<sha> NARROW_UPDATES=<남은 수>
```

E2 설정은 `configs/gast_narrow/e2/v23_check_gh200.yaml`(칸당 512 env)을 복사해 모델 목록만 바꾼다.
비교 기준: A는 6566, B는 6566·371·it602.

## 판정 기준 (E2, 기본 조건)

| 단계 | 통과 기준 |
|---|---|
| A | 후진 갭 건넘 ≥ 6566 − 1%p, 전진 갭 건넘·계단 통과 ≥ 6566 − 1%p, 후진 앞쪽 들림 ≤ 6566 |
| B | it602 대비: 막다른 길 서기, 막힌 통로 서기·침범, 열린 통로 통과, 갭 전진 머리 들림이 같거나 좋음, 갭·계단은 6566 − 1%p 이내. 실전 사용 조건(갭 90% 이상) 유지 |

## 감시와 중단

- 진행: `logs/gast/lineage16_{a,b}/<run>/` tensorboard, rank 0 로그. 50 update마다 저장 확인.
- 멈춤: `watchdog_v23.sh`의 감시 방식(노드 제외 후 마지막 저장본에서 재시작)을 그대로 쓰되, 재시작 명령은 위 이어 학습 명령으로 바꾼다.
- 조기 종료: `touch logs/launches/$GAST_RUN_ID/STOP` (저장 후 종료).
- 막힌 통로 정지가 학습 중 1~3%로 무너지는 구간은 128노드에서 4번 있었고 스스로 회복했다. 그 구간의 저장본은 E2 후보에서 뺀다.

## 바뀐 코드 (이 작업서와 함께 커밋)

| 파일 | 내용 | 기존 run 영향 |
|---|---|---|
| `scripts/gast_narrow/train_narrow_v23.py` | 후진 갭 체크포인트(6566·A 모델)에서 시작하는 경로 추가 (LiDAR 인코더 새로, 첫 층 0 열 확장) | 없음 (371·v2.3 경로 그대로) |
| `scripts/gast_narrow/train_narrow_v23.sh` | `NARROW_TRAIN_PY` / `NARROW_TASK` / `NARROW_EXPERIMENT` / `NARROW_SAVE_EVERY` / `NARROW_KEEP_RAMPS`, Top-5 기준 파일 외부 지정 | 없음 (기본값 = 기존 값) |
| `scripts/gast_narrow/train_lineage_a.py` | A단계: `train_backgap.py`에 다중 노드 설정(rank GPU, PG timeout, 스택 덤프)만 더해 실행 | 새 파일 |
| `src/gd_lab/gast_narrow/narrow_top5_v2.py` | Top-5 기준에 `min_narrow_hold_rate` / `max_narrow_breach_rate` (기준 파일에 있을 때만) | 없음 (키가 없으면 기존과 같음) |
| `configs/gast_narrow/online_top5_lineage16_b.json` | B단계 Top-5 기준 | 새 파일 |
| `scripts/gast_narrow/hangang/lineage16_{a,b}.sbatch` | 16노드 작업 스크립트 | 새 파일 |

로컬 검증 (2026-10-30, RTX 5090 1장 smoke, 192 env × 16 step × 5 update): A는 3906에서 BackV2 과제로 시작해 램프가 기본값(속도 300,000 step, 외력 150,000 step)으로 켜졌고, B는 6566(후진 갭 체크포인트)에서 시작해 LiDAR 인코더 10개 텐서를 새로 만들고 actor 235→252, critic 298→307 열을 0으로 넓혀 5 update까지 정상 종료. 16노드 실행은 한강에서 처음 확인한다.
