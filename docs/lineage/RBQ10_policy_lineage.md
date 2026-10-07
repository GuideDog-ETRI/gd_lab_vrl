# RBQ10 교사·학생 정책 계보

2026-10-07 작성, 2026-10-08 갱신 · Claude Docs 문서를 내보낸 사본 (원본이 기준)

## 요약

지금 교사 계보는 블라인드 d\_v3.6.21\_b1\_18 → BIVT-Ray 17206 → BIVT-Ray Clean 21068 → GAST 교사 v2.1(학습 중)입니다. 세 학생(GAST, RVLD, GAVD)은 이 GAST 교사가 끝나면 같은 교사에서 다시 증류할 계획입니다.

- 교사는 시뮬레이터의 높이맵을 보고 PPO로 걷는 법을 배웁니다. 학생은 카메라 영상만 보고 교사의 지형 latent와 행동을 따라 하도록 증류합니다.
- 모든 교사와 학생은 같은 DreamWaQ 보행 골격을 씁니다. 바뀌는 것은 지형 정보를 32차원 latent로 만드는 인코더뿐입니다.
- 약어는 논문 공식 명칭이 아니라 프로젝트 내부 이름입니다.

## 계보도

```text
교사 계보
  DWB d_v3.6.21_b1_18 (블라인드)
    └─ BIVT-Ray 17206 (vendor_legacy 카메라)
         └─ BIVT-Ray Clean 21068 (vendor_new, SHA fa397d22…)
              ├─ BIVT-Ray Clean 31625 ── BIVT-Ray v2 32953 / 34773 / 35908 (중단, 36201)
              └─ GAST 교사 v2.1 (21068 warm start, 학습 중) ── 업데이트 2000 (SHA db500e96…)

학생 (교사 21068)            GAST 학생 19840 (갭 집중 24960) · RVLD 19008 · GAVD 19008
학생 (교사 GAST v2.1 2000)   GAST 학생 증류 중 (일반 Top-1 17200, 갭 Top-1 7456)
```

그림 원본(도식): [Claude Docs 계보 문서](https://claude.ai/code/artifact/2fa8d569-e8f6-44ed-a536-a1c150e80e0d)

교사는 위에서 아래로 이어집니다. 31625에서 갈라진 v2 가지는 중단됐고, 배포 중인 학생 세 종의 현재본은 모두 21068에서 증류했습니다.

## 1. 블라인드 백본: DWB d\_v3.6.21\_b1\_18

**DWB = DreamWaQ Blind Baseline** (DreamWaQ 블라인드 기준 정책)

**특징**

- 카메라나 높이맵 없이 고유감각(관절, IMU)만으로 걷는 정책입니다. 지형은 발이 닿은 뒤에야 알 수 있습니다.
- 이후 모든 교사의 출발점입니다. 배포 저장소의 `resources/policy/dwb/d_v3.6.21_b1_18/`에 있습니다.
- 기반 방법은 [DreamWaQ (Nahrendra 외, 2023)](https://arxiv.org/abs/2301.10602)과 [PPO (Schulman 외, 2017)](https://arxiv.org/abs/1707.06347)입니다.

**네트워크 구성**

- **관측:** 한 시점 46차원(각속도 3, 중력 방향 3, 속도 명령 3, 관절 위치 12, 관절 속도 12, 이전 행동 12, payload 1). 5시점 이력은 230차원입니다.
- **CENet (Context-Aided Estimator Network, 상황 보조 추정망):** 이력 230 → MLP 128 → 64 → 몸통 속도 추정 3 + latent 16. VAE 방식으로 학습합니다.
- **Actor:** 최근 4시점 184 + CENet 코드 19 = 203 → 512 → 256 → 128 → 관절 행동 12 (ELU MLP).
- **Critic (학습 전용):** 특권 관측 298 → 512 → 256 → 128 → 가치 1. 실제 속도, 마찰, 질량, 높이 스캔을 봅니다.

## 2. BIVT-Ray 교사: 17206

**BIVT-Ray = Blind-Initialized Visual Teacher, Raycast Visibility** (블라인드 정책에서 초기화한 지형 관측 교사, 레이캐스트 가시성 버전)

**특징**

- DWB 가중치를 복사해 시작합니다. Actor 첫 층의 지형 입력 32열을 0으로 두어, 처음에는 블라인드와 같은 행동을 합니다.
- 지형은 메시를 레이캐스트해 얻은 높이맵입니다. 카메라에 보이는 칸만 남기고, 화각·거리·지형 가림·다리 가림으로 가려진 칸은 무효로 처리합니다. 렌더링이 필요 없어 학습이 빠릅니다.
- 보이는 칸이 하나도 없으면 지형 latent를 정확히 0으로 만듭니다(지형 게이트). 지형 입력 전체를 끊는 blackout도 학습합니다.
- 한 시점의 높이맵만 봅니다. 지나간 지형을 기억하는 구조는 없습니다.

**네트워크 구성**

- **지형 입력:** 11×17 격자의 높이 187 + 유효성 187 = 374차원.
- **지형 인코더 (HeightScanCNN):** Conv2d 2→8→16 (3×3, ReLU) → 5×8 pooling → Flatten 640 → MLP 64 → latent 32 (tanh).
- **Actor:** 203 + 지형 latent 32 = 235 → 512 → 256 → 128 → 12. CENet과 Critic은 DWB와 같습니다.
- **학습:** PPO가 Actor, Critic, 지형 인코더를 함께 갱신합니다. CENet은 별도 보조 손실로 학습합니다.

**체크포인트:** `17206_top1.pt` (2026-10-03, 관측 버전 `bivt_ray_occlusion_v2`, 카메라 프로파일 `vendor_legacy`). 대응 학생은 GAST 학생(legacy 카메라)과 GAVD 학생(캡처 11,008회에서 중단)입니다.

## 3. BIVT-Ray Clean 교사: 21068 / 31625

**BIVT-Ray Clean = BIVT-Ray Clean Gap Fine-tuning** (갭을 모서리에 닿지 않고 깨끗하게 건너도록 파인튜닝한 BIVT-Ray 교사)

**특징**

- 17206에서 이어 학습했습니다. 작업은 `Gd-VrlGapFinetuneCleanRaycast-Rbq10-Dreamwaq-v0`이고, 2026-10-05 22:54에 서버 GPU 3장(4096 env)으로 시작했습니다.
- 카메라 프로파일을 실기와 같은 `vendor_new`로 바꿨습니다.
- 갭 성공을 clean crossing(모서리나 갭 안쪽 접촉 없이 통과)으로 셈니다.

**네트워크 구성:** BIVT-Ray와 같습니다. 보상과 평가 기준만 다릅니다.

**체크포인트**

| 체크포인트 | 나온 곳 | 점수 / clean 통과율 | 뒷발 동시 공중 비율 / 최고 높이 | 쓰임 |
| --- | --- | --- | --- | --- |
| 21068 | Clean run 초반 Top-1 | 0.865 / 0.956 | 0.009 / 30 cm | GAST v2.1 교사의 출발점, 현재 학생 3종의 교사 |
| 31625 | 같은 run 후반 Top-1 | 0.884 / — | 0.129 / 47 cm | BIVT-Ray v2 검증 학습의 출발점 |

- 31625는 점수는 더 높지만, 뒷발 두 개를 함께 높이 들어 갭을 넘는 습관이 생겨 있었습니다.
- **BIVT-Ray v2 가지 (중단):** 31625에서 v2 보상(갭 모서리 마진, 계단 손잡이 외력)으로 이어 학습했습니다. Top 체크포인트는 32953, 34773, 35908입니다. 뒷발 습관이 남아 2026-10-07 11:00에 36201에서 멈추고 GAST로 전환했습니다.

## 4. GAST 교사 v2.1: 21068에서 warm start (학습 중)

**GAST = Gap-Aware Spatiotemporal Teacher-Student Learning** (갭을 의식하는 시공간 교사-학생 학습). 교사(GAST-T)와 학생(GAST-S)을 모두 포함하는 프로젝트 설계입니다.

**특징**

- 카메라 가림 없는 전체 높이맵을 봅니다. 뒷발 아래가 카메라에 가려 안 보이는 BIVT-Ray의 문제를 피하려는 것이 전환 이유입니다.
- **시간 기억:** 최근 4.8초 안에서 8시점(0.1초 단위 오프셋 48, 32, 16, 8, 4, 2, 1, 0)의 높이맵을 현재 로봇 위치에 맞춰 정렬해 넣습니다. 몸통이 지나간 갭을 뒷발이 더 잘 알 수 있게 하려는 구조입니다.
- **warm start:** BIVT-Ray 21068의 Actor, Critic, CENet, 관측 정규화, 행동 std를 그대로 복사합니다. 지형 인코더와 복원 decoder만 새로 만들고, 인코더 마지막 층을 0으로 두어 출발 시 latent가 0입니다. 학습률은 보행 쪽 1e-4, 새 지형 쪽 1e-3입니다.
- **v2.1 보상:** 갭 모서리 마진, 계단 손잡이 당김·밀기 외력에 Codex 교차 검토 내용을 더했습니다(`claude_handoff/GAST_TEACHER_V21_FINAL_REWARD_20261007.md`).

**네트워크 구성**

- **지형 입력:** 8시점 × (높이 187 + 유효성 187 + 관측 나이 1) = 8 × 375.
- **지형 인코더 (TemporalTerrainEncoder):** 시점마다 공유 CNN(Conv2d 2→8→16, 3×3, ELU → 5×8 pooling → 640 → 64) + 나이·유효성 embedding 64 → 시간 self-attention(64차원, 4 head) + LayerNorm → GRU 64 → latent 32 (tanh).
- **복원 decoder (학습 전용):** latent 32 → 128 → 187칸 × 6. 깨끗한 높이, 유효성, 갭, 상승·하강 경계, 지지면을 복원하는 보조 손실입니다.
- **Actor, Critic, CENet:** BIVT-Ray와 같은 모양입니다(Actor 235 → 512 → 256 → 128 → 12).

**실행:** 작업 `Gd-GastGapCleanV21-Rbq10-Dreamwaq-v0`, 2026-10-07 11:02 시작, 서버 GPU 3장, 4096 env, 3만 업데이트. 22:22 기준 2,572/30,000 업데이트이고, 업데이트당 약 15.7초 걸립니다.

**업데이트 2000 평가 (MuJoCo, 2026-10-07):** 0.65 m 갭에서 GAST 2000이 뒷발을 덜 들고, 고속과 계단 외력에서 21068보다 안정적이었습니다. 배포 묶음은 `gast/gast_v21_2000_oracle`, 체크포인트 SHA `db500e96…`입니다.

| 지표 | 21068 | GAST 교사 2000 |
| --- | --- | --- |
| 0.65 m 갭 완주 | 9/9 | 17/18 |
| 뒷발 최대 높이 | 29–33 cm | 23–32 cm |
| vx 1.2 m/s 최악 몸통 하강 | −11 cm | −2 cm |
| 계단 오르기 당김 200 N | 넘어짐 | 버팀 |
| 계단 내리기 밀기 250 N | 넘어짐 | 버팀 |

## 5. 학생 3종: GAST 학생, RVLD, GAVD

세 학생은 입력과 증류 방식이 같고, 영상을 지형 latent 32로 바꾸는 구조만 다릅니다.

- **공통 입력:** 카메라 4대 × (Depth, IR proxy) × 45×80. 시뮬레이션 IR은 렌더 흑백 영상으로 만든 대용물입니다.
- **공통 증류:** 교사의 CENet·Actor는 고정하고, 학생이 교사 지형 인코더를 대신합니다. 배포할 때는 학생 latent를 교사 Actor에 꼽습니다.

| 학생 | 전체 이름 | 특징 | 네트워크 구성 | 현재 체크포인트 (교사 21068) |
| --- | --- | --- | --- | --- |
| GAST 학생 (GAST-S) | Gap-Aware Spatiotemporal Teacher-Student, Student | 로봇 자세·이동량으로 정렬한 격자 기억으로 지나간 지형을 유지. 영상 누락·지연 시 품질 gate로 지형 기여를 0으로 | GAVD 구조 + 영상 나이 embedding 32 + 시간 attention(32차원, 4 head) + 칸별 GRUCell 32 → 공간 head 6 · 품질 head 1 · 187×32 → 64 → latent 32 | 19840 (로컬). 갭 집중 파인튜닝 24960이 완주 3/3 |
| RVLD | Recurrent Visual Latent Distillation (순환 시각 latent 증류) | 카메라마다 작은 CNN으로 압축한 뒤 GRU로 기억. 가장 가볍지만 작은 경계가 뭉개질 수 있음 | 공유 Conv 2→16→32→32 → 480 → 32 (카메라당) → 4개 연결 128 → 64 → GRUCell 64 → latent 32. 보조 hazard head 64 → 16 → 1 | 19008 (90.111) |
| GAVD | Grid-Attention Visual Distillation (격자 attention 시각 증류) | 영상의 공간 특징을 유지하고, 지형 격자 칸이 영상에 직접 묻는 방식. 경계 지도와 행동을 함께 학습 | CNN 2→16→32 → 카메라당 144 token(총 576) + 카메라 ID·기하 8차원 → 187개 지형 query의 cross-attention(32차원) → FFN → 공간 head 5 → GRUCell 63 → latent 32 | 19008 (90.111에서 학습, 10/07 배포 묶음) |

- 세 학생의 공정한 비교를 위해, GAST 교사 v2.1이 끝나면 같은 교사에서 세 학생을 다시 증류합니다(512 env, 캡처 20,000회).
- 2026-10-06 기록에 "현재 증류 코드는 BIVT-Ray 교사만 지원"이라고 되어 있습니다. 2026-10-07에 GAST 교사 → 학생 증류 경로를 추가했습니다 (gd\_lab\_vrl 70e40fb). 첫 증류는 GAST 교사 2000으로 GAST 학생을 카메라 캡처 28,000회 학습하는 중이며, 22:22 기준 3,456회입니다.

## 6. 다음 단계와 출처

- GAST 학생(교사 GAST 2000) 증류가 끝나면 자동으로 MuJoCo 평가를 돌립니다. 이어서 21068 / GAST 교사 2000 / GAST 학생 비교표를 만듭니다.
- GAST 교사 5,000 업데이트도 같은 방식으로 자동 평가합니다. 현재 속도면 2026-10-08 09시 전후입니다.
- 온라인 Top-5는 883에서 멈췄습니다. 게이트가 커리큘럼 램프 전 기록과 비교했기 때문입니다. 이제 5,000부터 다시 기록하며, 사고 기록 INC-001과 PF-001에 남겼습니다.
- GAST 학생이 교사를 따라오면 RVLD·GAVD도 GAST 교사에서 증류합니다.

**출처 (로컬과 서버 파일)**

- 방법 비교 문서: `gd_lab_vrl/gast/docs/rbq10_learning_methods_20260930.md`
- GAST 코드: `gast/src/gd_lab/gast/temporal.py`, `teacher.py`, `warm_start.py`, `student.py`
- 체크포인트 묶음: 서버 `ray_top1_17206_20261003/`, `ray_gap_clean_vendor_new_top1_21068_20261005/` (README, metadata.json)
- 진행 기록: `claude_handoff/OVERNIGHT_REPORT_20261007.md`, `CLAUDE_TO_CODEX_STATUS_20261006.md`
