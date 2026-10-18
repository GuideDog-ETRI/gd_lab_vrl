# GAST-Nav v1 기본 설계서 (v0.3.1, 2026-10-18)

모델 이름: **GAST-Nav v1** (작업명이던 "GAST-Nav"을 대체). 학습 코드는 `src/gd_lab/gast_nav/`, 실행기는 `scripts/gast_nav/`, 시험은 `tests/gast_nav/`, 설정은 `configs/gast_nav/`, 브랜치는 `feat/gast-nav-v1`이다.

좁은 공간 통과, 시계방향 제자리 회전, 계단 정렬을 학습하는 교사 정책의 기본 설계다. 백본은 Ray+IR 3906이다.

**상태**
- 설계 검토 단계다. 구현·학습은 하지 않는다.
- 검토 이력: v0.1 → UPDATE155, v0.2 → UPDATE156. 두 번 모두 Codex 두 세션이 "조건부"로 판정했다.
- v0.3은 UPDATE156 지적과 2026-10-18 사용자 결정을 반영했다. UPDATE158에서 두 Codex 모두 "조건부: 코드·CPU fixture 착수 가능, 계약 4개를 먼저 고정"으로 판정했다.
- **v0.3.1**은 그 계약 4개를 부록 C에 고정했다. 이 계약을 기준으로 구현(UPDATE159)을 시작한다. 학습은 하지 않는다.
- 현행 학습(3906) 구조는 `GAST_RAYIR_TEACHER_DESIGN_20261018.md`에 있다. 이 문서는 바뀌는 점만 적는다.

**v0.2 → v0.3 변경**

| 구분 | 변경 내용 |
|---|---|
| 사용자 결정 | 통과·회전 판정은 다리 포함 로봇 swept 폭 + margin 기준 |
| | 사용자 추적 입력 삭제 |
| | 장소별 위치는 조건 c로 지정 |
| | 후방 coverage는 가정으로 둠 |
| | 시계방향 제한은 TURN 모드만 |
| | 계단참은 네 발 접지 시 일시 정지. 1차는 AUTO release만, 입력 통로는 다중 소스 |
| | 역회전 수치는 학습 밖 설정값 |
| | 진급 기준과 실행 순서 확정 |
| UPDATE156 반영 | "감독기가 보장" 문구 삭제 |
| | TURN clamp와 mode handshake |
| | 계단참 혼합 감지, requested_go와 effective_go 분리 |
| | hold 후보와 전환 검증 |
| | 학습 중 감독기 개입 규칙 |
| | A0를 additive adapter + gradient mask로 |
| | 지지면 추정, free 판정, 근거리 clearance 생성법 |
| | 패킷·rollout 저장 계약 |
| | jerk 불일치 수정 |
| | kinematic 사람 모델 수정 |
| | 안전 horizon 단위 수정 |

---

## 0. 요약

| 항목 | 결정 |
|---|---|
| 백본 | Ray+IR **3906** (sha f3e7afe5…). aug149 5737과 혼동하지 않는다 |
| 권한 경계 | **gdm**: 목적지, 경로, 갈림길, 다음 계단, 회전 시작·종료, 조건 c, 하강 승인(선택)<br>**정책**: 주어진 단일 경로 안에서 장애물 회피, 중앙·오프셋 정렬, 계단 정렬, 불가 시 정지<br>**감독기 (RL 밖)**: 명령 중재, mode 전환, 계단참 FSM, 안전 감시, hold 전환 |
| 정책 형태 | 관절을 직접 출력하는 단일 정책. 비교 기준은 "감독기 + 기존 3906 보행" 2단 구조 |
| 기존 입력 | 고유감각 46(명령 3칸 포함) ×4 + CENet 230 + 하방 지형 [h, valid, IR] 187×8. 의미, 단위, 순서를 유지한다 |
| 새 입력 | ① 넓은 장애물 지도<br>② 근거리 clearance 지도<br>③ GAST-Nav 명령 (mode, go, 상한)<br>④ 계단참 phase<br>⑤ 조건 c |
| 학습 규모 | onvlm2 GPU 0·1·2 DDP, `--total_envs 4096` (rank당 1366/1365/1365), horizon 100 |
| iteration | 기대 **약 6k** (pilot + A 1k + B 3k + C 2k). 상한 15k (A ≤3k, B ≤7k, C ≤5k). 단계 gate를 통과해야 다음 단계로 간다 (7.6) |
| 시간 | 기대 약 2–2.8일 (학습 1.5–1.8일 + 평가 13–24 h, 부록 C4.5). 상한 15k면 학습만 약 3.6–4.5일 + 평가. pilot에서 다시 산정한다 |

**보장에 대한 원칙**
- 학습은 행동 경향만 만든다.
- 감독기는 명령을 제한하고, 위반이 감지되면 hold로 전환한다.
- 시계방향 준수, 비접촉, 정지 거리는 **시험 결과(0/N과 신뢰 상한)로만** 주장한다.

---

## 1. 요구 기능과 성공 기준

수치는 개발 gate용이며 사람 근접 운용의 승인 기준이 아니다. 유한 시험에서 사고가 0건이면 "0/N"과 단측 95% 상한을 함께 적는다(예: 0/12 → ≈22%).

### 1.1 좁은 통로 통과 (개찰구, 엘리베이터 문, 지하철 문, 좁은 복도)

| # | 요구 | 성공 기준 |
|---|---|---|
| P1 | 시각장애인이 오른쪽 뒤에서 하네스를 잡는다. 로봇·사람·승객 모두 비접촉이 필수다 | feasible 장면 안전 완주 ≥95%, 접촉 0/N |
| P2 | 통로 중앙에 정렬하고, 조건 `y_off`만큼 옆으로 비킨다 | y_off 추종 오차 < 5 cm |
| P3 | 통과할 수 없거나 판단할 수 없으면 결정 지점 전에 정지한다 | refusal recall ≥95%, false-stop ≤5% |

**통과 판정 (확정)**
- 통로 폭 ≥ `W_swept + 2·margin`이면 시도하고, 아니면 정지한다.
  - `W_swept`는 다리를 포함한 로봇의 걸음새별 swept 폭이며, 구현 전에 실측한다.
- 0.9 m 장애인용 개찰구를 포함해 모든 통로에 같은 규칙을 쓴다. 특정 폭을 기준으로 두지 않는다.
- 측정 불확실성은 margin과 별도로 감독기가 inflate한다.

### 1.2 좁은 공간 시계방향 제자리 회전 (지하철 안, 엘리베이터 안)

| # | 요구 | 성공 기준 |
|---|---|---|
| R1 | 열차 가감속(가정 범위, 7.5)을 견딘다 | 넘어짐 0/N |
| R2 | 의자·사람·기둥과 접촉하지 않는다 | 접촉 0/N |
| R3 | TURN 모드에서는 시계방향(ωz < 0)으로만 돈다 | 양의 yaw 누적 ≤ 허용치 (아래) |
| R4 | 회전 swept 원 + margin이 들어가지 않거나 판단할 수 없으면 회전하지 않는다 | refusal recall ≥95% |
| R5 | 제자리를 유지한다. 기준은 회전 시작 시의 차량 바닥 좌표 | 회전 중 **최대** 중심 이동 < 5 cm |
| R6 | 회전 종료는 gdm이 odom 각도로 정한다 | 종료 명령 → 정지 p95 < 0.5 s |

- R3의 두 수치는 학습에 넣지 않는다.
  - 허용치: 평가 설정, 제안 3°.
  - 실행 중 개입: 감독기 설정, 제안 2°.

### 1.3 계단 정렬 오르내리기 (단일 경로)

| # | 요구 | 성공 기준 |
|---|---|---|
| S1 | 상위는 go만 준다. 갈림길과 다음 계단은 gdm이 고르고, 학습은 단일 경로만 한다 | — |
| S2 | 계단 방향에 스스로 정렬한다 (NAV 모드, 양방향 yaw 보정) | yaw 오차 < 5°, 이탈 0/N |
| S3 | 장애물이 있으면 가용 폭의 중앙(+y_off)에 정렬한다 | 중심선 오차 < 5 cm, 접촉 0/N |
| S4 | 계단참에서 네 발이 접지하면 일시 정지한 뒤 하강한다 (3.5) | dwell 이행 100%, release 성공 ≥95% |

---

## 2. 정책 구조

```
기존 (3906에서 이관, 의미 유지)
  고유감각 46 ×4 step (vx, vy, ωz 3칸 포함) ─┬─────────────────────────────┐
  CENet 230 → z_t 19 ─────────────────────────┘                             │
  하방 지형 [h,v,IR]×187×8 → GAST encoder → latent 32 ──────────────────────┤ actor 기존 층 (W_old)
새로 추가 (additive adapter, 7.1)                                           │      +
  넓은 장애물 지도 (3.2a) → 장애물 CNN → 32 ─┐                              │ actor 새 열 (W_new, 0 초기화)
  근거리 clearance (3.2b) → 작은 CNN → 16 ───┼── concat ────────────────────┘      → MLP → 관절 12
  GAST-Nav 명령 (3.4) + phase (3.5) + 조건 c (4) ┘
critic: 위 전부 + 특권(GT SDF, 물리적 feasibility, 관측 가능성, 차량 가속도, 감독기 개입 상태)
감독기 (RL 밖): 명령 어댑터 → mode handshake → 계단참 FSM → 안전 감시 → [GAST-Nav | hold 제어기]
```

- 조건 c는 **concat으로 시작**한다. FiLM은 ablation으로 필요성을 확인한 뒤 별도 계약으로 추가한다. v0.2 그림의 FiLM은 뺐다.
- 새 encoder 내부는 정상 초기화한다. actor 쪽 새 연결(W_new)만 0으로 둔다.
- "장애물 feature 0"은 "장애물 없음"이 아니다. unknown은 지도 채널로 따로 표시한다.
- 하방 지형 하위 계약(17×11, 10 cm, 8프레임, 폭 561, GSC2)은 유지한다. 전체 배포 계약은 v2로 바뀐다 (6장).

---

## 3. 입력 설계

### 3.1 센서 독립의 의미

- 정책은 원시 센서를 보지 않는다. 처리기가 몸 중심의 공통 표현으로 바꾼다.
- "센서 독립"은 **같은 계약과 검증된 오차 범위 안에서 교체 가능**하다는 뜻이다. 범위, 정확도, 지연이 바뀌면 다시 검증한다.
- 학습에서는 센서 family를 무작위로 바꾼다: 장착 위치, FOV, 거리, 레이 밀도, 가림.

### 3.2 장애물 지도

**(a) 넓은 지도 (정책 입력)**

| 항목 | 설계 |
|---|---|
| 좌표·범위 | 몸 yaw 정렬, 몸 중심 원점, 셀 중심 기준. x −2.4…+2.4 m, y −2.1…+2.1 m, 10 cm → **49 × 43** |
| 높이대 | **추정 지지면 기준**. B1 0.03–0.30 m, B2 0.30–0.80 m, B3 0.80–2.20 m |
| 칸 값 | 높이대마다 2 bit: 0 = unknown, 1 = free, 2 = occupied, 3 = 예약(받으면 거절) |
| 칸 age | 높이대별 마지막 관측 이후 시간. 4 bit, 0.1 s 단위로 0–1.5 s. 15는 "≥1.5 s 또는 없음" |
| 시간 | 현재 + 과거 2 프레임 (10 Hz). 프레임마다 sample pose와 timestamp를 붙인다 |
| TTL | 칸 age가 0.5 s를 넘으면 unknown으로 처리한다. 처리기와 정책 쪽 history가 같은 규칙을 쓴다 |

**지지면 추정 (UPDATE156 C)**
- 칸별 지지면은 **센서가 관측한 바닥·디딤판 점에서만** 만든다.
  - 이미 관측된 바닥과 연결되어 있어야 한다.
  - 단차는 ≤0.25 m, 경사는 제한 범위 안이어야 한다.
  - 하방 지형(17×11)과 넓은 지도 쪽 관측을 함께 쓴다.
- 좌석 상판처럼 바닥과 연결되지 않은 수평면은 지지면이 아니라 B2 occupied다.
- 지지면을 모르는 칸은 모든 높이대를 unknown으로 둔다.
- GT support mesh는 actor 입력에 쓰지 않는다. critic과 oracle에만 쓴다.

**free 판정**
- 레이 첫 히트까지 지나간 구간만 free 증거다.
- 한 칸×높이대를 free로 판정하려면 그 체적을 지난 레이가 2개 이상이고, 칸 높이 범위의 50% 이상을 덮어야 한다(제안). 증거가 부족하면 unknown이다.
- 자기 몸 히트는 지우되, 그 뒤를 free로 만들지 않는다.
- 같은 레이에서 정적 메시 히트와 해석적 동적 물체 히트(사람 capsule, 문·날개 box) 중 **가장 가까운 것**을 쓴다.

**움직이는 물체**
- 짧은 history는 ego-motion만 정렬한다. 같은 age의 정적 물체와 동적 물체는 구별할 수 없다.
- occupied는 TTL까지 보수적으로 유지한다. warp 보간 때문에 occupied가 지워지거나 free가 넓어지지 않게 한다.

**(b) 근거리 clearance 지도 (정책 입력)**

| 항목 | 설계 |
|---|---|
| 범위 | x −0.9…+0.9 m, y −0.6…+0.6 m, 5 cm → **37 × 25** |
| 값 | **clipped unsigned clearance**: 가장 가까운 occupied 경계까지 거리 0–0.5 m, uint8 (1/255 × 0.5 m). 0은 occupied 위이거나 접촉 거리 |
| 채널 | ① 하단(B1+B2) clearance ② 상단(B3) clearance ③ unknown bit |
| 생성 | 10 cm 지도를 업샘플하지 **않는다**. 원시 점·레이로 5 cm 점유 격자를 따로 만든 뒤 거리 변환한다. 가장자리 효과를 피하려고 계산 영역은 ROI보다 사방으로 0.5 m 넓게 잡는다 |
| 한계 | 칸 중심 기준 대각선 오차가 약 3.5 cm라서, 이것만으로 5 cm margin을 보장하지 않는다. 감독기가 불확실성만큼 inflate한다 |

**(c) 안전 지도 (감독기 전용)**
- 후방 coverage 전체와 회전 swept 영역을 담는다.
- 센서로 얻은 기하에 불확실성과 unknown을 반영한다. 배포에서는 GT를 쓰지 않는다.

**센서 coverage 가정 (확정)**
- 전방: Ouster + Mid-360. 지면부터 천장까지 본다.
- 후방: **좌우 뒷다리 바로 위(뒷 hip, x ≈ −0.31 m)를 기준으로 그 뒤쪽 영역 전체**를 지도 범위 안에서 본다고 가정한다.
- 장착 위치는 확정하지 않는다. 학습에서 위치·각도를 무작위화하고 가림·잡음·결측을 넣는다.

### 3.3 핸들 사용자 상태 — 삭제

- 사용자 결정(2026-10-18)으로 사용자 추적은 입력에서 뺐다.
- 사람은 후방 장애물 지도에 occupied로 보이는 그대로 다룬다.
- 시뮬레이터에는 핸들 사용자를 물리 물체로 그대로 둔다 (5장).

### 3.4 명령: 기존 3칸 유지 + GAST-Nav 명령 (어댑터)

actor는 이미 `(vx, vy, ωz)`를 받는다. 이번 변경은 **명령 생성기와 분포**를 바꾸는 것이다.
- `V21VelocityCommand`의 계단 heading 강제와 `platform_gap_commands`의 heading 설정은 LEGACY 모드에서만 실행한다.

| 입력 | 값 | 의미 |
|---|---|---|
| 기존 3칸 | vx, vy, ωz | 어댑터가 모드별로 채운다<br>• LEGACY: 3906과 같다<br>• NAV: (effective_go·v_cap, 0, 0)<br>• TURN: (0, 0, effective_turn_rate)<br>단위·scale은 같다. 다만 NAV의 vx는 "상한을 목표속도 형식으로 넣은 것"이라 의미가 완전히 같지는 않다 |
| `active_mode` | one-hot {LEGACY, NAV, TURN} | 감독기가 handshake를 마친 모드 |
| `effective_go` | 0 / 1 | 감독기가 적용한 go. 요청 go가 1이어도 계단참·hold·unknown 대기 중에는 0일 수 있다 |
| `v_cap` | 0.3–1.0 m/s | **NAV1에만 둔다** (조건 c에서 뺐다) |
| `effective_turn_rate` | `clamp(requested, −min(turn_cap, ω_max_verified), 0)` | TURN 전용 |

**우선순위**: invalid·NaN·timeout > go=0 > 계단참 FSM·hold > mode 명령 > 속도·회전 상한.

**mode handshake**
- 순서: requested_mode → STOPPING → STABLE(안정 정지 확인) → active_mode.
- gdm은 active_mode, ack, hold_reason을 돌려받는다.
- 중복 메시지가 와도 TURN 시작점(기준 yaw)을 다시 잡지 않는다. manoeuvre ID로 구분한다.

**보상 masking**
- NAV: 기존 xy·yaw 추종 보상을 끄고, 경로 축 진행 보상을 쓴다.
- TURN: 기존 yaw 추종 대신 새 회전 보상을 쓴다. 앞뒤·좌우 drift와 속도 상한 관리는 유지한다.
- LEGACY: 기존 보상 그대로.

### 3.5 계단참 FSM (감독기, 학습·배포 공통 코드)

```
APPROACH ─(hint arm + 감지 확인)→ PRE_BRAKE ─(정지 envelope 지점)→ LANDING_ENTERED
LANDING_ENTERED ─(네 발 접지 + 안정 정지)→ DWELL ─(landing_release 수락)→ RELEASED → DESCEND
DESCEND ─(현재 계단참 이탈 확인 + 다음 구간 전이)→ APPROACH (re-arm)
```

**감지 (혼합)**
- gdm의 landing·next-segment hint로 FSM을 **arm**한다.
- 현장 진입과 안정 정지는 **공통 감지 함수**로 확인한다.
  - 근거: 하방 지형과 넓은 지도의 지지면.
  - 판정: 오르막 다음 평지, 그 앞 내리막 모서리.
  - 평지가 길어 내리막이 지도 밖이면 hint로 arm하고 "평지 지속"만 확인한다.
- hint가 없거나 감지와 맞지 않으면 자동 하강하지 않는다. hold하고 gdm에 알린다.
- 학습에서도 phase를 GT 라벨로 주지 않고, 배포와 같은 감지 함수 출력을 쓴다.

**사전 감속 (PRE_BRAKE)**
- 진입을 확인한 뒤 제동하면 늦을 수 있다.
- 실측 정지 envelope만큼 앞에서 감속을 시작한다.

**dwell**
- 시작 조건: 네 발 접지(실기 contact estimator, debounce·hysteresis)와 안정 정지(선속도 < 0.05 m/s, |ωz| < 0.1 rad/s).
- **연속 유지**를 요구한다. 미끄러짐, 발 이탈, 속도 초과가 생기면 timer를 0으로 되돌린다.
- go=0이면 timer를 멈춘다(pause, reset 아님). 다시 시작할 때 안정 상태를 재검증한다.
- 비상 정지는 dwell을 충족한 것으로 치지 않는다.

**requested_go와 effective_go**
- LANDING_ENTERED와 DWELL 동안 effective_go = 0이다. requested_go는 그대로 둔다.
- 그래야 release 조건이 스스로 막히지 않는다.

**release (확정)**
- 입력 통로: NAV1의 `landing_release` **하나**. 필드는 landing ID, seq, 소스 {AUTO, GDM, USER}, timestamp.
- 1차 구현은 **AUTO만**이다.
  - 조건: 안내 시간(1차 설정값 2.0 s) + 0.5 s 동안 안정 정지를 유지했고, requested_go = 1.
  - GDM 승인과 USER(음성 → ROS 토픽)는 나중에 이 통로에 붙인다. FSM은 고치지 않는다.
- 소스와 관계없는 공통 조건. 하나라도 어긋나면 무시하고 사유를 gdm에 돌려준다.
  - landing ID 일치
  - DWELL 안정 정지 중
  - 다음 하강 지지면이 관측됨
  - requested_go = 1
  - 활성 hold 없음
  - 명령이 신선함

**re-arm**
- 현재 계단참을 벗어난 것을 확인하고 다음 구간으로 넘어간 뒤에만 다시 arm한다. "첫 하강 단 통과" 즉시가 아니다.
- mode·route·odom reset, 센서 timeout, 새 landing ID가 오면 reset한다.

**stall masking**
- `stair_stall`은 감독기가 진행을 금지한 상태(PRE_BRAKE, LANDING_ENTERED, DWELL, hold, unknown 대기, 외부 stop)에서 모두 끈다.
- 정상 APPROACH에서는 유지한다.

**정책 입력**: phase one-hot, dwell 진행률(0–1).

**phase 이벤트**
- phase가 바뀔 때마다 gdm에 이벤트를 보낸다. 같은 이벤트는 다시 보내지 않는다.
- 1차는 로그와 Console에 표시한다.
- 나중에 TTS(ZMQ, TODO T1)로 안내한다. 예: "계단참입니다. 잠시 멈춥니다." / "내려갑니다."

---

## 4. 외부 조건 입력 c

| 조건 | 범위 | 의미 | 기본값 |
|---|---|---|---|
| `y_off` | −0.20…+0.20 m | **경로·계단 축 기준** 옆 오프셋 (+ = 진행 방향 왼쪽) | 0 |
| `margin` | 0.05–0.20 m | 한쪽 최소 여유. 절대 하한은 감독기가 강제한다 | 0.10 |
| `turn_cap` | 0.2–1.0 rad/s | 회전율 상한 | 0.6 |

**장소별 예시 (확정, gdm이 설정)**

| 장소 | 설정 |
|---|---|
| 계단 | `y_off` = 0 (중앙) |
| 출입문 | `y_off` = +0.05~+0.10 m (조금 왼쪽, 사람 어깨 공간) |
| 개찰구 | `y_off` = 0 (사람이 뒤로 서서 따라옴) |

- 정책에는 장소 이름을 넣지 않고 숫자 조건만 넣는다.
- **우선순위**: 충돌·관측 불확실성·하드웨어 한계 > margin > y_off·속도 추종.
  - 보상 가중치만으로는 이 순서를 보장할 수 없어서 감독기도 같은 순서를 적용한다.
  - 불가능한 y_off는 가능한 구간으로 clamp하고, 요청값과 적용값을 gdm에 돌려준다. 그래도 안 되면 정지한다.
- **학습**: 조건을 무작위로 준다. 30% 에피소드는 기본값이고, 일부 에피소드는 rate limit을 걸고 도중에 바꾼다.
- 누락·NaN·timeout은 runtime에서 기본값이나 정지로 처리한다.

---

## 5. 학습 장면 (모두 단일 경로)

| 장면 | 구성 (무작위화) | 기능 |
|---|---|---|
| 개찰구 | 폭 0.50–1.00 m(0.9 m 장애인용 포함), 길이 1.2–2.0 m, 날개 0.9–1.2 m. 날개는 동적 box | P1–P3 |
| 문 | 엘리베이터 0.7–1.1 m, 지하철 1.2–1.4 m, 문턱 0–3 cm, 승강장 갭 5–20 cm. 문은 동적 box | P1–P3, 갭 |
| 좁은 복도 | 0.6–1.5 m, 굽은 구간, 기둥, 짐 | P1–P2 |
| 지하철 차내 | 도면 기준(차체 2.65 m, 통로 1.2 m, 문 1.3 m, 긴 의자, 기둥) ±15%, 승객 0–4명 | R1–R6 |
| 엘리베이터 | 1.1–1.6 × 1.4–2.0 m, 탑승자 0–3명 | R2–R6 |
| 장애물 계단 | 폭 1.2–3.0 m, 장애물, 손잡이, 계단참 | S1–S4 |
| 기존 지형 | 3906 구성, LEGACY 모드 | 회귀 방지 |

- 폭 범위는 가능·불가 사례를 둘 다 만들기 위한 것이다. 판정은 1.1의 swept 폭 규칙으로 한다.

**사람 모델 (UPDATE156 수정)**
- 핸들 사용자: 오른쪽 뒤 연결점을 따라오는 추종자. 반응 지연 0.2–0.6 s, 속도·가속 제한, 멈칫함, 손 놓음.
- 다른 승객: 정지 또는 0–0.8 m/s 이동.
- 이동 방식은 **collision-aware 궤적**이다. 로봇과 닿을 것으로 예측되면 멈추거나 경로를 바꾼다.
  - kinematic pose로 로봇을 밀어 넣지 않는다. 힘 제한만으로는 관통을 막을 수 없기 때문이다.
  - 침투가 생기면 그 장면을 폐기하고 기록한다.
- 관측(capsule)과 물리 충돌체의 pose, 크기, 시간을 일치시킨다.

**지형 스캐너와 구조물**
- 하방 지형은 배포 매퍼처럼 구조물을 포함해서 본다.
- 다만 "모든 기하를 넣으면 v25or와 같다"고 가정하지 않는다. 단계 A 전에 검증 matrix(9장 V0)를 통과해야 한다.

---

## 6. 변경 범위 ① — 지형·센서 처리기

| # | 구성 요소 | 변경 |
|---|---|---|
| T1 | 하방 지형 (Ray+IR) | 하위 계약 유지. 구조물 포함 시 의미가 바뀌는지는 V0에서 검증 |
| T2 | 장애물 지도 (학습) | 레이 first-hit, 정적과 동적을 같은 레이에서 비교, 관측 지지면, free 증거 규칙, 칸 age. GPU 텐서 연산 |
| T3 | 장애물 지도 (배포) | 점군 → 몸 좌표, 자기 몸 제거, free-space ray evidence, 칸 age와 TTL. T2와 같은 규칙. sidecar(Python)로 시작해 C++로 옮긴다 |
| T4 | 근거리 clearance | 원시 점·레이 → 5 cm 점유 격자 → 거리 변환. 학습과 배포가 같은 코드를 쓴다 |
| T5 | RayCaster 제약 | Isaac `ray_caster.py`는 정적 메시 1개만 지원한다. 정적 구조물은 메시로 합치고, 동적 물체는 해석적 교차로 따로 계산한다. env 사이 오염을 막고 메시 수와 VRAM을 측정한다 |
| T6 | MuJoCo sidecar | 장애물·근거리 지도, 동적 물체, 관성 외력, 감독기, FSM을 학습과 같은 코드로 구현한다. GT·센서·필터 단계를 따로 기록하고 특권 정보 누출을 검사한다 |
| T7 | 장애물 지도 warp | 3상태 categorical 전용 warp: occupied 보존, free 비확장, 범위 밖은 unknown. 기존 `warp_memory`(bilinear, 11×17 고정)는 쓰지 않는다. Python·ONNX·C++ golden parity를 확인한다 |
| T8 | **배포 계약 v2** | 아래 |

**패킷**

| 패킷 | 내용 | 크기·전송 |
|---|---|---|
| GSC2 | 기존 지형 (변경 없음) | 기존 |
| OBS1 | 넓은 지도 1 프레임 (2 bit 상태 + 4 bit age, 49×43×3) + 근거리 (37×25 × uint8 ×2 + unknown bit)<br>헤더: magic, version, length, 지도 해상도·원점·band 메타, sample pose, sample time(CLOCK_MONOTONIC), seq, reset epoch, CRC32 | 약 4.7 kB + 2.0 kB. **로봇 내부 local IPC(loopback 또는 shared memory)로만** 보낸다. 물리 네트워크로 보낼 때는 MTU 이하로 분할하고 완결 재조립과 만료를 명세한다 |
| NAV1 (gdm → 감독기) | requested_mode, requested_go, v_cap, turn_rate, c, `landing_release`(ID, seq, 소스), manoeuvre ID, heartbeat, seq, timestamp | 필드 확정 후 산정 |
| NAV1-ACK (감독기 → gdm) | active_mode, effective_go, phase, dwell 진행률, hold_reason, 적용된 c, release 거절 사유, phase 이벤트 | 같음 |

**인코딩·시간·pose 규칙**
- bit 순서는 little-endian, x-fastest, band-major다. 예약 상태값 3을 받으면 패킷 전체를 거절한다.
- OBS1의 pose는 **그 지도 snapshot을 표현한 sample pose**다.
  - 수신 시 최신 pose로 대체하지 않는다.
  - 받는 쪽은 sample pose에서 현재 pose로 한 번만 변환한다.
- GSC2·OBS1·NAV1 사이 최대 skew를 정한다(제안 100 ms). 넘으면 unknown이나 정지로 처리한다.

**exporter·초기화**
- exporter(`gast_teacher_oracle.py`, `action_export.py`)에 새 ONNX 입력의 이름, 형상, 정규화, 버전 메타데이터를 추가한다.
- torch → ONNX → Pilot parity를 확인한다.
- history 초기화(`zeros` vs `repeat_first`): 기존 3906 reference의 실제 설정을 기록하고, v2에서 하나를 선택한 뒤 parity를 다시 확인한다. 기준 실행을 조용히 바꾸지 않는다.

**실기 경로**
- 지금은 시뮬레이션 전용이다(`--sim`).
- 실기 pose는 camel_pilot `/odom`이다. pose, 접지, 센서 시계, hold 통합 계약과 함께 포팅한다.

**지연 예산 (측정 전 목표)**

| 구간 | 예산 |
|---|---|
| 정책 tick | 10 ms (100 Hz) |
| 캡처 → 지도 통합 | p95 ≤ 100 ms, p99 ≤ 150 ms |
| 10 Hz 위상 | 최대 약 100 ms (별도 합산) |
| 새 encoder 추론 | ≤ 1 ms |
| 감독기 판단 → hold 전환 명령 | ≤ 1 tick |

- "전환 명령 ≤ 1 tick"이 실제 정지가 1 tick에 끝난다는 뜻은 아니다. 실제 정지 시간은 따로 측정한다.

---

## 7. 변경 범위 ② — PPO 학습

### 7.1 warm start = schema migration + additive adapter

1. **고정 reference**: 원본 3906을 SHA, normalizer, 명령·history 계약과 함께 고정 보관한다.
2. **migration**
   - key와 shape별로 검증 복사한다.
   - 허용 확장: actor 새 입력 열(W_new), critic 입력 확장, 새 encoder 2개, 새 입력 normalizer(새 통계). FiLM은 추가할 때 별도로 허용한다.
   - 변환 후 전체를 strict load한다.
3. **additive adapter**
   - actor 첫 층을 `W_old·x_old + W_new·x_new`로 나눈다. W_new는 0으로 초기화한 **별도 Parameter**다.
   - 같은 Parameter 텐서에 열을 붙이면, A0에서 기존 층을 동결할 때 새 열도 학습되지 않는다. 그래서 분리한다.
4. **optimizer**
   - PPO와 CENet optimizer를 새로 만든다. parameter group은 old와 new로 나눈다.
   - source iteration과 nav iteration을 따로 기록한다.
   - strict resume은 GAST-Nav 첫 checkpoint부터 한다.
5. **parity (학습 시작 조건)**
   - 기존 관측에서 기존 행동이 나온다. 어댑터가 값을 바꾸는 경우는 "같은 관측"이 아니므로 따로 표시한다.
   - 새 입력이나 조건을 바꿔도 처음에는 행동이 같다 (W_new = 0).
   - normalizer와 history가 보존된다.
   - PyTorch와 ONNX가 일치한다.
   - blackout, stale, reset 궤적이 일치한다.
6. **보호**
   - A0 (≤300 iter): W_old, CENet, 지형 encoder, 기존 normalizer를 동결한다. 학습 대상은 W_new, 새 encoder, 새 normalizer, critic이다. 새 경로의 gradient norm을 확인한다.
   - A0 이후: 기존 경로는 lr ×0.1로 둔다. LEGACY 장면에서만 고정 3906 행동에 KL anchor를 건다.
7. **mirror**
   - LEGACY 행에만 `mirror_gast`를 적용한다. mixed minibatch에서 LEGACY 행만 고른다.
   - 새 채널(장애물 지도, phase, c)을 가진 행은 mirror하지 않는다.
   - mirror된 행이 NAV/TURN으로 학습되지 않게 막는다.

### 7.2 변경 목록

| # | 항목 | 현재 (3906) | 변경 |
|---|---|---|---|
| L1 | 시작점 | 3906 | migration + additive adapter (7.1) |
| L2 | 규모 | 1 GPU, 1024 env | DDP 3 GPU, `--total_envs 4096`. rank 평균 gradient의 편향과 mode별 sample filtering에 따른 rank별 유효 sample 수를 기록한다 |
| L3 | 명령 | heading 100%, 정지 20% | mode별 생성기와 handshake (3.4, 7.3) |
| L4 | 관측 | 46×4 + 230 + 지형 | + 넓은 지도, 근거리 지도, GAST-Nav 명령, phase, c |
| L5 | 네트워크 | GAST encoder + MLP | + 장애물 CNN(→32), 근거리 CNN(→16), additive adapter |
| L6 | critic | 깨끗한 지형 | + GT SDF, 물리적 feasibility, 관측 가능성, 차량 가속도, 감독기 개입 상태 |
| L7 | 보상 | v2.1 + clean gap | + 7.4, mode·phase별 masking |
| L8 | 외란 | 계단 핸들 외력, 짐 0/6 kg | + 열차 관성 외력 (7.5). 같은 외력 buffer에 합산 |
| L9 | 커리큘럼 | family 레벨 평균 복원 | stage, 폭, 밀도, 가속, 조건, 노이즈 스케줄, RNG, baseline hash, gate 결과를 **전체 저장**. reset-resume과 bit-exact 재개를 구별해 기록 |
| L10 | 종료 | 넘어짐 등 | + 구조물·사람 접촉 (사람 접촉은 즉시 종료) |
| L11 | 저장 | float32 rollout | 7.2.1 |
| L12 | 모델 선택 | online Top-5 | 보조 지표(scene × mode × 조건별 표본). 최종 선택은 9장 |
| L13 | 감독기 개입 | 없음 | 7.2.2 |
| L14 | 넣지 않는 것 | — | mapper 오차 증강, fused SDPA, NAV/TURN mirror, 사용자 추적 입력, FiLM(1차) |

#### 7.2.1 rollout 저장 (UPDATE156 보완)

**저장 원칙**
- 장애물 지도와 근거리 지도는 **캡처 단위 packed(bit/uint8) 불변 객체**로 pool에 넣는다.
  - 키는 (env, reset epoch, capture seq)다.
  - rollout과 update가 끝날 때까지 참조 count로 pin한다.
- step마다 저장하는 것:
  - 그 시점에 **이미 도착해 있던** 캡처 ID와 history ID
  - current pose, 상대 warp, age
  - phase, 적용된 c와 mode, effective_go, reset 표시
- 나중에 도착한 캡처로 과거 step을 다시 만들지 않는다(미래 정보 누출 방지).
- 센서 노이즈를 update 때 다시 뽑지 않는다. 저장된 packed 관측을 복원하고, encoder만 현재 parameter로 다시 계산한다.
- old log_prob, value, action은 rollout 당시 값을 저장한다.
- update 사이에 latent cache를 재사용하지 않는다.
- minibatch 단위로 unpack → warp → encoder를 다시 계산한다. rollout 전체를 float32로 한 번에 풀지 않는다.
- dense와 packed 경로의 obs, action, logprob, gradient parity를 확인한다.

**저장량 (4096 env, 1 s rollout, 10 Hz)**
- 신규 10 캡처 + 이전 2 캡처 = 12 캡처 × 약 6.7 kB (넓은 지도 상태+age + 근거리) ≈ **330 MB**.
- 여기에 메타(pose, age, ID), critic GT SDF, 기존 지형·history storage가 따로 붙는다.
- 최종 값은 pilot에서 peak VRAM으로 측정한다.

#### 7.2.2 학습 중 감독기 개입 (UPDATE156 A)

학습 env에서도 배포와 같은 감독기를 돌린다. hold나 clamp가 정책 행동을 대신한 transition은 다음과 같이 다룬다.
- **기록**: 정책 제안 action과 log_prob, 실제 실행 controller ID와 action, override 사유.
- **actor loss**: **정책이 실제로 제어한 transition에만** 적용한다. hold 구간은 actor loss에서 뺀다.
- **value**: 개입도 환경 전이로 보고 계속 학습한다. 개입이 시작될 때 개입 페널티를 줘서 "아무렇게나 해도 hold가 살려 준다"를 학습하지 않게 한다.
- **보고**: 감독기 있음(shielded)과 없음(unshielded) 성능을 따로 낸다. 정책 자체의 refusal 능력과 감독기 덕분의 무충돌을 구분하기 위해서다.
- **DDP**: actor loss mask 때문에 rank별 유효 sample 수가 달라진다. 그 수를 기록하고, 필요하면 sample 수로 가중한다.

### 7.3 명령 분포 (에피소드 단위, 단계 B 기준)

| 작업 | 비율 | 명령 |
|---|---|---|
| 통로 통과 | 25% | NAV, go 전환 포함, v_cap·y_off 무작위 |
| 시계방향 회전 | 20% | TURN, turn_rate −0.2…−turn_cap. 다양한 자세·위상에서 시작과 종료 (handshake 경유) |
| 계단 정렬 | 20% | NAV, go만, FSM 작동, AUTO release |
| 정지·출발 | 10% | go 전환, mode 전환 (handshake 경유) |
| 기존 지형 | 25% | LEGACY, 3906 분포 |

- 실제 transition 비율과 task별 return·gradient 크기를 기록한다.
- TURN 커리큘럼 순서: 저속 → 중속, 정지 상태에서 진입 → 회전 중 정지 → blocked·unknown → 외란.

### 7.4 보상 추가

| 보상 | 내용 |
|---|---|
| 여유 | 몸·다리 swept 외곽이 GT SDF 기준 margin보다 가까우면 페널티. 접촉은 큰 페널티, 사람 접촉은 종료 |
| 중앙·오프셋 | 가용 폭 중심선 + y_off (경로 축) |
| 진행 | NAV, effective_go = 1, feasible이고 관측됐을 때 경로 축 진행 속도 추종 |
| 정지 | 아래 2축 표 |
| 회전 | TURN: turn_rate 추종. 반시계 회전에는 **임계값 없이** 양의 yaw rate와 누적량에 비례 페널티. 최대 중심 이동 페널티, swept 여유 |
| 계단 | 계단 축 yaw 오차, DWELL 정지 유지 (stall masking은 3.5) |
| 외란 | 기존 넘어짐·미끄러짐 항목 재사용 |
| 개입 | 감독기 개입 시작 페널티 (7.2.2) |

**feasibility 2축 (UPDATE156)**
- 축 1, 물리적 feasibility: GT 기하 기준으로 1.1·1.2의 swept 폭 규칙을 적용한다.
- 축 2, 관측 가능성: actor가 본 지도로 판단할 수 있는가.

| 물리 \ 관측 | 충분히 관측됨 | unknown (가림·시야 밖) |
|---|---|---|
| 가능 | 진행 보상. 불필요한 정지에 페널티 | 정지·대기·제한된 관측 행동 허용. 진행 보상 없음, 정지 페널티 없음 |
| 불가 | 결정 지점 전 정지에 보상, 진입에 페널티 | 정지에 보상, 진입에 페널티 |

- 로봇 발 주변의 자기 가림 unknown은 swept 영역과 겹칠 때만 unknown으로 친다.
- 판단 불가 대기(감독기 설정, 제안 3 s)가 끝나도 unknown이면 hold하고 gdm에 재계획을 요청한다. 3 s는 학습 밖 설정이다.

### 7.5 열차 관성 외력

**모델**
- 회전하지 않는 차량 좌표계를 쓴다. 모든 링크 COM에 `F_i = −m_i·a_train(t)`를 global frame으로 건다. m_i는 랜덤화 후 질량이고 payload를 포함한다.
- 핸들 외력과 같은 buffer에 합산한다. 매 physics step 적용하고 reset 때 지운다.
- 동적 사람·짐에도 같은 가속을 반영한다.
- 병진만 다룬다(차량 회전은 제외).

**IMU**
- 현재 관측은 gyro와 projected_gravity다. 가속 중의 gravity 추정 편향을 노이즈로 넣는다.
- accelerometer를 관측에 넣을 때만 specific force `Rᵀ(a_rel + a_train − g)`를 재현한다.

**가속 범위 (훈련 가정, 출처 미검증)**
- 진행 방향: 0.6–1.2 m/s², 드물게 1.8–3.0 m/s², 상승 시간 1–2 s.
- 좌우 정현파: 진폭 A와 주파수 f를 **함께 샘플**해서 jerk `2πfA ≤ 10 m/s³`를 지킨다.
  - 예: A = 1.5 m/s²이면 f ≤ 1.06 Hz, f = 2 Hz이면 A ≤ 0.8 m/s².
  - v0.2는 둘을 독립 샘플해서 jerk가 18.8 m/s³까지 나왔다. 이를 고쳤다.
- 진행 방향 프로파일에도 jerk ≤ 10 m/s³를 적용한다.

**검증·구분**
- moving-floor 모델과 비교하는 fixture를 둔다.
- force4와는 별개 시험이다.

### 7.6 단계, 진급 기준(확정), 예산

**실행 순서 (확정)**
- pilot → A → 평가 → B → 평가 → C.
- 단계마다 아래 진급 기준으로 진행 여부를 정한다.
- 평가는 250–500 iter마다 한다. 독립 seed batch 두 번 연속 통과해야 진급한다.

| 단계 | 최대 iter | 기대 iter | 진급 기준 | 중단·재설계 |
|---|---|---|---|---|
| P (pilot) | 300 | 100–300 | parity 통과. 3 rank의 peak VRAM, map ms, physics ms, rollout/update ms, step/s, p95 지연 측정 → 시간 재산정 | NaN, rank 불일치, 예산 초과 |
| A0 | 300 (A에 포함) | 300 | 새 경로 gradient 확인 | 새 branch가 학습되지 않음 |
| A | 3k | 약 1k | 넓은 장면 safe completion ≥95%, go/stop·TURN·handshake 동작, legacy 비열등 | legacy 하락, 2회 연속 개선 없음 |
| B | 7k | 약 3k | scene별 feasible 완주 ≥95%, false-stop ≤5%, 접촉 0/N, dwell 100%, release ≥95% | 특정 조건 악화, 영구 정지, label 모순, 개입률 증가 |
| C | 5k | 약 2k | held-out 센서·기하·외력·사람에서 같은 기준 + baseline 비열등, 조건 추종, age·dropout 회복, 정지 응답 | oracle에서만 성공, 실제 지연이 가정을 넘음 → 학습 대신 표현·처리기 수정 |

**공통 진급 기준 (확정, 제안값 채택)**
- feasible 안전 완주 ≥95%
- false-stop ≤5%
- 접촉 0/N
- 기존 갭·계단 E1(카메라+필터)과 MuJoCo depth_ir에서 고정 3906 대비 **2%p 넘게 떨어지지 않음** (같은 seed·노이즈·payload·명령)

**시간 추정**

| 근거 | 값 |
|---|---|
| 3906 실측 | 1024 env, 1 GPU(90.111 RTX 5090) **11.7 s/iter**. 5000 iter에 16.3 h |
| GAST v2.1 | 4096 env, 3 GPU(onvlm2) 약 16 s/iter (rank당 1366 env + DDP 동기화) |
| GAST-Nav 추가 비용 (가정) | 장애물 레이캐스트, 동적 물체, 근거리 거리 변환, 감독기, packed 저장·재계산으로 **+30–60%** → 약 21–26 s/iter |
| 기대 (약 6k) | 6,000 × 21–26 s ≈ **35–43 h (1.5–1.8일)**. 단계 평가(약 3회 × 3–4 h)를 더하면 **약 2–2.5일** |
| 상한 (15k) | 15,000 × 21–26 s ≈ **3.6–4.5일** + 평가 |

- 학습량 비교: 4096 × 100 × 6k = 24.6억 transition으로, 3906(1024 × 100 × 5000 = 5.1억)의 약 5배다.
- 3906은 1–2k iter에서 점수가 평탄해졌다. GAST-Nav은 작업이 여러 개라 단계당 1–3k를 기대한다.
- 실제 iter 시간은 pilot 측정값으로 바꾼다.

---

## 8. 배포 안전 구조 (RL 밖)

- Pilot은 정책의 관절 목표를 바로 적용한다(`RlWalker.cpp:285–308`).
- infer 실패 시 VisionHold/Damp fallback은 있지만, 새 감독기 요구를 충족하지 않는다.
- 기존 상태 기계와 새 감독기 사이의 우선순위와 소유권을 정한다. actuator writer는 하나만 둔다.

### 8.1 감독기

**예측 (단위 정정, UPDATE156 D)**
- 시간 horizon: `T_h = evidence age + 계산·구동 지연 + 실측 정지 시간`.
- 공간 여유: `T_h` 동안 swept volume을 적분한 영역 + 제동 중 drift + 측정 불확실성.
- 이 영역이 occupied나 위험 unknown에 닿을 것으로 예측되면 감속하고, 그래도 안 되면 hold로 전환한다.
- evidence age가 허용 한계를 넘으면 정해 둔 fallback(감속 → hold)을 실행한다.
- p95 지연은 최악 제동의 상한이 아니다. 최대값도 측정한다.

**TURN 모드**
- 명령은 3.4의 clamp를 쓴다.
- 실행 중 개입 기준(감독기 설정, 제안): 반시계 누적 2° **또는** 양의 yaw rate 0.1 rad/s가 0.3 s 지속(약 1.7°). 두 조건은 독립이며 OR로 묶는다.
- 누적 counter는 manoeuvre ID 단위다. mode를 왕복해도 지우지 않는다. 각도 unwrap과 deadband를 적용하고, 기준은 차량 바닥 yaw다.
- 이것은 사후 보호다. 개입 뒤에도 제동하는 동안 더 돌 수 있으므로 허용치(3°)보다 작은 값에서 개입한다.

**판단 불가 대기**
- 진행이나 회전에 필요한 영역이 unknown이면 정지하고 최대 3 s(설정값) 기다린다.
- 그래도 unknown이면 hold하고 gdm에 재계획을 요청한다.

**hold 제어기**
- 1차 후보: 원본 3906 + 영 명령.
  - **"검증된 hold"라고 부르지 않는다.** standing 20%는 학습 분포였을 뿐, 안전 정지가 실증된 것은 아니다.
  - 지형·CENet 경로를 공유하므로, encoder가 실패하면 독립 fallback이 아니다. 그 경우를 위해 기존 VisionHold/Damp 경로를 독립 fallback으로 남긴다.
- PD 서기 자세로 즉시 당기지 않는다. 계단 위아래 발이나 swing 중인 발을 끌 수 있다. 안정 접지 상태에서 제한된 전환의 마지막 단계로만 쓴다.
- **전환 검증**
  - 전환 종류: NAV→hold, hold→NAV, TURN→hold.
  - 조건 교차: 보행 위상, 속도, payload, 마찰, 경사, 계단 모서리, 갭 걸침, 차량 가속.
  - 측정: joint target 점프, dq, 토크와 토크 변화율, 접촉 충격, 미끄러짐, 최대 drift, 정지 거리·시간(p95/max), 역회전 overshoot.
- hold 정책의 history는 실제 최근 관측으로 유지한다. 전환할 때 zeros나 repeat_first로 인위적으로 reset하지 않는다.
- 새로 다가오는 사람이 정지한 로봇에 부딪히는 경우는 hold만으로 막을 수 없다. 운용 범위와 gdm 개입 상태를 명시한다.

### 8.2 출력

NAV1-ACK(6장)로 다음을 gdm에 돌려준다: 적용된 조건, hold 사유, phase 이벤트, release 거절 사유.

---

## 9. 검증 계획

| 시험 | 기준 |
|---|---|
| **V0 (단계 A 전)** 구조물 포함 하방 지형 | 아래 V0 상세 |
| **V1** hold 전환 fixture | 8.1 전환 검증 matrix |
| E1 갭·계단 (G, 카메라+필터) | 고정 3906 대비 −2%p 이내 |
| MuJoCo depth_ir 갭 | 같은 기준, 신뢰구간 함께 보고 |
| 통로 | feasible 완주 ≥95%, time-to-pass p50/p95, false-stop ≤5%, refusal precision/recall, 접촉 0/N |
| 회전 | 양의 yaw 누적 ≤ 허용치(평가 설정 3°), 접촉 0/N, 최대 중심 이동 < 5 cm, 종료 응답, 개입 빈도 |
| 계단 | yaw < 5°, dwell 100%, release ≥95%, 접촉 0/N, phase 이벤트 순서 |
| 조건 | y_off 추종 < 5 cm, 불가능한 y_off의 clamp 보고 |
| 정지 응답 | 위험 감지·go=0 → 정지까지 거리·시간 p95/max, 재출발 지연, stop/go 진동 |
| 감독기 유무 | shielded와 unshielded를 따로 보고 |
| 외력 force4 150 N × 2 s | 3906 비열등 |
| 비교 기준 | 감독기 + 3906 2단 구조 |
| **검증 세트와 최종 시험 분리** | C 단계 held-out은 validation이다. 최종 시험은 개발 중 쓰지 않은 별도 layout·seed pool로 하고, 모델 선택에는 쓰지 않는다 |

**V0 상세**
1. 고정 3906으로 같은 pose·scan에서 기존 지형 경로와 구조물 포함 경로를 비교한다: h, valid, IR, age, latent, action.
   - 대상: 벽(거리·높이·방향), 좌석 위·아래 가림, 천장, 얇은 기둥, 이동 사람.
2. 안전한 paired scene에서 갭·계단 성공, stall, 최소 여유를 잰다.
3. GT, camera-visible GT, v25or, MuJoCo depth_ir 단계로 나눠 비교한다.
4. 허용할 분포 변화 기준은 미리 정해 둔다.

**집계 분모**: scene별로 물리 가능·불가 × 관측됨·unknown × 폭·사람·가속·센서 age로 나눈다.

---

## 10. 사용자 결정 (2026-10-18, 모두 확정)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 권한 경계 | 갈림길과 다음 계단은 gdm이 정하고, 단일 경로만 학습한다 |
| 2 | 시계방향 | TURN 모드에만 적용한다. NAV의 yaw 보정은 양방향이다. 회전 종료는 gdm이 정한다 |
| 3 | 역회전 수치 | 학습에 넣지 않는다. 허용 3°는 평가 설정, 개입 2°는 감독기 설정이다. 학습은 임계값 없는 비례 페널티만 쓴다 |
| 4 | 계단참 | 네 발이 접지하면 일시 정지한다. release는 단일 통로에 AUTO·GDM·USER 소스다. 1차는 AUTO만 하고, TTS는 나중에 한다 |
| 5 | 통과·회전 판정 | 다리 포함 로봇 swept 폭 + margin이 들어가면 시도하고, 아니면 정지한다. 특정 폭 기준은 두지 않는다 |
| 6 | 사용자 추적 | 입력에 넣지 않는다 |
| 7 | 장소별 위치 | 조건 c(y_off 등)로 준다. 계단은 중앙, 출입문은 조금 왼쪽, 개찰구는 중앙(사람이 뒤에서 따라옴) |
| 8 | 후방 coverage | 좌우 뒷 hip 위 기준으로 뒤쪽 전체를 본다고 가정한다. 장착 위치는 미정 |
| 9 | 판단 불가 | 정지하고 최대 3 s 대기한 뒤 hold + gdm 재계획 요청 (설정값) |
| 10 | 치수·외란 | 기존 값을 쓰고, 판정은 5번 규칙으로 한다 |
| 11 | 진급 기준 | 제안값 채택 (7.6) |
| 12 | 실행 순서 | pilot → A → 평가 → B → 평가 → C, 단계마다 gate |

**구현 단계에서 확인할 것**
- 로봇 swept 폭 실측
- TTS ZMQ 주소와 메시지 형식
- 실기 `/odom` 오차

---

## 11. 위험 요소

| 위험 | 대책 |
|---|---|
| 기존 능력 망각 | additive adapter(W_new = 0), A0 동결, LEGACY에만 KL anchor, transition 비율 기록, 단계별 E1·MuJoCo |
| 지나친 보수화 | 2축 feasibility, false-stop 지표, 자기 가림 unknown 제외 |
| 투시 정보 누출 | first-hit 3상태, 관측 지지면만 사용, 보이지 않는 GT는 critic에만, 누출 검사 |
| 구조물이 하방 지형 의미를 바꿈 | V0 검증 matrix |
| hold가 정책을 대신해 학습을 왜곡 | 7.2.2 개입 규칙, unshielded 보고 |
| 정지 미보장 | 감독기 + hold 전환 검증(V1) + 독립 fallback |
| 메모리·시간 초과 | packed 저장, pilot 측정, 단계별 예산 |
| 시뮬레이터 간 차이 | 최종 선택에 MuJoCo 필수 |
| 실기 경로 부재 | 계약 v2, `/odom` 포팅 |

---

## 12. UPDATE156 지적 → v0.3 반영 위치

| 지적 | 반영 위치 |
|---|---|
| "감독기가 보장" 문구 | 0장 보장 원칙, 머리말 |
| turn_cap clamp, go·timeout 우선순위 | 3.4 |
| mode 전환 handshake, ack | 3.4, 6장 NAV1-ACK |
| v_cap 중복 | 3.4 (NAV1에만), 4장 |
| 3°와 1.7° trigger 불일치 | 8.1 (독립 OR, manoeuvre counter) |
| 계단참 hint + 감지 혼합 | 3.5 |
| requested_go / effective_go | 3.4, 3.5 |
| 계단참 진입 전 사전 감속 | 3.5 PRE_BRAKE |
| dwell 연속·reset·pause, stall masking 범위 | 3.5 |
| re-arm 조건 | 3.5 |
| hold 후보, PD 즉시 전환 금지, 전환 검증 | 8.1, 9장 V1 |
| stale encoder 시 독립 fallback | 8.1 |
| 감독기 개입과 PPO | 7.2.2 |
| A0에서 새 열이 학습되지 않음 | 7.1 additive adapter |
| FiLM·normalizer 허용 목록 | 7.1 |
| LEGACY-only mirror 구현 | 7.1 |
| 관측 지지면, GT 누출 | 3.2 지지면 추정 |
| free 판정 체적 기준 | 3.2 free 판정 |
| B3만으로 머리 여유 판단 불가 | 3.2(b) 상단 clearance 채널. 머리 여유를 보장한다고 표현하지 않음 |
| 근거리 SDF 업샘플 금지, unsigned 정의 | 3.2(b) |
| 칸 age, TTL 재현 | 3.2(a), 6장 OBS1 |
| sample pose 사용 | 6장 |
| categorical warp | 6장 T7 |
| MTU·fragmentation | 6장 OBS1 (local IPC) |
| enum, CRC, 버전 | 6장 |
| rollout 저장: 도착 캡처, pin, 불변 관측, 재계산, 메모리 | 7.2.1 |
| feasibility 물리·관측 2축 | 7.4 |
| 동적 ghost 표현 과장 | 3.2 움직이는 물체 |
| kinematic 사람 | 5장 사람 모델 |
| jerk 2πfA 불일치 | 7.5 |
| 안전 horizon 단위 혼합 | 8.1 |
| 구조물 포함 지형 검증 | 9장 V0 |
| 검증 세트와 최종 시험 분리 | 9장 |
| 사용자 추적 대체 시 기능 축소 | 3.3 삭제 (사용자 결정). 사람 추적에 의존하는 성공 기준 제거 |
| history 초기화 불일치 | 6장 |

---

## 부록 C. 구현 계약 (v0.3.1, UPDATE158 두 Codex 합의 반영)

본문과 이 부록이 다르면 **이 부록을 따른다**. 수치는 모두 설정값이며 학습 코드에 박아 넣지 않는다(`configs/gast_nav/`).

### C1. PPO·trainable 계약

**C1.1 controller 선택**
- v1에서는 **상태 기반**으로만 고른다.
- 감독기는 tick t의 상태와 명령만 보고, 정책 출력을 보기 **전에** controller {NAV_POLICY, HOLD_3906, DAMP}를 정한다.
- 정책 출력을 보고 위험한 action만 바꾸는 proposal 기반 shield는 v1에 넣지 않는다.

**C1.2 tick별 학습 규칙**

| 항목 | 규칙 |
|---|---|
| reward | tick t에 실제로 실행된 controller의 결과를 그 tick에 귀속한다. HOLD로 들어간 첫 tick에는 개입 페널티 `r_int`를 준다 |
| actor 학습 (surrogate, entropy, KL) | **NAV_POLICY tick에만** 쓴다. 이 mask는 rollout에 저장한다 |
| value | 모든 tick으로 학습한다. GAE도 전체 궤적으로 계산하며, HOLD tick의 보상도 return에 포함한다 |
| timeout bootstrap | 기존과 같다 |
| advantage 정규화 | actor에 쓰는 유효 행만으로 평균과 표준편차를 구한다 |
| 보조 손실 (지형 재구성) | 기존 규칙 그대로 둔다 |

**C1.3 DDP 분모**
- 각 rank는 `loss_r = world_size × Σ_local(valid loss) / N_global_valid`로 loss를 만든다. `N_global_valid`는 all_reduce로 구한다.
- 유효 행이 0인 rank도 같은 collective를 호출한다.
- 전역 유효 행이 0이면 모든 rank가 actor step을 함께 건너뛴다.
- entropy와 KL 통계도 같은 분모를 쓴다.

**C1.4 trainable manifest**

`configs/gast_nav/trainable_A0.yaml`

| 구분 | 대상 |
|---|---|
| 학습 | `W_new`, 장애물 CNN, 근거리 CNN, 새 입력 normalizer(통계 갱신 on), critic 전체 |
| 고정 | actor `W_old`, actor bias, 첫 층 이후 actor MLP, `log_std`, CENet(optimizer step off), 지형 encoder·재구성 decoder(aux step off), 기존 normalizer(통계 갱신 off) |
| 구현 방법 | `requires_grad=False` + optimizer parameter 목록에서 제외 + normalizer 갱신 플래그 off |
| 검증 | 고정 대상의 checksum을 매 iteration 확인한다 |

- A0가 끝난 뒤(`trainable_A.yaml`)에는 전부 학습한다. old group의 lr은 ×0.1이다.
- critic은 자기 특권 입력을 받는다. actor의 새 encoder를 **공유하지 않는다**.
- gradient fixture: step 1에서 W_new의 gradient ≠ 0이어야 한다. 새 encoder의 gradient는 W_new ≠ 0이 된 뒤에 ≠ 0이어야 한다. 첫 step에서 encoder gradient가 0인 것은 정상이다.

**C1.5 mirror**
- **LEGACY 행만** 모든 채널을 일관되게 반사한다.
  - 대상: 고유감각, 하방 지형, 장애물·근거리 지도(y 반전), y_off 부호, phase
  - LEGACY 장면에는 비대칭 과제가 없기 때문에 전부 반사해도 된다.
- NAV/TURN 행은 반사하지 않는다.
- 반사한 행의 mode는 원래 행의 mode를 그대로 유지한다.

### C2. 지도 golden 계약

**C2.1 지지면 추정**

`gast_nav/support.py`. 함수 인자로 gdm `segment_type`을 받는다. 힌트가 없으면 FLAT으로 본다.

- 씨앗: 네 발 접지 칸, 하방 지형에서 몸 주변 valid 칸.
- 경사 정의: 이웃 히트로 국소 평면을 적합한 뒤, 그 법선과 연직축 사이 각(= 수평면 대비 경사)을 쓴다. 35° 이하만 지지면 후보다.
- 영역 확장 범위는 구간 종류에 따라 다르다. 기하만으로는 계단과 좌석을 구분할 수 없어서 경로 힌트로 나눈다.
  - **FLAT / PASSAGE / TURN_SPOT**: 로봇 현재 지지면의 ±0.10 m 안만 지지면이다. 그보다 높은 수평면은 장애물로 처리해 해당 band를 occupied로 둔다. 좌석, 상자, 단상이 여기에 해당한다.
  - **STAIR_UP / STAIR_DOWN / LANDING**: 단차 0.25 m 이하이고, 위에서 관측된 수평 히트가 있는 이웃 칸으로 영역을 넓힌다.
- 지지면을 모르는 칸은 세 band 모두 unknown이다.
- 임계값(0.10 m, 0.25 m, 35°)은 configs에 둔다.

**반례 fixture**

| 구간 | 장면 | 기대 결과 |
|---|---|---|
| FLAT | 0.4 m 상자 | 장애물 |
| FLAT | 0.2 m 상자 | 장애물(B1) |
| FLAT | 0.2 m 단을 두 번 거쳐 닿는 0.45 m 좌석 | 장애물 |
| STAIR | 계단 오르막 | 디딤판 모두 지지면 |
| — | 승강장 갭 | 바닥 없음. 장애물 아님, unknown 또는 하방 지형의 깊음 |

**C2.2 지지면 전달**
- OBS1에 칸별 지지면 높이를 넣는다.
  - 형식: uint8, `z = −1.0 + 0.02·code` (code 0–254 → −1.0…+4.08 m, base 기준). 255는 unknown.
  - 크기: 49×43 = 2,107 B.
- 착지 감지와 안전 지도도 같은 값을 쓴다.
- history band는 지지면이 바뀌어도 다시 분류하지 않는다. 그 프레임의 sample 지지면을 함께 저장한다.

**C2.3 free 판정**
- band를 높이 10 cm 단위 sub-bin으로 나눈다(B1 3개, B2 5개, B3 14개).
- 그 칸의 **모든 sub-bin을 레이 1개 이상이 first-hit 전에 지나갔을 때만** free다.
- sub-bin 하나라도 안 지나갔으면 unknown이다(보수적).
- occupied는 히트가 있는 sub-bin이 하나라도 있으면 된다.

**C2.4 근거리 지도**
- 5 cm 점유 격자를 원시 점·레이로 따로 만든다.
- unknown 비트는 그 칸의 B1·B2·B3 중 **하나라도** unknown이면 1이다.
- clearance는 unknown 칸을 장애물로 보지 않는다. 대신 정책과 감독기 모두 unknown 비트를 따로 본다.
- age:
  - 근거리 지도도 칸별 4 bit age를 가진다. 크기는 37×25 = 925 B ÷ 2 ≈ 463 B다.
  - TTL 0.5 s가 지나면 unknown으로 바꾼다.

**C2.5 age 양자화와 packing**

| 항목 | 규칙 |
|---|---|
| age | 0.1 s 단위로 **올림**(보수적). 15 = ≥1.5 s 또는 관측 없음 |
| packing | little-endian, x-fastest, band-major. 2 bit 값은 byte 안에서 LSB부터 채운다 |
| 상태 코드 | 3은 예약값이며, 받으면 패킷 전체를 거절한다 |
| reset epoch | episode reset마다 +1. seq는 epoch마다 0부터 다시 센다 |
| 시간 | 모든 timestamp는 CLOCK_MONOTONIC ms |

- **golden 파일**: `tests/gast_nav/golden/obs1_*.json`. 입력(점군, pose, 지지면)과 기대 출력(packed bytes, decode 결과)을 담는다.
  - Python 구현이 통과해야 한다. 나중에 ONNX·C++ 구현도 같은 파일로 시험한다.
- 명세가 없는 정보를 GT로 채우지 않는다.

### C3. 명령·FSM 계약

**C3.1 route hint (gdm → 감독기, NAV1)**

| 필드 | 내용 |
|---|---|
| `segment_id` | 구간 번호 |
| `segment_type` | {FLAT, PASSAGE, STAIR_UP, LANDING, STAIR_DOWN, TURN_SPOT} |
| `route_axis_yaw` | odom frame에서의 경로 축 방향 |
| `dist_to_next_m` | 다음 구간까지 거리 |
| `next_segment_type` | 다음 구간 종류 |

- y_off는 이 경로 축을 기준으로 정한다.
- hint가 없으면 경로 축은 "현재 yaw로 고정"이고, landing 자동 하강은 하지 않는다.

**C3.2 AUTO release**
- **감독기 FSM이 직접** 만든다.
- 외부 입력과 같은 `landing_release` 내부 큐에 넣고, 소스 태그는 AUTO다.
- ACK에는 해당 seq와 accepted/rejected, 사유를 넣는다.

**C3.3 timer 2개**

| timer | 내용 |
|---|---|
| `dwell_timer` | 안정 정지가 이어진 시간 |
| `announce_timer` | 안내 시간(1차 2.0 s) |

| 상황 | dwell_timer | announce_timer |
|---|---|---|
| go=0 | 멈춤(pause) | 멈춤(pause) |
| 미끄러짐·발 이탈·속도 초과 | 0으로 reset | 유지 |
| hold·위험 판정 | 0으로 reset | 0으로 reset |

- release 조건: `dwell ≥ announce_end + 0.5 s` 그리고 공통 조건(3.5).

**C3.4 내리막이 안 보이는 계단참**
- 다음 하강 지지면이 관측되지 않으면 hold하고, 사유 `NO_DESCENT_VIEW`를 보낸다.
- AUTO release는 내지 않는다.
- 관측이 회복되면 다시 확인한다. gdm 승인이 와도 지지면 관측 조건은 그대로 적용한다.

**C3.5 기본값과 이상값 처리**

| 상황 | 처리 |
|---|---|
| 시작 시 미설정 | 기본값 사용 (y_off 0, margin 0.10, turn_cap 0.6) |
| 주행 중 NaN, 범위 초과, timeout | **hold** + 사유. 기본값으로 조용히 되돌리지 않는다 |
| margin을 낮추는 변경 | 정지 상태에서만 받는다 |

### C4. 감독기·평가 계약

**C4.1 health 상태 → controller (actuator writer는 RlWalker 하나)**

| health | 조건 | controller |
|---|---|---|
| OK | 모두 정상 | NAV_POLICY |
| TERRAIN_STALE | GSC2 또는 OBS1이 300 ms 넘게 갱신 안 됨 | 감속 → HOLD_3906 (지형 latent 0 = blackout으로 학습된 상태) |
| POSE_STALE | odom > `RBQ_GAST_POSE_MAX_AGE_MS` | 위와 같음 |
| SUPERVISOR_RISK | 8.1 예측 위험, TURN 역회전 개입 | HOLD_3906 |
| INFER_FAIL | ONNX 실패, NaN | sim: 기존 VisionHold / 실기: Damp |
| NAV1_INVALID | C3.5 | HOLD_3906 |

- 현재 UPDATE157 경로처럼 pose가 stale일 때 빈 scan으로 latent를 0으로 만들어도 "추론 성공"으로 보지 않는다. TERRAIN_STALE/POSE_STALE health로 따로 올린다.

**C4.2 horizon 시간축**
- 방식 (a): 지도 sample 시각부터 지금까지 odom으로 로봇 이동을 재구성하고, 그다음 `T_h = 계산·구동 지연 + 실측 정지 시간`만큼 앞으로 예측한다.
- 움직이는 장애물(occupied인데 사람 크기)은 `v_person_max(설정, 0.8 m/s) × (map age + T_h)`만큼 키운다.
- 제동 drift가 swept 예측에 이미 들어 있으면 따로 더하지 않는다. 남는 오차만 `residual_margin`(설정)으로 둔다.
- 허용 map age, 운용 속도 상한, envelope 밖에서의 fallback은 설정값으로 둔다.

**C4.3 실패와 시뮬 오류 집계**

| 분류 | 정의 | 처리 |
|---|---|---|
| 정책 실패 | 정책이나 감독기가 제어하는 중 로봇이 사람·구조물과 접촉 | **실패 분모에 남긴다** |
| invalid | 초기 겹침, NaN, 물리엔진 폭주, 정책과 무관한 사람 모델 침투 | 원인 코드와 함께 건수·분모를 따로 보고한다. invalid 비율이 1%를 넘으면 그 평가 batch는 무효다 |

- 사람 모델 다양성: collision-aware로 비키는 승객과, 멈칫하거나 지연되거나 통로를 막는 승객을 나눠 두고 각각 따로 집계한다.

**C4.4 수치 정정**
- jerk는 예시값이 아니라 `A ≤ 10/(2πf)` 식으로 clamp한다. 예: f = 2 Hz이면 A ≤ 0.7958.
- NAV 진행 보상은 `min(v_path, v_cap)/v_cap`이고, v_cap을 넘을 때만 페널티를 준다. 좁은 구간에서 천천히 가는 것은 벌하지 않는다.
- PRE_BRAKE 중 기존 vx 칸에는 감독기의 감속 profile 값을 넣는다.
- "자기 가림 unknown 제외" 문구를 다음으로 통일한다: "swept 영역 밖의 자기 가림 unknown만 무시한다. swept 안이면 위험 unknown이다."

**C4.5 평가 비용 (7.6 정정)**

| 평가 | 주기 | 내용 | 시간 |
|---|---|---|---|
| 중간 probe | 250–500 iter마다 | 장면별 소규모(env 64, 30 s, seed 2) | 약 20–30 분 |
| 단계 말 full | A·B·C 끝 | E1 + MuJoCo 전체 | 약 3–4 h |

- 6k 기준 평가 합계: probe 12–24회 × 0.3–0.5 h(≈4–12 h) + full 3회 × 3–4 h ≈ **13–24 h**.
- 학습(35–43 h)과 합치면 **약 2–2.8일**이고, pilot은 별도다.
- 기대 총량은 pilot 포함 6.1–6.3k, 상한은 15.3k다.
