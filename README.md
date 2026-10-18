# gd_lab_vrl

RBQ10 안내견 로봇의 비전 보행 학습 저장소다. 블라인드 DreamWaQ 정책에서 출발해 지형을 보는 교사 정책을 PPO로 학습한다.
현행 구조는 **GAST Ray+IR 교사 직결**이다.
- 배포 경로: 로봇 하부 RealSense D4xx 카메라 4대(depth + IR) → 지형처리기 → 몸 주변 11×17 칸(10 cm) 지형 지도 → 교사 정책.
  - 지형처리기: camfilter136 이상치 제거 → v25or 매퍼. 두 이름은 아래 [용어](#용어)에 설명한다.
  - 지도의 칸마다 높이, valid(관측 여부 0/1), IR 밝기 값이 들어간다.
- 카메라 영상을 직접 받는 "학생" 정책은 쓰지 않는다. 교사가 지도를 직접 받는 구조라서 "교사 직결"이라 부른다.
- 관련 저장소: 블라인드 학습은 `gd_lab`, 배포(Pilot·MuJoCo 검증)는 `gd_rbq10_deploy_vrl`.

**전체 설명 문서**
- [docs/design/VISION_RL_OVERVIEW_20261018.pdf](docs/design/VISION_RL_OVERVIEW_20261018.pdf): 목적, 입출력, 지형처리기, Actor-Critic·PPO, E1·RL Replay·MuJoCo 도구, 배포, 블라인드 기초편.

## 현재 상태 (2026-10-18)

| 항목 | 내용 |
|---|---|
| 기준 교사 | **Ray+IR 3906** (Isaac 태스크 `Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0`). 3906은 학습 iteration 번호다. 블라인드 정책 DWB-38000에서 blind_init으로 시작했다. 지형 칸의 50%가 승강장 갭이고, 병렬 로봇 1024대(env)로 5000 iteration 학습했다 |
| 3906 E1 | E1(Isaac 평가, 아래 [E1](#e1-평가의-입력-경로-arm)) 결과다. 수치는 갭을 발 빠짐 없이 깨끗이 넘은 비율이다. v25or 지도 입력 갭 96.8%, 카메라+필터 지도 입력 갭 96.2%, 계단 100% |
| 배포 후보 | 원래 3906과 A2769_top1 중 최종 선택을 기다리는 중이다. A2769_top1은 Ray+IR 2769를 aug149(매퍼 오차 증강 파인튠, 브랜치 `aug149-mapper-aug`)로 이어 학습한 모델의 Top-1이다 |
| MuJoCo depth_ir 갭 (12 seed) | MuJoCo 배포 시뮬(실제 Pilot 코드)에서 depth+IR 카메라 → 지형처리기 경로로 갭 구간 12회를 시험해 완주한 횟수다. A2769_top1 11/12, 3906 10/12 |
| 진행 중 | pose(odom) 오차 평가 도구는 완료 (UPDATE157, [결과](https://github.com/GuideDog-ETRI/gd_rbq10_deploy_vrl/blob/main/docs/experiments/POSE_ERR_ODOM_20261018.md))<br>GAST-Nav v1 (다음 교사: 좁은 통로 통과, 시계방향 제자리 회전, 계단 정렬) 구현 착수<br>sim_docker 통합: 팀원 Isaac 시뮬레이터에 3906 실행기를 구현했고, in-sim smoke와 PR은 전이다<br>UPDATEnnn은 Codex 작업 지시 번호이며, 문서는 `claude_handoff/`에 있다 |
| 실기 | GAST 경로는 아직 시뮬레이션 전용이다. Pilot `/odom` 구독과 실기 odom 오차 측정은 포팅할 때 한다 ([실기 포팅 때 할 일](#실기-포팅-때-할-일-아직-구현-안-됨)) |

**계보** (숫자는 iteration 번호, 화살표는 이어 학습함)
- 3906: DWB-38000 → Ray+IR gap50 → 3906 → aug149 A3906
- 2769: `d_v3.6.21_b1_18` ONNX → BIVT-Ray 17206 → Clean 21068 → GAST v2.1 2000 → Ray+IR 2769 → aug149 A2769 / C2769
  - `d_v3.6.21_b1_18`은 로봇 기본 배포 블라인드 정책이다.
  - A = 증강 적용, C = 증강 없는 대조군이다. `_top1`은 그 학습의 online Top-1, `l2000`은 마지막(local +2000) 체크포인트다.
- 전체 계보(실패·중단 포함): [docs/lineage/records_20261016/LINEAGE_20261016.md](docs/lineage/records_20261016/LINEAGE_20261016.md), 그래프 `rbq10_lineage.html`

## 용어

### 정책과 학습

| 용어 | 설명 |
|---|---|
| 블라인드 정책 / DreamWaQ | 카메라 없이 관절·IMU 같은 고유감각만으로 걷는 정책. DreamWaQ는 CENet(속도·맥락 추정기)을 쓰는 학습 방식이다. 개념 설명은 PDF 기초편 A0–A3 |
| DWB / DWB-38000 | DreamWaQ Blind. DWB-38000은 우리가 처음부터 학습한 ARM4 블라인드 정책의 38000 iteration 모델이다 (sha c6cfe178) |
| ARM1–4, `TRAIN_ARM` | 블라인드 학습 비교 실험의 4가지 조건. 제어 주기 50/100 Hz × PD 게인 두 종류다. ARM4 = 100 Hz, Kp 123.39/127.77, Kd 2.4이며 현행 기준이다. `TRAIN_ARM=4`로 선택한다 |
| 교사 / 학생 | 교사는 시뮬레이터에서 지형 정보를 받아 학습한 정책이다. 학생은 카메라 영상만으로 교사를 흉내 내도록 증류한 정책이다. 현행은 교사만 쓴다 |
| blind_init / warm start | 블라인드 정책의 가중치를 복사하고, 지형 입력 경로만 새로 붙여 학습을 시작하는 방법. 시작할 때는 블라인드와 똑같이 걷는다 |
| BIVT-Ray | Blind-Initialized Visual Teacher, Raycast. 블라인드에서 시작해, 카메라에 보이는 칸만 입력받도록 학습한 초기 교사 |
| GAST | Gap-Aware Spatiotemporal Teacher. 최근 8장(최대 4.8 s)의 지형 지도를 현재 자세로 맞춰 겹쳐 보는 시간 기억 encoder를 가진 교사 |
| Ray+IR | 현행 GAST 교사. 학습 입력을 배포 지형처리기가 실제로 만들 수 있는 형태로 맞췄다. 레이캐스트로 카메라에 보이는 칸만 쓰고, 노이즈·결측(dropout)·센서 끊김(blackout)을 넣고, IR 밝기 채널을 더했다 |
| gap50 | 학습 지형의 50%를 승강장 갭(폭 2–26 cm)으로 둔 설정 |
| aug149 | Ray+IR 모델을 이어서 2000 iteration 더 학습하면서, 배포 매퍼의 오차를 actor 입력에 흉내 내 넣은 파인튠. pit을 넓게 보거나 칸이 비는 오차다. A3906, A2769는 증강을 적용했고 C2769는 대조군이다 |
| iteration / env | iteration은 모든 로봇이 100 step 데이터를 모은 뒤 PPO로 한 번 갱신하는 단위다. env는 동시에 학습하는 시뮬 로봇 수다 |
| online Top-5 / Top-1 | 학습 중 rollout 통계(갭 성공률 등)로 고른 상위 5개 체크포인트와 그중 1위. held-out 평가가 아니므로 최종 선택은 E1과 MuJoCo로 한다 |
| RVLD / GAVD / GAST-S | 카메라 학생 모델들. 보류 상태이고 코드는 그대로 둔다 |

### 센서와 지형처리기

| 용어 | 설명 |
|---|---|
| D4xx depth + IR | 로봇 하부의 Intel RealSense 카메라 4대. 거리 영상(depth)과 적외선 밝기 영상(IR)을 준다 |
| 11×17 지형 지도 | 몸 방향 기준 앞뒤 1.6 m × 좌우 1.0 m를 10 cm 칸 187개로 나눈 지도. 칸마다 높이, valid(관측됐으면 1), IR 값을 담는다 |
| 지형처리기 / 매퍼 | 카메라나 LiDAR 측정을 11×17 지도로 바꾸는 사람이 만든 처리 코드 (`tools/rl_replay/`). 학습한 신경망이 아니다 |
| camfilter136 | depth 영상의 튀는 점(spike)을 3×3 이웃 median 기준으로 지우는 필터. 지운 픽셀은 "모름"으로 둔다. 136은 작업 번호다 |
| v25or (L0S) | 현행 배포 매퍼, "v2.5-OR". 두 부분으로 이뤄진다. (1) h0d: 칸 높이를 중심과 주변 4점(±2.5 cm) 중 가장 낮은 값으로 정한다(deepest-of-5). (2) structure130: 카메라 레이가 바닥 아래로 빠지는 모양으로 갭을 찾는다. 2.5 cm 어긋난 두 격자 중 하나라도 갭이면 갭으로 본다(DUAL=2, OR). 갭 깊이 표시 값은 0.7 m(DDEEP)다 |
| oracle 지도 | 매퍼 대신 시뮬레이터의 정답 지형을 넣은 입력. 진단용이다 |

### E1 평가의 입력 경로 (arm)

E1은 Isaac에서 정책을 고정하고, 같은 seed와 같은 초기 상태에서 **지형 지도를 만드는 방법(arm)만 바꿔** 갭·계단을 시험하는 평가다. 성능 저하가 정책 탓인지 지도 탓인지 나눠 볼 수 있다.

| arm | 정책에 들어가는 지도 | 의미 |
|---|---|---|
| G | 정답(GT) 지형 그대로 | 정책의 상한 |
| GV | 정답 지형 중 센서가 최근 2 s 안에 본 칸만 | 가림·시야 제한만의 영향 |
| v25or | LiDAR 측정 → v25or 매퍼 | 매퍼 오차까지 포함 |
| 카메라 | depth 카메라 4대 측정 → v25or 매퍼 | 카메라 노이즈 포함 |
| 카메라+필터 | 카메라 → camfilter136 → v25or | **실제 배포 경로** |

결과의 "갭 X%"는 갭에 다가간 횟수 중, 발이 갭 안으로 떨어지지 않고 깨끗이 넘은 비율(clean / reached)이다.

### 배포·검증

| 용어 | 설명 |
|---|---|
| MuJoCo 배포 시뮬 | `gd_rbq10_deploy_vrl`의 실제 Pilot(C++) + ONNX를 MuJoCo 물리엔진에서 돌리는 검증. Isaac과 엔진이 달라서 최종 선택에 반드시 쓴다 |
| sensor sidecar / depth_ir | MuJoCo에서 카메라 depth+IR을 흉내 내 매퍼로 지도를 만들어 Pilot에 보내는 별도 프로세스. depth_ir은 그 카메라 모드다 |
| force4 | MuJoCo 갭+계단 코스. 계단 4지점에서 손잡이를 자동으로 당기거나 미는 외력 시험 |
| pose / odom | 로봇 위치·방향. 시뮬레이션은 정답 pose를 쓴다. 실기는 camel_pilot의 `/odom`(50 Hz)을 쓸 예정이다 |
| sim_docker | 팀원의 Isaac 시뮬레이터 저장소 (GuideDog-ETRI/sim_docker) |
| camera_profile `vendor_new` | 벤더 SDK의 현재 카메라 장착 위치·각도. 학습과 평가가 같은 profile이어야 한다 |

## 실기 포팅 때 할 일 (아직 구현 안 됨)

지금 GAST 경로는 **시뮬레이션 전용**이다. 학습, E1, MuJoCo, sim_docker 모두 시뮬레이터의 정답(GT) pose를 쓴다. 실기로 옮길 때 다음을 한다.

| 할 일 | 내용 |
|---|---|
| **Pilot에 `/odom` 구독 추가** | camel_pilot이 발행하는 `/odom`(nav_msgs/Odometry, odom→base_link, 50 Hz)을 받아 지형 기억 warp와 매퍼의 pose로 쓴다. 현재 MuJoCo의 가짜 odom 패킷(ODM1, UPDATE157) 자리를 바꾸면 된다. 모델 입력 형식(캡처마다 world x, y, yaw)은 이미 맞춰 두었다 |
| **실기 odom 오차 측정** | 실기 `/odom`과 실제 이동을 비교해 drift, 미끄러짐, 계단의 z·pitch 오차, 지연을 잰다. 그 값을 pose 오차 평가(MuJoCo `POSE_ERR`, E1 `tools/rl_replay/run_pose_error157.sh`)에 넣어 3906의 민감도를 확인하고, 필요하면 pose 노이즈 파인튠을 한다. 가정 레벨 결과(4.8 s 상대오차 약 3 cm까지 저하 없음, 약 8 cm부터 30 cm 갭 실패 증가): [POSE_ERR_ODOM_20261018.md](https://github.com/GuideDog-ETRI/gd_rbq10_deploy_vrl/blob/main/docs/experiments/POSE_ERR_ODOM_20261018.md) |
| 실제 D4xx 스트림 → 매퍼 | 시뮬레이션 sidecar 대신 실제 depth+IR을 camfilter136 → v25or로 연결한다 |
| Pilot `--sim` 제한 해제 조건 | Pilot은 지금 `--interface lo --sim`에서만 GAST 모델을 받는다. pose, 접지, 센서 시계, hold 동작을 확인한 뒤 해제한다 |
| history 초기값 통일 | 계약은 `zeros`, 실행기는 `RBQ_VRL_HISTORY_INIT=repeat_first`를 쓴다. 하나로 맞춘다 |

## 실행

| 용도 | 명령 |
|---|---|
| Ray+IR 교사 학습 (3906 방식) | `bash scripts/gast/train_rayir_gap50_1gpu.sh dwb smoke\|train [gpu]` |
| Ray+IR 교사 학습 (2769 방식) | `bash scripts/gast/train_rayir_from2000_1gpu.sh smoke\|train [gpu]` |
| 교사 진입점 | `scripts/gast/train_teacher.py` (`--warm_start_{dwb,gast,bivt}`, `--distributed --total_envs N`) |
| E1 평가 | `experiments/e1/run_trial128.sh`, 큐 `experiments/e1/e1_136.sh` ([experiments/e1/README.md](experiments/e1/README.md)) |
| RL Replay 뷰어 | `TRAIN_ARM=4 python3 tools/rl_replay/server.py --port 8767` ([tools/rl_replay/README.md](tools/rl_replay/README.md)) |
| 예전 실험 (BIVT, 학생) | `experiments/<방법>/` ([experiments/README.md](experiments/README.md)) |

**공통 규칙**
- `TRAIN_ARM=4`로 실행한다 (100 Hz, Kp 123.39/127.77, Kd 2.4).
- `GAST_MATH_SDPA=1`을 켠다. GAST encoder의 attention 계산을 math 방식으로 고정하는 설정이다. 기본 fused 방식은 학습 중 NaN(무한대 gradient)을 냈다.
- 평가할 때는 가중치 옆에 학습 설정 `params/env.yaml`이 있어야 한다. 평가 도구가 이 파일로 camera_profile이 `vendor_new`인지 검사한다.

## 구조

```
src/gd_lab/          모든 코드: gast(교사·학생·Ray+IR), rl(PPO·CENet·Top-5·DDP), mdp(지형·보상·명령), core(카메라 계약),
                     teachers(cvtt, bivt), students(rvld, gavd), deploy
scripts/gast/        GAST 학습·평가 진입점과 실행 셸
experiments/         실행 셸 (e1, gast, bivt, rvld, gavd)
tools/rl_replay/     지형처리기(매퍼), E1 러너, RL Replay 뷰어, provenance
configs/ tests/ assets/ containers/
docs/design/         설계 문서 (VISION_RL_OVERVIEW, GAST_RAYIR_TEACHER_DESIGN, GAST_NAVSKILL_DESIGN)
docs/lineage/        정책 계보
checkpoints/         교사·학생 묶음 메타데이터(README, params, SHA256, E1 결과)
logs/                학습 로그 (git 밖)
```

## 체크포인트와 가중치

- git에는 메타데이터(README, `params/`, SHA256SUMS, E1 `result.json`)만 둔다.
- 가중치는 두 개만 git에 둔다. 둘 다 `git add -f`로 강제 추가했다.

  | 가중치 | 경로 |
  |---|---|
  | DWB-38000 (블라인드 루트) | `checkpoints/teachers/blind/arm4_38000/teacher/model_38000.pt` |
  | Ray+IR 3906 | `checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/best_top5/3906_top1.pt` |

- 다른 `*.pt`(PyTorch 체크포인트)·`*.onnx`(배포용 그래프)는 `.gitignore`로 제외한다. 학습 서버(로컬 5090, 90.111, onvlm2)에만 있으며, SHA256SUMS.txt로 확인하고 직접 복사한다.
- mp4·webm 영상은 git에 넣지 않는다.

## 문서

- [비전 RL 구현 총정리 (PDF)](docs/design/VISION_RL_OVERVIEW_20261018.pdf)
- [Ray+IR 교사 설계 (3906 기준)](docs/design/GAST_RAYIR_TEACHER_DESIGN_20261018.md)
- [GAST-Nav v1 설계 (v0.3.1, 구현 착수)](docs/design/GAST_NAV_V1_DESIGN.md)
- [sim_docker GAST 교사 통합 현황](docs/design/SIMDOCKER_GAST_INTEGRATION_STATUS_20261018.md)
- [정책 계보](docs/lineage/records_20261016/LINEAGE_20261016.md)
- [E1 평가](experiments/e1/README.md), [RL Replay](tools/rl_replay/README.md)

## 설치

Apptainer 이미지(IsaacSim 5.1.0 + IsaacLab 2.3.1)와 호스트 venv를 기준으로 한다. 절차는 [docs/history/README_before_reorg_20261007.md](docs/history/README_before_reorg_20261007.md)의 Quick start를 따른다.

```bash
run() { OMNI_KIT_ACCEPT_EULA=YES apptainer exec --writable-tmpfs \
        ~/workspace/gd_lab_isaaclab.sif ~/workspace/venv/bin/python "$@"; }
PYTHONPATH=src run -m pytest tests -q   # 전체 단위 테스트 (GAST는 tests/gast/)
```

10/07 재편 이전의 README와 구조 문서는 `docs/history/`에 있다.
