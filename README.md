# gd_lab_vrl

RBQ10 안내견 로봇의 비전 보행 학습 저장소다. 블라인드 DreamWaQ 정책에서 출발해 지형을 보는 교사 정책을 PPO로 학습한다.
현행 구조는 **GAST Ray+IR 교사 직결**이다.
- 배포 경로: 4× D4xx depth+IR → 지형처리기(camfilter136 → v25or 매퍼) → 11×17 [높이, valid, IR] 지도 → 교사 정책. 카메라 학생은 쓰지 않는다.
- 관련 저장소: 블라인드 학습은 `gd_lab`, 배포(Pilot·MuJoCo 검증)는 `gd_rbq10_deploy_vrl`.

**전체 설명 문서**
- [docs/design/VISION_RL_OVERVIEW_20261018.pdf](docs/design/VISION_RL_OVERVIEW_20261018.pdf): 목적, 입출력, 지형처리기, Actor-Critic·PPO, E1·RL Replay·MuJoCo 도구, 배포, 블라인드 기초편.

## 현재 상태 (2026-10-18)

| 항목 | 내용 |
|---|---|
| 기준 교사 | **Ray+IR 3906** (`Gd-GastRayIrV21Gap50-Rbq10-Dreamwaq-v0`). DWB-38000(ARM4 블라인드)에서 blind_init, 갭 50% 지형, 1024 env × 5000 iter |
| 3906 E1 | v25or 갭 96.8%, 카메라+필터 96.2%, 계단 100% |
| 배포 후보 | 원래 3906, A2769_top1 (aug149 매퍼 오차 증강 파인튠, 브랜치 `aug149-mapper-aug`). 최종 선택 대기 |
| MuJoCo depth_ir 갭 (12 seed) | A2769_top1 11/12, 3906 10/12 |
| 진행 중 | pose(odom) 오차 평가 도구 (UPDATE157), NavSkill 설계 v0.2 검토 (좁은 통로·시계방향 회전·계단 정렬) |
| 실기 | GAST 경로는 아직 시뮬레이션 전용이다. 실기 pose는 camel_pilot `/odom`을 쓸 예정이고, 포팅할 때 구현한다 |

**계보**
- 3906: DWB-38000 → Ray+IR gap50 → 3906 → aug149 A3906
- 2769: `d_v3.6.21_b1_18` ONNX → BIVT-Ray 17206 → Clean 21068 → GAST v2.1 2000 → Ray+IR 2769 → aug149 A2769 / C2769
- 전체 계보(실패·중단 포함): [docs/lineage/records_20261016/LINEAGE_20261016.md](docs/lineage/records_20261016/LINEAGE_20261016.md), 그래프 `rbq10_lineage.html`

## 약어

| 약어 | 전체 이름 | 역할 |
|---|---|---|
| DWB | DreamWaQ Blind | 블라인드 DreamWaQ 정책 (DWB-38000 = ARM4 100 Hz, sha c6cfe178) |
| BIVT-Ray | Blind-Initialized Visual Teacher, Raycast Visibility | 블라인드 정책에서 시작한 카메라 가시 높이맵 교사 |
| GAST | Gap-Aware Spatiotemporal Teacher(-Student) | 8프레임 pose-warp 시간 기억 지형 encoder를 쓰는 교사 |
| Ray+IR | GAST Ray+IR teacher | 레이캐스트 가시 셀 + 노이즈·dropout·blackout + IR 채널로 학습한 GAST 교사 (현행) |
| v25or | L0S 매퍼 v2.5-OR | 배포 지형처리기 (h0d deepest-of-5 + structure130, DUAL=2, DDEEP=0.7) |
| E1 | Isaac 폐루프 평가 | 같은 초기 상태에서 지형 입력 경로(G/GV/v25or/카메라)만 바꿔 비교 |
| RVLD / GAVD / GAST-S | 카메라 학생들 | 보류 (코드는 유지) |

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
- `GAST_MATH_SDPA=1`을 켠다. fused SDPA는 NaN을 낸다.
- 평가할 때는 가중치 옆에 `params/env.yaml`이 있어야 한다. camera_profile `vendor_new`인지 검사한다.

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

- 다른 `*.pt`·`*.onnx`는 `.gitignore`로 제외한다. 학습 서버(로컬 5090, 90.111, onvlm2)에만 있으며, SHA256SUMS.txt로 확인하고 직접 복사한다.
- mp4·webm 영상은 git에 넣지 않는다.

## 문서

- [비전 RL 구현 총정리 (PDF)](docs/design/VISION_RL_OVERVIEW_20261018.pdf)
- [Ray+IR 교사 설계 (3906 기준)](docs/design/GAST_RAYIR_TEACHER_DESIGN_20261018.md)
- [NavSkill 설계 v0.2 (검토 중)](docs/design/GAST_NAVSKILL_DESIGN_20261018.md)
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
