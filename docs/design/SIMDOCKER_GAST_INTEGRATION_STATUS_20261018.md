# sim_docker GAST 교사 통합 현황 (2026-10-18)

팀원의 Isaac 시뮬레이터 `sim_docker`(GuideDog-ETRI/sim_docker)에서 학습한 GAST Ray+IR 교사를 돌리는 작업의 현황이다. **대상 모델은 원래 3906 하나로 정했다** (사용자 결정, 2026-10-18; A2769_top1은 넣지 않는다).
- 원 작업: UPDATE151 (5090 Codex, 2026-10-17)
- 보고서: `claude_handoff/CODEX5090_TO_CLAUDE_UPDATE151_20261017.md`
- 산출물: `archive/codex_worktrees/update151_simdocker/`

## 요약

| 문제 정의 | 결과 | 반영 방향 |
|---|---|---|
| 실행기 코드 | 브랜치 `feat/gast-teacher-runtime`의 worktree `~/gd_project/archive/sim_docker_worktrees/gast_runtime`에 구현했다. **커밋·푸시·PR은 하지 않았고, 원격 브랜치도 없다.** 기준은 sim_docker main 24680ac이며, 그 뒤 main이 바뀌지 않아 그대로 올릴 수 있다 | in-sim smoke 후 커밋 → PR (사용자 확인 후) |
| 계약 | deploy 쪽 `tools/write_simdocker_contract.py`가 `simdocker_contract.json`(schema `simdocker.gast.oracle.v1`)을 만든다. 실행기는 ONNX SHA를 검사한다 | 완료 |
| offline parity | 같은 입력을 넣었을 때 terrain frame·direct_obs·cenet_obs 차이 0, memory 입력 차이 ≤3.7e-9, action 차이 ≤9.5e-7 | 완료. 이것은 계약 parity이며, 기록된 Pilot 주행과의 비교는 아니다 |
| 지원 묶음 | **oracle(374, PhysX 정답 높이 스캔)만** 지원한다. 3906 oracle 묶음 `gast_rayir_3906_gap50_dwb38000_oracle_20261016`는 git에 있다 | 3906은 바로 시험할 수 있다 |
| in-sim 동작 | **확인하지 않았다** | Isaac Sim GPU로 smoke (아래 체크리스트) |
| 동역학 차이 | sim_docker의 모터 envelope(포화 200 N·m, 속도 한계 50 rad/s)와 2 physics step(10 ms) 행동 지연이 남아 있다. PD(Kp 123.39/127.77, Kd 2.4)와 토크 상한(93.6/126 N·m)은 적용했지만, 학습 액추에이터 곡선(saturation 104/140)과 같지는 않다 | 보행 결과를 해석할 때 주의 |
| 노이즈·payload | runtime이 노이즈를 새로 넣지는 않는다. payload 관측은 `GUIDEDOG_PAYLOAD_KG`로 준다. 실제 scene의 질량과 맞춰야 한다 | 실행할 때 설정 |
| pose | GT pose를 쓴다 (2026-10-16 결정과 같음) | 실기 `/odom`은 포팅할 때 |
| Pilot 주행과 tick 단위 비교 | telemetry dump(actor tick별 direct_obs, cenet_obs, terrain inputs, poses, ages, actions, reset)가 없어서 하지 못했다 | 선택 사항 |

## 변경 파일 (worktree, 커밋 전)

| 파일 | 내용 |
|---|---|
| `workspace/scripts/gast_runtime.py` | `GastTeacherRuntime`, `GastTerrainMemory` |
| `workspace/scripts/run_rbq10_sim.py` | 묶음 선택, 100 Hz decimation, GT scan 입력, 관절 mapping, PD와 limit 적용 |
| `workspace/scripts/check_gast_runtime.py` | Isaac 없이 도는 CPU 묶음 smoke |
| `RBQ10_SIM.md` | 실행, 계약, 제한에 대한 한국어 문서 |

## in-sim smoke 실행법 (미실시)

```bash
# 1) deploy 저장소: 계약 생성 (이미 생성된 묶음은 생략)
python3 tools/write_simdocker_contract.py resources/policy/gast/gast_rayir_3906_gap50_dwb38000_oracle_20261016

# 2) 묶음을 branch worktree의 workspace/RBQ/policy/에 두거나 컨테이너에 read-only mount
#    (호스트 묶음 경로는 컨테이너에 자동으로 mount되지 않는다. /isaac-sim/workspace가 branch worktree인지 확인)

# 3) branch sim_docker worktree에서 실행 (컨테이너 내부 경로)
GUIDEDOG_POLICY_BUNDLE=/isaac-sim/workspace/RBQ/policy/gast_rayir_3906_gap50_dwb38000_oracle_20261016 \
GUIDEDOG_PAYLOAD_KG=6 ./run_rbq10_screendoor_gap_sim.sh
```

**확인 항목**
- 두 그래프 로딩과 100 Hz 동작
- 관절별 순서, target, gain, effort
- 평지에서 높이와 valid 값
- history 연속성과 reset
- invalid terrain을 넣었을 때 latent가 0이 되는지
- NaN이 없는지
- scan과 추론이 10 ms 안에 끝나는지
- 기존 heavyfix/camel 경로가 그대로 동작하는지

## 다음 순서 (제안)

1. 3906 oracle 묶음으로 in-sim smoke를 한다. 로컬 GPU를 쓰는 pose 오차 실험(UPDATE157)이 끝난 뒤에 한다.
2. 결과가 괜찮으면 `feat/gast-teacher-runtime`을 커밋하고 PR을 올린다. 사용자 확인 후 진행한다.

관련 문서: `VISION_RL_OVERVIEW_20261018.pdf` 6.2절(번들과 계약), 6.4절(한계).
