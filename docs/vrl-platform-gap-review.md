# VRL platform_gap 구현 및 텐서 검토 — 2026-09-21

이 문서는 당시의 기록이다. 최신 카메라 관측 계약과 실제 실행 검증 결과는
[2026-09-22 검증 문서](vrl-validation-2026-09-22.md)를 우선한다.

## 범위

수정 대상은 `gd_lab_vrl`이다. 형제 디렉터리 `gd_lab`의 파일·실행 프로세스는 변경하지 않았다.
실제 학습, GPU/Isaac Sim 실행, 학습 대기열 등록은 하지 않았다.
추가로 활성화한 지형은 **platform_gap 하나**다. 기존 계단·거친 지면·경사면·ring_fence는 유지한다.

## 지형과 명령

`tasks/vrl_teacher.py`가 camera-free teacher와 `tasks/vrl_rough.py`의 camera student에 공통 적용된다.
VRL teacher의 등록 경로도 새 공통 설정으로 변경했다. Blind task ID의 설정은 그대로다.

| 항목 | 설정 |
|---|---|
| 지형 분포 | 15개 열: 기존 10종 각각 1열, platform_gap 5열(33.3%) |
| 기본 타일 | 8 × 8 m |
| 갭 위치 | 스폰 중심에서 world X 방향 ±1.5 m |
| 폭 | 난이도에 따라 0.02 → 0.24 m |
| 반대편 높이차 | 절댓값 0 → 0.16 m, 두 방향에 독립적인 상/하 부호 |
| 피트 바닥 | 양쪽 발판 중 낮은 쪽보다 0.65 m 아래 |
| 스폰 | 중앙 발판 높이 0, XY ±0.3 m, yaw 0 또는 π 근처 |
| 갭 명령 | 전진 0.20 m/s 이상, 상한은 일반 명령 커리큘럼에 따라 0.30–0.60 m/s |
| 방향 | world ±X, 횡방향 명령 0, heading 제어 유지 |
| 정지 | 기존 20% standing 명령 유지; 갭에서 추가 pulse 변조는 해제 |

기존 미사용 platform_gap 구현은 세 갭 중 중앙 갭과 스폰 원점이 겹치고,
랜덤 발판 높이와 반환 스폰 높이가 불일치할 수 있었다. 안전한 중앙 발판과 양쪽 두 갭으로 변경했다.
NumPy의 seeded RNG를 사용하여 같은 seed의 지형 재현성을 보존한다.

PLAY는 난이도 0.4로 고정하고 승급을 끈다. 지형 생성 자체는 열별 family 배치를 유지하여
지형별 보상 마스크가 실제 지형과 어긋나지 않게 한다. 동역학 randomization은 기존 PLAY 설정을 따른다.

## 보상 및 커리큘럼

기존 속도·yaw 추종, 접촉·미끄럼·착지·관절·행동 regularization을 유지한다.

| 항목 | 설계 |
|---|---|
| 발 빠짐 | 발판 지지 높이보다 22 cm 넘게 내려간 발에 비용. 추가 25 cm에서 포화, 네 발 평균, 가중치 −2.0 |
| 통과 성공 | 네 발 모두 갭의 바깥쪽 경계 + 6 cm를 넘고, 최소 두 발이 접촉하며, 몸이 서 있고, 발이 피트 안에 없을 때 0.1초 확인 후 +0.5 |
| 반복 방지 | 좌/우 각각 에피소드당 한 번만 지급; reset된 환경의 상태만 초기화 |
| 우회 방지 | 타일 Y 경계를 벗어난 에피소드는 성공 보상 및 승급 대상에서 제외 |
| 생존 조건 | 종료 실패 상태에서는 성공 보상 지급 안 함; 성공 후 나중에 넘어져도 승급 안 함 |
| 몸통 높이 | 갭 바닥을 뺀 발판 ray 평균을 기준으로 유지. 피트를 포함한 평균 때문에 몸을 낮추는 유인 제거 |
| pitch | 갭을 경사면으로 해석하지 않고 수평 자세를 기준으로 함. 기존 roll 제약 유지 |
| 수직 속도 비용 | 갭 family에서 기존 비용의 0.25배 |
| 착지 속도 비용 | 갭 family에서 기존 비용의 0.5배 |
| 정지 관절 자세 비용 | 높이가 다른 두 발판 위 자세와 충돌하지 않도록 갭에서는 제외; 정지 접촉 비용은 유지 |

IsaacLab RewardManager는 가중 보상에 `dt`를 곱한다. 연속 발 빠짐 비용은 이를 그대로 따르고,
통과 이벤트 함수는 `event / dt`를 반환하여 최종 보상이 제어 주기와 무관하게 +0.5가 되도록 했다.
두 신규 보상은 platform_gap에서만 켜진다.

갭 승급은 명령 이동 시간이 2초 이상이고 실제 통과 성공 이력이 있으며 실패 종료가 없을 때만 한다.
이동 기회가 있었는데 실패했거나, 충분한 명령 거리(2.5 m)가 있었는데 통과하지 못하면 강등한다.
기존 지형은 기존 명령 거리 기반 커리큘럼을 그대로 사용한다.
성공/발 빠짐은 정책 관측에 추가하지 않는다. privileged 정보는 보상·critic·teacher supervision에만 사용한다.

## 입출력 계약

아래 `B`는 배치 크기이며 기본 action 수는 12다.

| 경로 | 텐서 |
|---|---|
| 현재 proprioception | `[B,46]`: 각속도3 + 중력3 + 명령3 + 관절각12 + 관절속도12 + 이전 action12 + payload1 |
| Policy 관측 | `[B,230]` = 46 × 5; term-major, 각 term 내부 oldest → newest |
| Actor 직접 입력 | 최신 4프레임, newest first → `[B,184]` |
| CENet code | 추정 속도3 + 잠재값16 → `[B,19]` |
| Teacher terrain latent | `[B,32]` |
| Actor 결합 입력 | `[B,235]` = 184 + 19 + 32 |
| Critic 관측 | `[B,298]` |
| Teacher 높이맵 | critic의 `[110:297]`, `[B,187]` → `[B,1,11,17]` |
| Student 영상 | `[B,4,2,45,80]`, camera 순서 front0/front1/hind2/hind3, channel 순서 depth/IR proxy |
| Student recurrence | hidden 입출력 `[B,64]`, terrain latent 출력 `[B,32]` |
| Action | `[B,12]`; 기존 joint-position action scale 0.25 및 관절 한계 처리 유지 |

Critic 상세 slice(끝 인덱스 미포함):

| 항목 | slice |
|---|---|
| proprioception(payload 제외) | 0:45 |
| base linear velocity(CENet 감독, 관측 scale 2.0) | 45:48 |
| 발 주변 높이 | 48:64 |
| 발 접촉 | 64:68 |
| 발 접촉력 | 68:80 |
| 마찰 | 80:82 |
| 몸통 질량 offset | 82:83 |
| actuator gain | 83:107 |
| push Δv | 107:110 |
| height scan | 110:297 |
| payload | 297:298 |

## 계층별 구조

| 모듈 | 순서와 출력 크기 |
|---|---|
| CENet encoder | 230 → Linear128 → ELU → Linear64 |
| CENet heads | 64 → velocity3 / μ16 / logvar16 |
| CENet decoder | 19 → 64 → 128 → 46, hidden ELU |
| Teacher terrain CNN | 1×11×17 → Conv(8,3×3,p1) → ReLU → Conv(16,3×3,p1) → ReLU |
| Teacher pooling/head | 16×11×17 → adaptive pool16×5×8 → flatten640 → 64 → ReLU → 32 → tanh |
| Actor | 235 → 512 → 256 → 128 → 12, hidden ELU |
| Critic | 298 → 512 → 256 → 128 → 1, hidden ELU |
| Student per-camera CNN | `(4B,2,45,80)` → Conv16×23×40 → Conv32×12×20 → Conv32×6×10, 각각 ReLU |
| Student pooling/projection | 32×6×10 → pool32×3×5 → flatten480 → Linear32 |
| Student camera fusion | 4×32 = 128 → Linear64 → ReLU |
| Student memory | GRUCell(input64, hidden64) → hidden64 |
| Student latent head | 64 → 32 → tanh |
| Student auxiliary head | 64 → 16 → ReLU → 1; 학습에서만 사용하며 ONNX 출력에는 포함하지 않음 |

Student CNN stride는 모두 2, kernel은 첫 계층 5·이후 3, padding은 첫 계층 2·이후 1이다.

## 학습 경로와 수정점

1. Teacher PPO는 actor・critic・terrain CNN을 업데이트한다. CENet code는 detach한다.
2. CENet은 PPO 미니배치 처리가 끝난 뒤 별도 optimizer로 속도 감독 + 다음 관측 복원 + KL을 학습한다.
   종료 직후 관측은 다음 에피소드이므로 reconstruction 타깃에서 제외한다.
3. VRL actor의 CENet z는 rollout과 PPO 재평가에서 모두 μ를 사용하도록 수정했다.
   이전에는 같은 관측에도 z를 다시 샘플링하여 업데이트 전부터 action mean/log-prob가 달라질 수 있었다.
   VAE 보조 학습의 샘플링은 유지한다. AdaBoot 속도 대입은 기존 iteration 단위 규칙을 유지한다.
4. Student는 teacher를 freeze한 rollout에서 latent MSE와 보조 높이 불연속 MSE를 학습한다.
   BPTT 기본 8 camera ticks, window 내부 hidden 그래프 유지, 종료 row는 0, window 끝에서 detach한다.
5. 보조 높이 불연속 타깃은 height_scan 관측 scale(기본 5)을 나누어 미터 단위로 고쳤다.
   이는 거리/충돌 확률이 아니라 주변 높이 불연속 크기다. CLI의 iterations도 optimizer 횟수가 아닌 camera ticks로 명시했다.
6. student 실행의 actor 입력은 teacher와 동일한 4프레임 경로를 사용한다.
   actor exporter는 time-major 배포 history를 학습용 term-major로 바꾼다.
7. 학습 시 teacher는 Warp ray-cast 높이맵을 사용하므로 camera 렌더링이 필수는 아니다.
   student 학습/실행에는 TiledCamera 렌더링이 필요하다.

기본 PPO: rollout 100 steps/env, 5 epochs, 4 minibatches, γ=0.99, λ=0.95,
clip=0.2, entropy=0.01, adaptive LR(초기 1e-3, 하한 3e-5), 좌우 mirror augmentation.
실행 인자/experiment arm이 이 값을 덮어쓸 수 있다.

## Export 계약

Actor: `direct_obs[B,46]`, `cenet_obs[B,230]`(time-major), `terrain_latent[B,32]`
→ `actions[B,12]`, `z_t[B,19]`. 여기서 z_t는 CENet code이며 terrain latent가 아니다.

Student: `frames[1,4,2,45,80]`, `hidden_in[1,64]`
→ `terrain_latent[1,32]`, `hidden_out[1,64]`. 배포 batch는 1로 고정되어 있다.
GRU hidden은 배포 측에서 다음 camera tick에 되먹임해야 한다.

Blind actor(203 입력)와 VRL actor(235 입력)는 체크포인트를 그대로 strict load할 수 없다.
새 teacher는 VRL 구조로 학습하거나 명시적인 weight migration이 필요하다.

## 검증 범위와 남은 위험

최종 결과: **68 passed, 1 skipped**, 35.60초. skip은 Isaac Sim 실행이 필요한 테스트 모듈이다.
Ruff, 프로젝트 convention 검사, import 계층 검사(3개 계약), `git diff --check`도 통과했다.
ONNX exporter deprecation warning 4건은 있었으며 테스트 실패는 없었다.

검사 코드는 `tests/test_platform_gap.py`, `tests/test_vrl_pipeline.py`에 있다.
실제 네트워크의 계층별 forward 출력, gradient 경로, 실제 RSL-RL PPO의 더미 rollout/저장/재개,
좌우 대칭 증강, JIT/ONNX 수치 parity와 GRU 연속 실행을 CPU에서 검사한다.
지형 mesh 형상 및 seed 재현성, 지형 열 비율, 발 빠짐, 성공 중복 방지, row별 reset,
피트/옆 우회/넘어짐/종료/무접촉에서 성공이 지급되지 않는지 검사한다.
보상 관리자 테스트는 실제 함수에 얇은 CPU adapter를 사용하며 PhysX 검증을 대체하지 않는다.

다음 항목은 아직 검증하지 않았으며 이 문서는 실환경 안전 인증이 아니다.

- 실제 Kit/PhysX에서 환경 생성·reset·접촉·보상·카메라 prim 초기화와 영상 갱신 타이밍.
- 10 cm 높이맵 간격보다 좁은 갭은 위치에 따라 teacher scan이 놓칠 수 있다.
  관측 차원을 유지하기 위해 이번 변경에서는 scanner 해상도를 바꾸지 않았다.
- IR 채널은 RGB grayscale 기반 proxy + noise다. 실제 active IR 물리 모델과 같지 않다.
- 카메라 FoV/몸통 가림으로 관측할 수 없는 영역까지 teacher 높이맵에는 포함된다.
  Student recurrent memory가 보완할 수는 있지만, 한 번도 보지 못한 지형의 복원은 보장되지 않는다.
- Student는 현재 teacher rollout의 latent distillation이다. student가 직접 제어하는 rollout의
  성공률·발 빠짐·가장자리 접촉·회복 성능은 별도 폐루프 평가가 필요하다.
- vendor의 최신 카메라 extrinsic/intrinsic 및 실기 전처리와의 일치, MuJoCo/실기 배포는 이번에 변경·실행하지 않았다.
- 갭 폭·높이차와 보상 가중치는 초기 설계값이다. 마찰·하중·센서 결손을 포함한 실측 성공률에 따라 조정해야 한다.

---

## 후속 (2026-09-21 저녁) — 확인된 결함, 미수정

헤드리스 재개 중 코덱스가 아래를 확인했으나, AppArmor 의 비특권 user namespace
제한(`apparmor_restrict_unprivileged_userns=1`)으로 `bwrap` 파일 샌드박스가
`RTM_NEWADDR: Operation not permitted` 를 내며 **파일 쓰기가 전부 차단**되어
수정하지 못했다. 읽기·분석만 수행됐고 커밋은 없다.

> VRL 환경이 `terrain` 관측 그룹과 카메라 계약을 실제로 사용하지 않고 있었고,
> 교사 latent 가 critic 의 **전체 height scan** 을 직접 정규화해 읽고 있었다.
> critic 전체 scan 은 유지하면서 actor/증류 target 만 **카메라 가시 mask 가 포함된**
> 공통 입력을 쓰도록 정렬해야 한다.

이는 위 "검증 범위와 남은 위험" 의 *"카메라 FoV/몸통 가림으로 관측할 수 없는 영역까지
teacher 높이맵에는 포함된다"* 항목이 실제 결함으로 확인된 것이다. 교사가 카메라로
볼 수 없는 영역까지 담은 latent 를 만들면 학생은 원리상 그것을 재현할 수 없다.

남은 작업: terrain 관측 그룹과 actor/teacher latent 연결 보완 → 테스트 실행 → 커밋.
재개 시 `-c sandbox_mode=danger-full-access` 를 쓰거나 AppArmor 제한을 먼저 해제한다.
