# GAST Ray+IR 교사 설계문서 — 원래 3906 기준 (현행 학습, 2026-10-18)

이 문서는 **이미 수행한 학습**(DWB-38000 → Ray+IR gap50 3906 → aug149 파인튠)의 구조와 방법을 정리한다.
새로 계획 중인 NavSkill 학습은 `GAST_NAVSKILL_DESIGN_20261018.md`에 따로 있다.

출처:
- 3906 학습 설정: `checkpoints/teachers/gast/rayir_top5_20261016/3906_gap50_from_dwb38000/params/{env,agent,gast_warm_start}.yaml`
- 코드:
  - `src/gd_lab/gast/{rayir,temporal,teacher,observations,geometry}.py`
  - `src/gd_lab/rl/{actor_critic,cenet}.py`
  - `src/gd_lab/mdp/gap_stair_v21.py`
  - 브랜치 `aug149-mapper-aug` (04813ed)
- 계보: `docs/lineage/records_20261016/LINEAGE_20261016.md`, `rbq10_lineage.html`

---

## 1. 계보 요약 (3906 줄기)

| 단계 | 시작점 | 핵심 변경 | env · iter · GPU@host | 결과 |
|---|---|---|---|---|
| DWB-38000 | scratch | ARM4 블라인드 DreamWaQ (`Gd-Blind-Rbq10-Dreamwaq-v0`) | 4096 · 0→38000 · 1G onvlm2 | SHA c6cfe178, 블라인드 뿌리 |
| **Ray+IR gap50 → 3906** | DWB-38000 `blind_init` (45개 텐서 복사, 지형 24개 새로) | 레이캐스트 가시성 + IR 채널 + GAST 시간기억, platform_gap 비율 50% | 1024 · 0→5000 · 1G 90.111 | Top-1 3906 (SHA f3e7afe5). E1 결과: v25or 갭 96.8%, 카메라+필터 96.2%, 계단 100% |
| aug149 A3906 | 3906 strict resume | 매퍼 오차 증강(actor 전용) | +2000 · 90.111 | 후보: 5737 top1, 4999, l1500, l2000(=5906) |

같은 방법의 다른 줄기도 있다:
- 2769: b1_18 ONNX → 17206 → 21068 → GAST v2.1 2000 → Ray+IR → 2769
- 2836: GAST 26123 → Ray+IR gap50 → 2836
- aug149 자식: A2769 (top1 3815), C2769 (대조군, top2 4146)

---

## 2. 제어·로봇 설정 (ARM4)

| 항목 | 값 |
|---|---|
| 물리 / 정책 주기 | sim dt 0.005 s × decimation 2 → **100 Hz** |
| 에피소드 | 20 s |
| PD | Kp hip 123.39 / knee 127.77, Kd 2.4 (DelayedDCMotor, 0–1 step 지연) |
| 토크 한계 | 93.6 / 126 N·m |
| 행동 | `JointPositionActionWithLimit`, scale 0.25, default offset, soft margin 5° |
| 액추에이터 랜덤화 | Kp·Kd × U(0.85, 1.15), reset마다 |

---

## 3. 관측 구조

### 3.1 Proprio (`policy` 그룹, 46차원 × 5 history)

| 항목 | 차원 | 노이즈 |
|---|---|---|
| base_ang_vel | 3 | ±0.2 |
| projected_gravity | 3 | ±0.05 |
| velocity_commands | 3 | — |
| joint_pos_rel | 12 | ±0.01 |
| joint_vel_rel | 12 | ±1.5 |
| last_action | 12 | — |
| payload_mass | 1 | — |

- history_length 5 → CENet 입력 230차원.
- `velocity_target_slice` (45, 48) 구간이 CENet 속도 추정의 타깃이다.

### 3.2 지형 (Ray+IR, `RaycastIrTerrain`)

- **격자**: 몸체 yaw 정렬 11×17, 10 cm 간격 (1.6 × 1.0 m). 셀 187개.
- **가시성**: GT 히트점 중 장착 카메라 4대(카메라 계약 `vendor_new`)에서 레이캐스트로 보이는 셀만 사용한다. 다리 캡슐이 가리는 것도 반영한다.
- **노이즈 체인** (한 term이 전부 담당):
  1. 높이 노이즈 1 cm + 에피소드 bias N(0, 1.5 cm)
  2. 3% dropout
  3. blackout (`BlackoutSchedule`)
     - 스텝당 시작 확률 0.001, 길이 50–300 step
     - 에피소드 전체 blackout 5%
     - blackout 시 IR도 0
  4. 무효 셀은 height 0 / valid 0 / IR 0
- **캡처 시점**: 카메라 계약 시계에 맞춰 캡처하고, 다음 캡처까지 값을 hold한다. height·mask·IR은 같은 캡처 pose를 공유한다.
- **IR 프록시** (셀 단위; gap·지형 라벨은 사용하지 않음):
  - 식: `I = clip(a + g·ρ(x,y)·max(0,n·l)·(r0/r)^0.5 + N(0,σ), 0, 1)`
  - 상수: a 0.05, gain 0.57, r0 0.6 m, σ 0.03, IR dropout 3%.
  - ρ는 월드에 고정된 재질 필드다. 어두운 패치 약 10%, 밝은 패치 약 15%, 저반사 띠가 있다.
  - g는 에피소드별 노출값이다.
  - 인코딩은 `0.05 + 0.95·I`이고, IR이 없는 셀은 0이다.
  - 보정 근거: noen rosbag RealSense IR (평균 0.34, p10 0.22, p90 0.46, 포화 없음).
  - 한계: IR은 depth가 유효한 셀에만 존재한다.
- **출력 그룹**:
  - `terrain` = [height 187, valid 187]. latent gate와 critic이 쓴다.
  - `gast_history` = 8 프레임 × (3·187 + age) = **4496**.
    - 프레임 offset은 (48, 32, 16, 8, 4, 2, 1, 0) step, 즉 0.1 s 단위로 최대 4.8 s 과거다.
    - 과거 프레임은 현재 pose로 warp한다 (`warp_memory`).
  - `gast_clean` = 깨끗한 높이 / valid / semantic gap / gap-known. 재구성 보조 손실의 타깃이다.

### 3.3 Critic (privileged)

- Proprio 전부
- base_lin_vel
- feet_around_height, feet_contact, contact_forces
- friction, base_mass_offset, actuator_gain_scale
- push_delta_v
- 전체 height_scan
- payload

---

## 4. 네트워크

```
proprio history (230) ──► CENet (VAE; enc 128-64, dec 64-128)
                          → z_t = v̂ (3) + latent (16) = 19    [detach 후 actor로]
gast_history (4496) ──► TemporalTerrainEncoder
    프레임별 CNN: Conv(3→8) → Conv(8→16) → AvgPool(5×8) → FC 640→64
    + age 임베딩 (age/5, valid 평균)
    → MultiheadAttention(64, 4 heads) → LayerNorm → GRU(64) → FC 32 + Tanh
    → terrain latent 32  (valid 셀이 하나도 없으면 0)
actor  : [proprio 46×4 (최근 4 step) | z_t 19 | latent 32] = 235 → MLP 512-256-128 (ELU) → 12 행동
critic : privileged obs → MLP 512-256-128
aux    : latent 32 → FC 128 → 187×6 재구성
         (height smooth-L1 + valid/gap BCE, gap 셀 가중 ×5)
```

- 재구성 보조 손실:
  - 계수 0.1
  - 업데이트마다 2048개 샘플
  - PPO와 같은 Adam을 쓴다.
- CENet:
  - β 1.0, lr 1e-3
  - AdaBoot: 최소 10 업데이트, EMA 0.8
  - 학습 중에는 GT 속도 주입 비율을 조절하고, 추론 때는 추정값을 쓴다.
  - **CENet 이상치 방어 = 타깃 clamp** (velocity / recon / history). gradient clipping은 측정 결과 효과가 없었다.
- 대칭: `mirror_gast` 데이터 증강을 사용한다. mirror loss는 0이다.

---

## 5. 초기화 (3906: `blind_init`)

| 항목 | 내용 |
|---|---|
| 출처 | `dwb_model_38000.pt`, SHA c6cfe178 (기대값과 일치 확인), source_kind `blind_dreamwaq` |
| 복사 | 45개 텐서: actor, CENet, normalizer 등 |
| 새로 학습 | 24개 지형 텐서. terrain encoder와 decoder |
| latent head | 0으로 초기화. 처음에는 블라인드와 동일하게 동작한다 |
| 옵티마이저 | PPO 새로. CENet optimizer는 복원하지 않음 |
| 학습률 | policy 1e-4, terrain 1e-3 |
| schedule | fixed |

2769는 GAST 2000에서 warm start했다. 이때 첫 conv의 입력 채널을 2개에서 3개로 늘렸고, IR 채널 가중치는 새로 학습했다.

---

## 6. 환경

### 6.1 지형

- curriculum generator: 15 rows × 20 cols, 타일 8 × 8 m, max_init_level 3.
- 비율 가중치를 모두 더하면 20이다. platform_gap이 10/20 = **50%**를 차지한다.

| sub-terrain | 비율 | 범위 |
|---|---|---|
| platform_gap | 10 | 폭 0.02–0.26 m, 높이 차 0–0.16 m |
| pyramid_stairs / inv | 1 / 1 | 0.05–0.20 m |
| stairs_wide / inv_wide | 1 / 1 | 0.05–0.25 m |
| stairs_nose / inv_nose | 1 / 1 | 0.05–0.25 / 0.20 m |
| random_rough | 1 | 0.01–0.10 m |
| hf_pyramid_slope / inv | 1 / 1 | 0.1–0.5 |
| ring_fence | 1 | 0.03–0.15 m |

- curriculum 항목:
  - terrain_levels (family별)
  - command lin / ang levels
  - penalty_terrain_schedule
  - family별 termination 통계

### 6.2 명령 (`V21VelocityCommand`)

| 항목 | 값 |
|---|---|
| resample 주기 | 3–7 s |
| heading 명령 | 100% (stiffness 0.5) |
| standing env | 20% |
| vx | ±1.2 m/s |
| vy | ±1.0 m/s |
| ωz | ±1.0 rad/s |
| heading | ±π |

platform_gap 위에서는 heading을 gap 방향으로 맞추는 규칙이 있다.

### 6.3 도메인 랜덤화 / 외란

| 항목 | 값 |
|---|---|
| 마찰 | static 0.2–1.25, dynamic 0.16–1.0 (64 bucket) |
| 몸통 질량 | ±2 kg. COM과 관성도 랜덤화 |
| 페이로드 | 0 kg 20% / 6 kg 80% (x ±5 cm, z +7 cm). proprio 관측에 포함 |
| push | 10–15 s 간격, Δv ±0.5 m/s |
| 계단 손잡이 외란 (오르막) | 0–200 N, 0.3–1.0 s, 20–45°. 가끔 300–400 N 저크 |
| 계단 손잡이 외란 (내리막) | 0–150 N, 0.1–0.5 s |
| 초기 상태 | x, y ±0.5 m, yaw ±π, 관절 × U(0.5, 1.5) |
| 종료 조건 | time_out, base_contact, nonfinite_state |

### 6.4 보상 (v2.1 기본 + clean gap + stair handle)

**기본 locomotion**

| 보상 | 가중치 |
|---|---|
| track_lin_vel_xy_exp | 1.0 |
| track_ang_vel_z_exp | 0.8 |
| lin_vel_z_l2 | −2.0 |
| ang_vel_xy_l2 | −0.15 |
| base_height | −15 |
| flat_orientation roll / pitch | −2 / −1 |
| dof_torques | −2.5e-5 |
| dof_acc | −2.5e-7 |
| action_rate | −0.02 |
| smoothness | −0.02 |
| joint_power | −2e-5 |
| dof_vel | −5e-4 |
| dof_pos_limits | −2 |
| undesired_contacts | −1 |
| termination | −2 |
| joint_deviation_hip | −0.3 |
| stand_still / stand_still_feet_contact | −0.5 / −1 |
| foot_clearance | −1 |
| feet_slipping | −0.1 |
| feet_touchdown | −1 |
| feet_cadence_overrun | −1 |
| overspeed | −1 |

**갭**

| 보상 | 가중치 |
|---|---|
| platform_gap_foot_drop | −2.0 |
| crossing | +0.5 |
| monitor | 1.0 |
| intrusion | −3.0 |
| clean | +0.5 |
| gap_foothold_margin | −0.25 |
| gap_slot_probe | −1.0 |
| gap_hind_hop | −1.0 |
| gap_overlift | −0.25 |
| gap_foot_clean | +0.25 |

**계단**

| 보상 | 가중치 |
|---|---|
| stair_foothold_margin | −0.2 |
| stair_handle_disturbance | −1.0 |
| stair_push_fall | −3.0 |
| stair_push_slip | −0.2 |
| stair_stall | −0.5 |

---

## 7. 학습 알고리즘 (`GastPPO`)

| 항목 | 값 |
|---|---|
| horizon | 100 step/env |
| epochs | 5 |
| minibatches | 4 |
| lr | 1e-3 (fixed, 하한 3e-5) |
| γ, λ | 0.99, 0.95 |
| clip | 0.2, clipped value loss |
| entropy | 0.01 |
| desired_kl | 0.01 |
| max_grad_norm | 1.0 |
| init_noise_std | 1.0 (scalar) |
| obs normalization | actor와 critic 모두 사용 |
| 3906 규모 | 1024 env × 5000 iter, 1 GPU (90.111), seed 42 |

**비유한값 방어**

- rollout 또는 상태에 NaN/inf가 있으면 해당 업데이트를 건너뛰고 마지막 정상 스냅샷으로 복원한다.
- 보조 손실 단계에서는 gradient를 fp64 norm으로 검사한다. 랭크 간 평균은 정상일 때만 낸다.
- GAST v2.1의 8016 / 8148 정지는 fused SDPA backward가 원인이었다. 그래서 `GAST_MATH_SDPA=1`로 math kernel을 쓴다.
- encoder 실행은 8192개 단위 chunk로 나누고 checkpoint한다. 대칭 증강으로 배치가 두 배가 되면 CUDA attention grid 한계를 넘기 때문이다.

**Online Top-5**

- 체크포인트 간격 500 iter 이상.
- platform_gap 평균 레벨이 8.0 이상이어야 후보가 된다.
- 저장 주기는 250 iter.

---

## 8. aug149 매퍼 오차 증강 파인튠 (A3906 / A2769 / C2769)

**목적**: Isaac의 깨끗한 Ray 입력과 실제 배포 경로 사이의 오차를 actor가 견디게 하는 것.
배포 경로는 4× D4xx depth → camfilter136 → v25or 매퍼다.

**증강 내용** (actor 입력에만 적용, critic은 clean 유지)

- S0: deepest-of-5 footprint와 에피소드 고정 pit 확장. 최종 질의 반경은 10 cm 이하다.
- gap 타일에서만 support-0.7 m pit 깊이를 적용한다.
- 군집형 unknown: valid 셀의 5–15%를 blob과 맵 경계 띠로 지운다.
- IR은 센서 단서로 유지하고, 무효 셀은 0으로 둔다.
- 에피소드의 25%는 clean으로 둔다.
- 국소 500 업데이트 동안 강도를 선형 ramp한다.
- 증강은 별도 RNG를 쓴다. 기존 노이즈 난수열을 바꾸지 않는다.

**strict resume** (`--resume_rayir149`)

- 정확한 SHA / task를 확인한다.
- PPO와 CENet optimizer, family 레벨까지 복원한다. 조용한 fallback은 없다.
- noise / force / speed는 원본의 절대 진행도를 유지한다.
  - 주의: `--max_iterations`는 누적이 아니라 추가분이다. resume 시 curriculum이 리셋되는 이슈가 있다.
- Online Top-5는 국소 +500 이후부터 집계하고, ramp 구간 에피소드는 제외한다.
- local_0250 … 2000 체크포인트를 보존한다.

**규모**: +2000 iter. A2769·C2769는 onvlm2, A3906은 90.111에서 학습했다.

---

## 9. 평가 파이프라인

| 단계 | 내용 |
|---|---|
| E1 (Isaac 폐루프) | 32 env × 30 s, seeds 42–45 (초기 상태 짝 맞춤), vx 1.0. 입력 경로 G / GV / v25or(LiDAR) / camera+camfilter136 / IR=0. 지표: gap clean/도달, 계단(pit climb-out, pyramid up) |
| MuJoCo depth_ir | `gd_rbq10_deploy_vrl`의 Pilot C++ + sensor sidecar (GSC2 UDP, 4× depth+IR 렌더 → 매퍼). gap-prefix A–E arm, 시드 12회 |
| force4 | 계단 코스의 4지점(오르막 당김/밂, 내리막 당김/밂) 자동 외력 + depth_ir, 고정 시드 반복 (`force4_depthir_repeat.sh`, `fall_watch.py` 조기 종료) |
| 리뷰 | `deploy/mujoco_review_20261018/<model>/run_sim.sh`. 조이스틱으로 수동 확인. 기본 PAYLOAD 6 kg, 외력 150 N × 2 s |

**주요 결과 (현재까지)**

| 문제 정의 | 결과 | 반영 방향 |
|---|---|---|
| 원래 3906 E1 | v25or 갭 96.8%, 카메라+필터 96.2%, 계단 100% | 기준선 |
| A3906 증강 효과 (E1) | camf gap 98.7–100% | Isaac에서는 개선 |
| MuJoCo depth_ir 12시드 | A2769_top1 11/12, 원래 3906 10/12, A3906 l2000 10/12, A3906 top1(5737) 1/5 (oracle 2/6) | 5737은 탈락. 3906 vs l2000은 동률 |
| force4 (200 N × 2 s) | 3906이 오르막 당김에 가장 약함 (센서 10/20). 26123이 가장 강함 | 학습 범위(오르막 ≤200 N, ≤1 s / 내리막 ≤150 N, ≤0.5 s)를 넘는 조건이었다. 150 N × 2 s로 재측정할 예정 (100회 반복은 사용자 요청으로 중단됨) |

---

## 10. 배포 계약 (현행)

- 학습 env.yaml의 `camera_profile` (`vendor_new`)이 평가 시점과 일치해야 한다. 가중치 옆에 `params/env.yaml`이 반드시 있어야 한다.
- pose:
  - 시뮬레이션(Isaac 학습·E1·MuJoCo)은 GT pose를 쓴다. pose 오차는 넣지 않는다 (2026-10-16 결정).
  - 실기 pose는 camel_pilot이 발행하는 `/odom`(nav_msgs/Odometry, odom→base_link, 50 Hz)을 쓸 예정이다. rosbag에서 토픽을 확인했다.
  - 모델의 pose 입력 형식(캡처마다 world x, y, yaw)은 `/odom`과 맞춰 두었다. 포팅할 때 토픽만 바꾸면 된다.
  - 우리 Pilot 사본에는 `/odom` 구독이 아직 없다. 실기 포팅 때 구현한다. odom 오차는 그때 다룬다.
- 매퍼 출력: 11×17 [height, valid, IR] 프레임을 카메라 주기로 보낸다. Pilot이 8 프레임 history를 warp해서 encoder에 넣는다.
- 번들: `gd_rbq10_deploy_vrl/resources/policy/gast/gast_rayir_*`. 런처는 `scripts/gast/teacher/<name>/run_sim(_ir).sh`.
- sim_docker용 `camel.policy.v1` GAST runtime은 브랜치 `feat/gast-teacher-runtime`에 있다. offline parity 약 1e-6, 시뮬 내 smoke 테스트는 아직 하지 않았다.

---

## 11. 알려진 한계 / 교훈

1. **두 블라인드 뿌리.**
   - 3906은 우리 ARM4 DWB-38000(100 Hz)에서 출발했다.
   - 2769는 50 Hz b1_18 ONNX에서 출발했고, 그 뒤 계보를 100 Hz로 학습했다.
   - 두 줄기의 외란 특성이 다르다. 오르막 당김에는 3906이 약하다.
2. **10 cm 격자, 1.6 × 1.0 m 범위.**
   - 좁은 통로나 후방 인식에는 부족하다.
   - NavSkill에서 확장하는 것을 검토 중이다.
3. **IR 프록시의 한계.**
   - IR은 셀 단위다.
   - depth가 무효인 곳에는 IR도 없다.
4. **v25or deepest-of-5 샘플링.**
   - pit을 5–10 cm 넓게 본다.
   - aug149는 이 오차를 증강으로 흉내 내서 대응한다.
5. **수치 안정성.**
   - fused SDPA 대신 math SDPA를 쓴다.
   - CENet은 타깃을 clamp한다.
   - 비유한값이 생기면 업데이트를 건너뛰고 스냅샷으로 복원한다.
6. **외력 평가 조건이 학습 범위를 넘었다.**
   - 200 N × 2 s는 학습 범위 밖이다.
   - 평가 외력은 학습 분포와 맞춰야 비교가 의미 있다.
