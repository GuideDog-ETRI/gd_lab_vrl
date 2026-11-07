# 한강 작업 지시(설계안): Stage B′ — 좁은 통로 + 후진 통로, A′(G5) 모델 출발 — 2026-11-01

**상태: 설계안.** 코드는 아직 없다. 순서: 사용자 결정(§8) → 구현(§6, 새 파일만) → 로컬 smoke → a6000 사전 검증(g5fin 출발, 300 update) → A′ 모델 선택 → 한강 제출. A′(job 1194183, G5 레시피 = G4 + R20)가 도는 약 34 h 동안 준비한다.

앞 문서: `WORKORDER_STAGE_B_V25_20261031.md`(B 원안, v25p 레시피·감시 규칙·재제출 체인), `WORKORDER_STAGE_APRIME_20261101.md`(A′), `GAIT_V3_APRIME_TERMS_20261101.md`(보행 항목·충돌 표), `GAIT_V3_DESIGN.md` §3.7–3.8, `GAST_NARROW_V25_DESIGN.md`.

## 1. 목적과 판정

A′ 최고 모델에서 **v25p 통로 레시피 + 보행 v3(G5) 항목**으로 좁은 통로 통과·막히면 서기·막다른 길 서기·후진 통로를 학습한다. 회전·낭떠러지 타일은 없다(C1·C2 단계). B 원안의 판정에 A′가 얻은 보행 지표를 유지한다는 조건을 더한다.

| 기능 | 판정 (E2, `configs/gast_narrow/e2/stage_bprime_hangang.yaml`, §7) |
|---|---|
| 좁은 통로(전·후진), 막히면 서기, 막다른 길 서기 | 602 − 1 %p 이내, 막다른 길 서기 ≥ 90 %, 막힌 통로 침범 ≤ 5 %, 후진 통로 ≥ 전진 − 5 %p |
| 전·후진 갭 | 건넘 ≥ A′ − 1 %p, 후진 갭 넘어짐 ≤ 1 %, **후진 0.3 m/s 셀에서 정지 없음**(R20 확인), 전진 갭 머리 20° 초과 ≤ 약 10 % |
| 계단 상·하행, 내리막 경사 | A′ − 1 %p 이내, 내리막 경사 ≤ 3.1 % |
| 보행(A′ 문서 §5) | 평지 머리 들림·착지 속도·발 높이가 A′ 수준(MuJoCo + `tools/gait_diag/diagnose.py`) |

## 2. 레시피: v25p와 A′(G5)의 합성

| 항목 | v25p (B 원안) | A′ (G5) | **B′** | 이유 |
|---|---|---|---|---|
| 과제 클래스 | `GastNarrowV25TeacherCfg` (`NARROW_TURN=0 NARROW_CLIFF_COLUMNS=0`) | `GastGaitV3GapTeacherCfg` | 새 `GastNarrowBPrimeTeacherCfg(GastNarrowV25TeacherCfg)` + 보행 v3 항목 적용 | 둘 다 6566 과제의 자식. 통로 쪽을 부모로 두고 보행 항목을 얹는다(§6) |
| 지형 | 44열(§3) | 6566 지형 | **44열(v25p)** | 통로 단계. 보존 열에 갭 10·계단 2+2·나머지 1열씩 |
| 명령 | `NarrowV25VelocityCommand` (통로 타일에서 vx·wz 덮어씀, `backward=False`) | `SlowBackGapVelocityCommand` (R20) | 새 `BPrimeVelocityCommand(NarrowV25VelocityCommand)`: `super()._resample_command` 뒤 갭 타일의 후진 속도만 R20 분포로 다시 뽑음 | 통로 덮어쓰기는 통로 타일만, R20은 갭 타일만 — 겹치지 않음(`mdp.py _apply_narrow` 확인) |
| 보행 항목 R1·R4–R10·R15–R19 | 없음 | 켬 | **켬, A′ 가중치 그대로. R1은 통로 타일에도 일반 보행과 같은 기준(3°)** (사용자 11-01) | A′가 배운 보행을 유지 |
| ramp | — | κ 0.2 → 1 (200 update) | **끔**(`GAIT_RAMP_STEPS=0`) | 출발 모델이 이미 보행 항목을 학습함(A′ 이어 학습과 같은 규칙) |
| 갭 머리 들림 | `gap_nose_up` 5°, −30 /s·rad + `gap_nose_peak` −6 (v2.3·v2.4 통로 과제가 추가) | R1만 (슬롯 근처 12°). A′ env.yaml에 `gap_nose_*` 없음, `backward_nose_up` null | **A′와 같게: `gap_nose_up`·`gap_nose_peak` 제거, R1만** | 통로 부모 클래스가 다시 넣는 항목을 걷어내 제출된 A′ 규칙을 유지 |
| 통로 머리 들림 `narrow_nose_up` | 서기 10° / 보행 20°, −3 | — | **제거** | R1(3°)이 통로에서도 켜지므로 더 느슨한 같은 대상 항목은 이중 감점일 뿐 |
| `feet_slipping` | −0.1 | −0.3 | −0.3 | A′ 값 |
| `gap_foothold_margin` | 4 cm, −0.25 | 6 cm, −0.4 | 6 cm, −0.4 | A′ 값 |
| `backward_nose_up` | −5 | 삭제 | 삭제 | R1이 대신함 |
| 보존 앵커 | 비통로 타일에서 STAGE_A 행동으로 MSE, `NARROW_ANCHOR_GAP_NOSE=1` | — | STAGE_A = A′ 모델, **`NARROW_ANCHOR_GAP_NOSE=0`** | 갭 머리 들림 항목을 제거하면 마스크가 없음. A′는 갭에서 머리를 들지 않으므로 제외할 표본이 없다 |
| 서기 앵커 | 차단 플래그 표본 → 출발 정책의 명령 0 행동 | — | 그대로(목표 = A′) | |
| Top-5 | `online_top5_lineage16_b_v25p.json` | `online_top5_g2.json` | v25p 파일 그대로 | 통로 관문(서기 ≥ 0.85, 침범 ≤ 0.05, 막다른 길 ≥ 0.85, 들림 6 s 초과 ≤ 0.3) + 갭 레벨 ≥ 8.5·갭 종료 ≤ 0.145·계단 종료 ≤ 0.078 |

## 3. 지형 44열 (v25p)

| 패밀리 | 열 | 비고 |
|---|---|---|
| `narrow_corridor` 4, `narrow_door_pos` 3, `narrow_door_neg` 3, `narrow_bend_left` 4, `narrow_bend_right` 4, `narrow_dead_end` 4 | 22 | 폭 1.20 → 0.50 m(막다른 길 0.80 m). 후진 통로 25 %(`reset_narrow_back`), 후방 Mid-360 70 % |
| `platform_gap` | 10 | 전·후진 갭. R20 후진 속도 분포 |
| `pyramid_stairs`, `pyramid_stairs_inv` | 2 + 2 | 계단 보존 |
| `pyramid_stairs_wide`, `_inv_wide`, `_nose`, `_inv_nose`, `hf_pyramid_slope`, `hf_pyramid_slope_inv`, `random_rough`, `ring_fence` | 1씩 = 8 | 3906 지형 보존(`RETENTION_COLUMNS` 기본 1) |

높이 스캐너(정책의 하방 Depth+IR 지형 입력 `RaycastIrTerrain`과 보행 항목 R1·R6·R15가 쓰는 레이캐스트)는 1.6 × 1.0 m 격자(17 × 11, 0.1 m)로 몸통 좌우 ±0.5 m를 본다. 통로 벽(높이 1.0 m 상자)은 지형 메시의 일부라 **지금은 벽 윗면이 하방 지형 입력에 그대로 들어온다**(602 계열도 그렇게 학습). **B′ 결정(사용자 11-01): 좁은 통로에서는 하방 Depth를 쓰지 않고 LiDAR만 쓴다** — §4 C2–C4, C14.

## 4. 충돌 점검 (B′에서 새로 생기는 것)

A′ 문서 §4의 충돌(계단·갭·경사·후진 갭·ramp)은 B′에서도 같은 회피 방법이 그대로 적용된다(§5). 아래는 통로 타일·통로 항목·앵커가 더해져 **새로 생기는** 충돌이다.

| # | 충돌 | 근거 (측정·코드) | 회피 방법 | 남은 위험 |
|---|---|---|---|---|
| C1 | R1 머리 들림(3°, −4 /s·rad) ↔ 통로 자세·막다른 길 서기·`narrow_nose_up` | 602 계열은 통로 보행 13°, 막다른 길 서기 13.6°(v25 설계 §2) — 하지만 B′ 출발점은 A′(평지 3° 습관). `narrow_nose_up`(10°/20°)은 R1보다 느슨한 같은 대상 | **R1을 통로에도 일반 보행과 같게 적용(사용자 11-01), `narrow_nose_up` 제거** | 통로에서 벽을 보려고 머리를 드는 행동이 막힘 — 막다른 길 서기·막힌 통로 서기 관문으로 감시. R1 지형 경사 계산(`ground_pitch`)에 문 벽이 섞이는지 확인(C4) |
| C2 | R6 발 들기(`foot_clearance_ground`, −8 /s·m) ↔ 통로 벽 | 처음 추정: 벽 옆 발의 최근접 히트가 벽 윗면이면 오벌. **실측(11-01, A′형 정책 36 env 통로 replay): 오벌 없음** — 벽을 그대로 읽으면 R6 −0.004 /s, 바닥 기준 −0.025 /s(평지 보행과 같은 수준). 벽 옆 히트가 오히려 부족분을 가려 R6가 통로에서 거의 꺼져 있었음 | **통로 타일에서 타일 바닥 높이를 지면으로 사용**(하방 Depth 미사용 원칙) → 통로에서도 평지와 같은 발 들기 기준 | 없음 |
| C3 | R15 발 높이 상한 ↔ 통로 벽 | 벽이 스캔에 보이면 '평평한 바닥' 판정이 꺼져 R15가 통로에서 대부분 0. 실측: 벽 그대로 −0.019 /s, 바닥 기준 −0.081 /s | **통로 타일에서 바닥 기준 → 평지와 같이 9 cm 상한** | 없음 |
| C4 | R1 지형 경사 기준(`ground_pitch`) ↔ 벽 | 앞·뒤 \|y\| < 0.25 m 히트의 중앙값. 폭 0.5 m 문에서 벽 윗면이 섞이면 오르막으로 읽힘 | **통로 타일에서 지면 경사 = 0(바닥 평평)** | 없음 |
| C5 | R1(슬롯 근처 12° 허용) ↔ 통로 부모가 추가하는 `gap_nose_up`(5°, −30) + `gap_nose_peak`(−6) | 제출된 A′에는 두 항목이 없음(env.yaml 확인). 남기면 같은 대상 3중 감점, 5° 기준은 R1이 12°를 허용한 이유(clean −10 %p)와 충돌 | **제거(A′ 규칙 유지), `NARROW_ANCHOR_GAP_NOSE=0`** | — |
| C6 | 보존 앵커(비통로 타일에서 A′ 행동으로 당김) ↔ 보행 항목 | 목표가 A′ 자신이므로 보행 항목과 같은 방향 | 충돌 없음 | A′에 남은 결함(예: 느린 후진 머뭇거림)도 함께 고정된다 → **A′ 선택 관문에 후진 0.3 m/s 셀을 포함**(§1) |
| C7 | 서기 앵커·`stand_still_feet_contact_stand`·`feet_cadence_overrun_stand_gated` ↔ R6·R7·R15·R17 | 서야 하는 구역에서 명령은 유지되고 보행 사건 항목(착지·스윙)은 서 있으면 0. R6·R15는 moving 게이트, R17은 슬롯 근처만 | 충돌 없음 | — |
| C8 | R16 후진 과속(갭·계단) ↔ `narrow_overspeed_capped`(통로, min(명령 + 0.15, V_SAFE + 0.10)) | 타일이 분리됨 | 충돌 없음 | — |
| C9 | R20 후진 갭 속도 분포 ↔ 통로 후진(25 %, `reset_narrow_back`) | R20은 갭 타일의 `backward` 환경만. 통로 명령은 `_apply_narrow`가 덮어쓰고 `backward=False` | 충돌 없음(명령 클래스 상속 순서로 보장, §6) | — |
| C10 | `narrow_stall`(통과 가능 통로에서 정지 1 /s) ↔ 보행 항목 | 보행 항목 중 정지를 유도하는 것은 없음(R11 꺼짐). R17은 갭만 | 충돌 없음 | — |
| C11 | ramp 없음 ↔ 새 통로 항목 | 통로 항목(`narrow_*`)은 처음부터 v25p 값, 보행 항목은 전체 가중치 | 보존 앵커가 비통로 타일을 잡고, 통로 항목은 602를 만든 값 그대로 | 처음 50 update에 통로 보상이 보행 지표를 밀 수 있음 → u400 중간 E2에서 평지 지표(A′ 문서 §5) 확인 |
| C12 | 6566 `penalty_terrain_schedule` ↔ 보행 항목 | 스케줄 대상은 smoothness·action_rate·dof_acc·torque·power·dof_vel 6개뿐(G4 env.yaml) | 겹치지 않음(확인) | — |
| C13 | `feet_slipping` −0.3 ↔ 벽 접촉(`narrow_leg_touch` −1, `wall_proximity` ≤ 4 /s) | 미끄럼은 디딤발, 벽 접촉은 다리 — 다른 대상 | 충돌 없음 | — |
| C14 | **정책 하방 지형 입력(`RaycastIrTerrain`) ↔ 통로 벽** | 실측: 높이 채널에는 벽이 안 들어감(벽 윗면은 배 아래 카메라 시야 밖, 보이는 칸 높이 범위 2.6 cm). 대신 **벽 그림자로 보이는 칸이 줄어** 벽 위치가 드러남(통로 67.7칸 vs 바닥 기준 72.0칸) | **통로 타일에서 '벽 없는 평평한 바닥'으로 촬영**(`NarrowFloorRaycastIrTerrain`: 스캔 점을 바닥에 두고 가시성은 바닥 평면에 대해 검사). 벽 정보는 LiDAR로만 | 실기 정합(D5) |

## 5. A′ 문서 §4 충돌 표의 B′ 적용

| A′ §4 항목 | B′ |
|---|---|
| R1 ↔ 계단 / R1 ↔ 경사 | 그대로(계단 끔, 경사 기준). 계단·경사 보존 열에서 작동 |
| R1 ↔ 전진 갭 | 그대로(슬롯 12°). **C5로 v25p 항목과의 중복 제거** |
| R4·R7·R9·R10 ↔ 후진 갭 (R19) | 그대로 |
| R5 ↔ 갭 착지, R17 ↔ 갭 진행, R18 ↔ 좁은 발판 | 그대로. 보존 앵커가 같은 방향 |
| R6 ↔ R15, R6·R15 ↔ R4·R5 | 그대로. **통로에서는 R6 지면 선택 수정(C2), R15 자동 꺼짐(C3)** |
| 보행 감점 합 ↔ 추종·갭 보상 (ramp) | ramp 없음(C11). 출발 모델이 이미 학습 |
| ramp ↔ 6566 스케줄 | 해당 없음(C12) |
| R1 ↔ R13, R11 ↔ 버팀 | 그대로(R13 삭제, R11 꺼짐) |
| R20 ↔ 후진 갭 학습 분포 | 그대로(C9) |

## 6. 구현 범위 (3906·backgap·gast_narrow v2.x 코드 불변, 새 파일만)

| 파일 | 내용 |
|---|---|
| `src/gd_lab/gast_bprime/tasks.py` (새) | `GastNarrowBPrimeTeacherCfg(GastNarrowV25TeacherCfg)`: 보행 v3 항목 적용(`gast_gait.tasks`의 항목 추가를 함수로 빼서 호출), R6·R15·R1 `floor_on_narrow=True`, 지형 입력 `NarrowFloorRaycastIrTerrain`, `gap_nose_up`·`gap_nose_peak`·`narrow_nose_up` None, 명령 클래스 교체. `GastNarrowBPrimeRunnerCfg(GastNarrowV25RunnerCfg)` |
| `src/gd_lab/gast_bprime/commands.py` (새) | `BPrimeVelocityCommand(NarrowV25VelocityCommand)`: `_resample_command` 뒤 `self.backward` 환경(갭 타일)만 `sample_back_speed`로 다시 뽑음 |
| `src/gd_lab/gast_gait/terms.py` | `foot_clearance_ground`·`FootLiftCap`·`ground_pitch`에 `floor_on_narrow=False` 인자: 켜면 통로 타일 env에서 스캐너 대신 타일 바닥 높이(경사 0) 사용. 기본값 끔(A′ 경로 불변) |
| `src/gd_lab/gast_bprime/terrain_obs.py` (새) | `NarrowFloorRaycastIrTerrain(RaycastIrTerrain)`: 통로 타일 env의 하방 지형 입력을 평평한 바닥 값으로 대체 |
| `src/gd_lab/gast_gait/tasks.py` | `__post_init__` 본문을 `apply_gait_v3(cfg, *, narrow=False)`로 분리(동작 동일) |
| `scripts/gast_bprime/train_bprime.py` (새) | `train_narrow_v25.py`와 같은 로더(새 LiDAR 인코더 + 0 열 확장, STAGE_A 앵커)에 과제만 `Gd-GastNarrowBPrime-Rbq10-Dreamwaq-v0` |
| `scripts/gast_narrow/hangang/lineage16_b.sbatch` | `LINEAGE_RECIPE=bprime` 분기: 위 트레이너·과제, `NARROW_TURN=0 NARROW_CLIFF_COLUMNS=0 GAIT_RAMP_STEPS=0 GAIT_R20=1 NARROW_ANCHOR_GAP_NOSE=0`, Top-5 v25p, 실험 이름 `gast/lineage16_bprime` |
| `configs/gast_narrow/e2/stage_bprime_hangang.yaml` (새) | §7 |
| 이 문서 → 작업 지시로 확정, `GAIT_V3_DESIGN.md` §6 B′ 행 갱신 |

**구현 상태(11-01)**: 위 파일 모두 작성(작업본, 커밋 전). `scripts/gast_bprime/train_bprime_3gpu.sh`(onvlm2 GPU 0–2 실행기), E2 `task: bprime`(`e2_trial.py` 표에 추가), `lineage16_b.sbatch`의 `LINEAGE_RECIPE=bprime` 분기와 재제출 체크포인트 경로를 `$NARROW_EXPERIMENT` 기준으로 고침.

검증 순서(제출 전, 모두 로컬·a6000):
1. **[완료 11-01] A′형 모델(g4fin) 통로 replay**: B′ 과제로 36 env replay, 통로 타일에서 R6·R1·R15 비용 /s를 평지와 비교 → 통로에서 R6·R15·R1 비용이 평지 수준인지, 통로 타일 env의 하방 지형 입력이 평지와 같은지.
2. **[완료 11-01] 로컬 smoke(512 env, 3 update, g4fin 출발)**: source_kind gait_v3_aprime, LiDAR 인코더 10개 새로, actor 235→253·critic 298→307 0 열 확장, 44열, ramp 없음, 오류 없음. env.yaml에 `BPrimeVelocityCommand`, `NarrowFloorRaycastIrTerrain`, `gap_nose_up`·`gap_nose_peak`·`narrow_nose_up` null, R1·R6·R15 `floor_on_narrow: true`. A′ 과제(`GastGaitV3Gap`) 설정은 리팩터링 전과 보상·커리큘럼·명령이 완전히 같음(env.yaml diff).
3. **a6000 사전 검증(사용자 승인 11-01)**: g5fin에서 B′ 레시피 300 update(5,120 env, GPU 0–2), E2(통로·갭 0.3/0.5/0.9·계단). 관문: 통로 통과·서기가 v25p 단독 학습의 같은 시점과 비슷, 갭·계단 A′ − 2 %p 이내. A′가 끝나기 전에 레시피 결함을 거른다.
4. A′ 선택(§1 관문 + MuJoCo) → `STAGE_A` 확정.

## 7. E2·중간 E2

`stage_b_v25p_hangang.yaml` 복사본: 모델 `stageA`(A′), `v23u602`, `b6566`, B′ Top-5; 갭 속도 **0.3 / 0.5 / 0.9**; 통로 suite(통과·막힘·막다른 길 4 사례); 계단; **terrain suite 켬**(내리막 경사, 보존 열이 1열뿐이라 회귀를 E2로 본다). 중간 E2는 B 원안대로 update 400 / 800 / 1,200 / 1,600, 두 번 연속 평탄하면 STOP(사용자 결정).

## 8. 결정 (11-01)

| # | 내용 | 결정 |
|---|---|---|
| D1 | 갭 머리 들림 | 제출된 A′ 규칙 유지: R1만, 통로 부모의 `gap_nose_up`·`gap_nose_peak` 제거 |
| D2 | a6000 사전 검증(g5fin 출발 300 update) | 실행 |
| D3 | 통로 보행 머리 기준 | 일반 보행과 같게: R1(3°)을 통로에도 적용, `narrow_nose_up` 제거 |
| D4 | 좁은 통로 하방 Depth | **쓰지 않음, LiDAR만**(사용자 11-01): 정책 하방 지형 입력은 통로에서 평평한 바닥, 보행 항목 R1·R6·R15도 통로에서 바닥 기준 (C2–C4, C14) |
| D5 | 실기 정합 | 배포 하방 지도(camfilter136 → v25or + IR)가 통로에서 벽을 어떻게 넣는지 확인 필요. 학습과 맞추려면 통로에서 바닥만 남겨야 함 — B′ 배포 전 확인 |

## 9. 운용 규칙

B 원안 §5(모든 원격 명령 `timeout -k 10 120`, 보고는 요청 시 1회), §6(재제출 체인 확인), §9(한강 세션 대기, 모니터·루프 없음)를 그대로 쓴다. 한강에서는 push하지 않는다.

## 10. A6000 사전 검증 결과와 레시피 수정 (11-02)

| 문제 정의 | 결과 | 반영 방향 |
|---|---|---|
| 300 업데이트(g5fin 출발, bpfin) | 통로 통과 전진 tight 73 / 후진 77(602: 98 / 0.6), 막힌 통로·막다른 길 서기 0~16 %, 갭·계단 유지 | 서기 미학습 → 300 업데이트 이어 학습 |
| +300 업데이트(cont_fin, it300~599) | 막힌 통로 서기 96~98 %, 막다른 길 서기 100 %, 침범 2~4 %. 갭 전진 95.4 / 후진 94.7, 후진 0.3 정지 없음, 계단·slope_inv 유지. **전진 pass_tight(폭 0.66, 여유 0.05) 통과 11.9 %**(bpfin 76, 602 98) | 서기는 길이 문제가 맞았음. 좁은 통과 통로에서도 서는 새 문제 |
| 폭·여유 스윕 E2(cont_fin / bpfin / 602, 전진, 5가족 × 32) | 0.66/0.05 통과 17.5 / 65.6 / 98.1, 0.72/0.05 통과 92.5 / 79.4 / 100, 0.66/0.11 막힘 서기 95.6 / 0 / 77.5 | cont_fin은 **여유 입력이 아니라 폭**으로 선다. 원인은 추출 분포: 폭 ≤ 0.62는 항상 막힘, 0.66은 67 %, 0.70은 33 % 막힘이라 "좁으면 선다"가 기대값이 큼. 여유 입력은 A′→B′ 확장 때 0 열이라 새로 배워야 함 |

**레시피 수정 (`gast_bprime/margin.py`, `tasks.py`; v2.x 코드 불변, `BPRIME_MARGIN_BALANCE=0`으로 원복):**
- 여유만으로 갈리는 폭 구간 [0.62, 0.74)에서 막힘/통과를 먼저 50/50으로 뽑고 여유를 그 구간 안에서 뽑는다(±4 cm 사각지대 유지, 막다른 길·회전·낭떠러지 타일 제외). 그 밖 폭은 v2 추출 그대로. 검증: 폭 0.62/0.66/0.70/0.73 막힘 비율 1.00/0.67/0.33/0.09 → 0.50, 여유 범위·사각지대 0 확인.
- 통과 가능 통로 정지 벌점 1.5 s / −1 → **1.0 s / −2**(`BPrimeStallPenalty`, 낭떠러지 타일 제외 유지).
- 로컬 smoke(192 env × 5 업데이트) 통과, env.yaml에 `reset_narrow_bprime`·`BPrimeStallPenalty` 반영 확인.
- 사전 검증: A6000에서 cont_fin부터 300 업데이트(bprime_continue). 관문은 §7 + pass_tight 통과 ≥ 90 %(602 − 8 pt 이내)와 막힌 통로 서기 ≥ 90 % **동시**.

**수정 후 사전 검증 결과 (bprime_margin, cont_fin + 300, it600~899, E2 nominal 5가족 평균; margin_t1 = it859 / margin_fin = it899 / cont_fin):**

| 문제 정의 | 결과 | 반영 방향 |
|---|---|---|
| 좁은 통과 통로에서 선다 | 전진 pass_tight **91.3** / 63.8 / 23.8, 스윕 0.66/0.05 통과 86.9 / 63.1 / 20.6, 벽 거리 중앙값 0.17~0.19 → 0.04~0.06 m | 관문 통과(t1). 레시피 fc7d4e9 확정 |
| 막힌 통로 서기 유지 | block_margin 98.1 / 98.1 / 96.9, block_narrow 99.4 / 100 / 97.5, 막다른 길 98.4 / 100 / 98.4(침범 0) | 관문 통과 |
| 경계 셀 0.66/0.08(여유 +0.02), 0.72/0.13(−0.02) | 통과 53.7 / 15.0, 서기 74.4 / 96.3 — 두 모델 사이 진동 | ±BAND(0.04) 사각지대 안이라 학습에서 뽑지 않는 셀: **판정 제외** |
| 후진 통과 | pass_tight 후진 84.4 / 81.2 / 91.2 | 한강 감시 항목(cont_fin은 91 → 학습 가능) |
| 오른쪽 굽은 통로 | bend_right pass_tight 65.6 / 37.5 | 한강 감시 항목 |
| 갭·계단·지형 유지 | 갭 전진 94.4 / 94.9, 후진 92.6 / 94.0, 계단 95.3 통과 / 1.2 넘어짐, 지형 넘어짐 0 — cont_fin과 같은 수준 | 수정으로 악화 없음(602 대비 −1~2 pt는 cont_fin부터) |
| 후진 갭 L14 0.3 m/s 넘어짐 | 14.6 / 16.7 / 16.7(602 8.3) | B′가 아니라 A′(G5 R20) 항목. A′ E2에서 판단 |

**한강 B′ 제출:** 코드 fc7d4e9, `LINEAGE_RECIPE=bprime`, STAGE_A = 선택된 A′ 모델, 131,072 envs × 2,000, E2 `configs/gast_narrow/e2/stage_bprime_hangang.yaml`. 감시: 후진 pass_tight, bend_right, 막힌 통로 서기, 갭·계단 유지.
