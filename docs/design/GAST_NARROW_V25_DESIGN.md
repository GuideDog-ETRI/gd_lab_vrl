# GAST-Narrow v2.5: 낭떠러지 정지 + 회전 성공 기준 재정의 (설계서)

2026-10-31 · B.S.Seo (ETRI) · 출발 모델: v24h Top-1 it653 (`narrow_v24h_602_5120x100_800_20261031/best_top5/653_top1.pt`, `7c359da9…`) · 정지 anchor 목표: 602 (`69e7e065…`)

v2.4 설계(`GAST_NARROW_V24_TURN_DESIGN.md`) 위에 얹는 변경만 적는다. v2.4 §15(v24h E2)와 §16(v2.4e 7차 런 실패)이 출발점이다.

## 1. 문제 정의

| 문제 정의 | 현재 (v24h 653, E2 10-31) | 목표 (B단계 판정) |
|---|---|---|
| 조이스틱으로 밀어도 **건널 수 없는 갭(낭떠러지) 앞에서는 서야 한다**. 앞·뒤·좌·우 모든 방향 | 갭은 "건너는 것"만 배웠다. 건너편이 없는 끝선을 본 적이 없다 | 네 방향 모두 끝선 1 m 안에서 멈춰 1.5 s 서기 ≥ 90 %, 발이 끝선을 넘는 비율 ≤ 5 % |
| 좁은 곳 제자리 회전의 성공 기준이 너무 엄격해 학습이 "돌다가 멈추는" 쪽으로 기운다 | 성공 = 도달 + 몸 중심 이동 ≤ 10 cm → 5~22 %. 실패의 절반은 "도달했으나 이동 초과", stall 13~23 % | 성공 = 도달 + 넘어지지 않음 + 몸통 접촉 없음 + 다리 접촉 충격 상한 이내 → ≥ 90 %. 10 cm 이내는 "깨끗한 회전"으로 따로 집계 |
| 막다른 길 서기 회귀 (602 100 % → 44~78 %) | 앞을 18~27° 든 채 흔들려 순간 속도 판정(1.5 s 연속)을 못 넘김 | v2.4e ①(필터 판정)·②(앞 들림 벌점)로 회복. ③(막힌 구역 anchor 확장)은 **끈다** |
| 좁은 통로(전·후진, 막히면 서기), 갭·계단 | 602 수준 유지 중 | 그대로 유지 (판정 불변) |

## 2. v2.4e 7차 런(v24i)에서 배운 것

| 변경 | 결과 | v2.5 반영 |
|---|---|---|
| ③ 정지 anchor를 막힌 구역(플래그 0) 표본까지 확장 | 통로 통과 0.99 → 0.37~0.47, 막힌 통로 서기 0.98 → 0.62~0.75가 50 it 안에 떨어져 it175까지 정체. 막힌 통로와 지나갈 수 있는 통로는 **여유 입력 m만 다른 같은 관측**이라 "서라"는 지도 신호가 번짐 | **제거** (`NARROW_STAND_ANCHOR_ZONE=1`일 때만 켜짐, 기본 꺼짐). 교훈: 지도 신호(회귀)는 **actor가 보는 명시적 입력(플래그)이 1인 표본에만**. 구역 소속처럼 actor가 못 보는 조건으로 당기지 않는다 |
| ① 막힌 구역 hold 판정 필터 속도 | 부작용 없음 | 유지 |
| ② stand_pitch 10° (should_stand 구간만) | 보상 합계 0 (거의 발동 안 함: 구간이 좁고 653은 거기서 표본이 적음). v24는 막다른 길에서 18~27°, 602도 13.6°를 든다 | **`narrow_nose_up`으로 확대** (사용자 지시 10-31: "몸이 들리는 것도 잡아야 해"): 좁은 공간 타일 전부에서 nose-up 초과분에 −3 /s, 문턱은 **서야 하는 구간(should_stand: 회전 불가·막힌 구역·막다른 길·낭떠러지 구역)에서 로봇이 실제로 느릴 때(필터 속도 < 0.3 m/s) 10°, 그 밖(통로 보행·회전 가능 타일·구역 안 접근 중) 20°**. 막힌 구역(zone)은 선 앞 1.2 m 전체라 접근 보행의 일시 피크까지 10°로 벌하면 접근을 주저하게 만들 수 있어 속도 조건을 둔다(v24i의 stand_pitch 합계는 ≈ 0이라 그 런의 하락 원인은 아님). 근거(E2 v24h 행 분석, A6000 10-31): 막다른 길 nose-up은 센서 조건과 무관(±1°)하므로 센서 시야 행동이 아니라 자세 습관이고, 실패를 가르는 것은 최대 각도(held·not held 모두 ≈21°)가 아니라 **10° 넘게 든 채 꿈틀거리는 시간**(held 4~7 s vs not held 12~17 s). 통로 전진 보행에서는 602도 최대 23~28°(p90 32~37°)를 1~2 s 들므로 10° 문턱을 보행에 걸면 일반 보행을 건드린다 → 보행 구간은 20°. 회전 타일은 5° 미만이라 영향 없음. 갭 타일의 `gap_nose_up`(5°, −30)·`gap_nose_peak`은 그대로. 온라인 행에 `nose_up_max`·`nose_up_t10` 기록, 지표 `stand_nose_up_t10_mean`(서기 행, 10° 초과 시간) 관문 ≤ 6 s |
| ④ dead_end 서기 관문 0.85 | 지표가 제대로 잡힘 (0.20 → 0.54) | 유지 + 낭떠러지 관문 추가 |
| ⑤ TurnStall 1.0 s / −2 | 판단 불가 (회전 성공 기준이 stall을 유발) | 유지 |
| ⑥ 경계 ±4~10 cm 여유 추출 40 % | 판단 불가 | 유지 |
| ⑦ turn_drift −6 | 성공 기준을 바꾸면 과함 | **−3**으로 복귀 (v2.4b 값) |

## 3. 낭떠러지 타일 (`narrow_cliff`, kind `cliff` = 6)

### 3.1 지형

| 항목 | 값 | 근거 |
|---|---|---|
| 모양 | 타일(8 × 8 m) 중앙에 정사각 단(platform) 2L × 2L, 윗면 z = +1.0 m. 단 밖은 타일 바닥(z = 0) | 지하철 승강장 단차 ≈ 1.1 m. 단 끝 → 이웃 타일 거리 = 4 − L ≥ 1.8 m (학습한 갭 최대 0.5 m의 3배 이상) |
| 반폭 L | 레벨 0: **2.2 m** → 레벨 14: **1.6 m** | 시작 여유 e₀ = L − 0.40 ± 0.3 ≥ 0.9 m > 구역 0.6 m: 어느 레벨에서도 "접근 → 정지" 구간이 있다(검토 A.6). 작을수록 반응 시간이 준다 |
| 열 수 | **3열** | tight 3열과 같은 비중(env의 4.6 %) |
| env 원점 | 단 윗면 중앙 (z = 1.0). 생성기는 sub-terrain이 돌려준 origin z를 그대로 env 원점에 쓴다(검토 A.1: IsaacLab terrain_generator `_add_sub_terrain`은 xy만 더함) | |
| 총 열 | 62 + 3 = **65** (`NUM_COLS` assert 65) | 유지용 타일 비율 50 % → 48 % |

### 3.2 에피소드

| 항목 | 값 |
|---|---|
| 시작 | 단 중앙 ±0.3 m, yaw 무작위(0~360°). `reset_cliff`는 `reset_narrow_v2`·`reset_turn` **뒤**에 등록(v2가 먼저 x = −3.0~−2.2에 놓으므로 덮어써야 함) |
| 명령 | 몸 기준 방향 d ∈ {앞 40 %, 뒤 20 %, 왼쪽 20 %, 오른쪽 20 %}, 속도 앞·뒤 0.3~0.6 m/s, 옆 0.2~0.3 m/s, ωz = 0. 에피소드 내 일정(통로와 같은 갱신 잡음). `_apply_cliff`는 v2 `_apply_narrow` 뒤에 세 성분을 모두 덮어쓴다(`_apply_turn`과 같은 위치). 명령은 끝까지 유지 |
| 끝선 여유 e | e = L − max(\|px\|, \|py\|), 점 = 트렁크 6점(±0.40, ±0.10) + 발 4점(`body_pos_w`), 타일 좌표 |
| 구역(zone) | **e < 0.6 m** (센서가 끝선을 보는 거리에 맞춤, §3.3). 구역 안: 선속도 추종 OFF, `cliff_stand` ON, should_stand |
| 서기 성공(hold) | 구역 안 + 침범 없음 + 필터 속도 \|v_f\| < 0.10, \|ωz_f\| < 0.15 가 1.5 s 연속 → `update_cliff`가 monitor.held / hold_event / success를 **직접** 세움(v2 monitor는 rects 없는 타일에서 아무것도 안 세움, 검토 B.1) → hold 보너스 +4, 0.5 s 뒤 종료 |
| 침범(breach) | 발 4점 중 하나라도 끝선 밖 > 2 cm, 또는 트렁크 점이 끝선 밖 → monitor.breached (−4) + 종료(`narrow_breach_done_no_turn` 경로) |
| 추락 | 몸통 z < 원점 z − 0.3 m → 종료 `cliff_fall`; 온라인 행의 `base_contact` 필드에 OR(B 성분과 커리큘럼 하강에 반영) |
| 커리큘럼 | 회전 타일처럼 **자체 규칙**: held → 상승, breach·추락 → 하강, `cmd_moving_time` 조건 없음(v2 규칙은 2 s 조건이라 2.0 s에 끝나는 성공 에피소드가 무시됨, 검토 A.6) |

### 3.3 관측 (검토 A.2 반영)

새 입력은 없다. 그러나 **LiDAR 지도는 낙차를 보지 못한다**: 지도는 발 기준 높이 0.08~1.0 m 띠의 반환만 1로 표시하므로(`lidar.py BAND`) 단 위와 허공이 모두 "빈칸"이다. 낙차를 보는 입력은 둘이다.
- **높이 스캔**(1.6 × 1.0 m, 0.1 m 격자, 몸 중심 기준 x ±0.8, y ±0.5): −1.0 m 낙차는 포화값(1.0)으로 들어와 평지(0.05)·갭(0.70)과 구분된다. 발끝 기준 앞·뒤 **0.35 m**, 옆 **0.25 m**까지 본다.
- **하방 Depth+IR**(11 × 17, 같은 격자, 4.8 s 기억): 앞·뒤 카메라만 있어 옆은 카메라 시야에 드는 칸만 보인다.

따라서 로봇은 끝선을 **발끝 0.25~0.35 m 앞에서야** 인지하고, 구역(0.6 m)과 보상은 거기에 맞췄다. 앞·뒤 0.3~0.6 m/s에서 0.6~1.2 s, 옆 0.2~0.3 m/s에서 0.8~1.2 s의 반응 시간이다. 옆 방향은 가장 약하므로 E2에서 방향별로 따로 본다(관문 §9). **보류한 대안**: LiDAR 지도에 "깊은 낙차" 표시(예: 셀 값 0.5)를 넣으면 끝선을 1 m 밖에서 볼 수 있지만 653이 본 적 없는 입력이 갭 타일(깊이 0.65)에도 생겨 갭 회귀 위험이 있고, 실기 지도 생성기도 같은 규칙이 필요하다. v2.5 결과에서 옆·뒤 정지가 약하면 v2.6에서 다룬다.

후방 Mid-360은 좁은 공간 타일에서 70 % 켜진다(`REAR_NARROW_P`)는 점만 적어 둔다(낭떠러지 인지에는 무관).

critic: `narrow_privileged_v25`가 낭떠러지 타일에서 슬롯 7 = 1(서야 함), 슬롯 8 = e / 2(끝선 여유)로 **명시적으로 덮어쓴다**(v2 기본값은 blocked = 0, 검토 B.1). 슬롯 수 불변.

### 3.4 보상·종료 (낭떠러지 타일에서 활성인 항목)

| 항목 | 값 | 비고 |
|---|---|---|
| `cliff_stand` | +1.0 /s × exp(−(v/0.1)² − (ωz/0.2)²), 구역 안 | `turn_stand`와 같은 식 |
| `narrow_hold` | +4 (held 이벤트) | v2 공유 |
| `narrow_breach` | −4 (breach 이벤트) + 종료 | v2 공유 |
| `narrow_nose_up` | −3 /s × (nose-up − 10°)⁺, 구역 안·느릴 때 (그 밖 20°) | §2 |
| `track_lin_vel_xy` | 구역 밖 ON, **구역 안 OFF** (v25 래퍼: `_not_in_zone` × ~cliff_zone) | 충돌 방지 (§5) |
| `track_ang_vel_z` | ωz 명령 0 → ωz ≈ 0에 +0.8 /s, 서기·보행 무관 (중립) | |
| `base_height` | **v25 래퍼**: 낭떠러지 타일에서는 기준 높이를 단 위 광선만으로(ray_z를 원점 z − 0.3 이상으로 클램프) | 검토 A.3: 기본 식은 모든 광선 평균이라 끝선에서 20~50 %가 허공이면 기준이 0.2~0.5 m 내려가 −0.6~−3.8 /s → `cliff_stand`보다 커서 웅크리거나 물러남 |
| `feet_cadence_overrun` | 구역 안 OFF | should_stand |
| `stand_still_feet_contact` | 구역 안 ON (제자리 걸음 벌점) | should_stand |
| `narrow_stall` (v2.2) | **OFF** (v25 래퍼가 kind == cliff 제외) | 검토 A.5 확인: 안 끄면 1.5 s 뒤 −1/s |
| `narrow_centering` | OFF (래퍼) | 경로 개념 없음 |
| `narrow_stand`(v2 구역 서기 +0.5) | 0 (v2 zone 아님) | |
| `narrow_wall_proximity`, `narrow_leg_touch` | 벽 사각형 없음 → 0 | |
| `narrow_overspeed` | ON (명령 + 0.15 초과) | |
| `narrow_success_done` | held 0.5 s 뒤 종료 | v2 공유 |
| 6566 유지 anchor | 낭떠러지 행은 narrow 플래그 1 → 제외 | 의도대로 |

## 4. 회전 성공 기준 재정의

| 항목 | v2.4 | v2.5 |
|---|---|---|
| 성공(pass) | 도달 + drift_max ≤ 0.10 | **도달 + drift_max ≤ 0.30 + 몸통 접촉 없음 + 다리 접촉 충격량 < 30 N·s** (처음 안은 최대력 150 N이었으나 로컬 파일럿에서 벽 접촉의 PhysX 최대력이 0.1 s 미만 접촉에서도 1~7 kN으로 튀어 "접촉 금지"와 같아짐 → 충격량 Σ\|F\|·dt로 변경) |
| 깨끗한 회전(clean) | = 성공 | 성공 + drift_max ≤ 0.10 + 벽 접촉 시간 < 0.1 s. **보조 지표**(`turn_clean_rate`), 점수에는 안 들어감 |
| 종료 | drift > 0.20, 또는 도달 후 drift_max > 0.10 (실패) | drift > 0.30만. "도달 후 이동 초과" 종료 **삭제**. critic 슬롯 8의 drift 정규화 분모는 0.20 유지 |
| `turn_drift` | −6 /s (v2.4e) | **−3 /s** (0.05 초과분 비례) |
| `turn_stall` | 1.0 s / −2 | 유지 |
| 회전 가능 판정 | r_free ≥ 0.46 + m | 유지 (플래그 규칙은 배포 규약) |
| 다리 접촉 충격량 | 없음 | 접촉 센서(thigh/calf) 합력 크기 × dt의 에피소드 합(N·s). 30 N·s ≈ 체중(440 N)으로 0.07 s, 또는 50 N으로 0.6 s 스침. 최대력(`turn_leg_force`)과 충격량(`turn_leg_impulse`) 모두 온라인 행에 기록, p90을 지표로 두고 런 중 보정 |

이유: v24h E2에서 "도달했으나 이동 초과" 23~33 %와 "0.2 m 종료" 29~41 %가 실패의 대부분이고, 기본 보행이 180° 회전에서 15~19 cm 밀린다. 이동 초과가 즉시 실패·종료이므로 정책은 **돌지 않고 서는 쪽(stall)이 기대 벌점이 작다**. 성공을 "안 넘어지고 세게 안 부딪치고 다 돈다"로 두면 pass 보너스 +8이 회전을 끝까지 끌고 가고, drift −3과 접촉 벌점이 깨끗함을 점진적으로 올린다.

## 5. 보상 충돌 점검 (상황별로 동시에 켜지는 항목)

규칙: 같은 상황에서 **반대 방향으로 비슷한 크기**를 내는 항목이 있으면 하나를 끄거나 조건을 나눈다. 크기는 1초당, 이벤트는 1회.

### 5.1 낭떠러지 구역 (새로 생기는 상황)

| 항목 | 방향 | 크기 | 판정 |
|---|---|---|---|
| `cliff_stand` | 서라 | +1.0 /s | 주 신호 |
| `narrow_hold` | 서라 | +4 (1.5 s 뒤) | |
| `track_lin_vel_xy` | 가라 | +1.0 /s | **충돌 → 구역 안 OFF** (v2 막힌 구역과 같은 처리) |
| `narrow_stall` | 가라 (서면 −1/s) | −1.0 /s | **충돌 → 낭떠러지 타일 제외**. 지금 코드는 `passable = narrow & ~blocked`라 cliff에 blocked = False를 두면 켜진다 |
| `stand_still_feet_contact` | 발 떼지 마라 | −1.0 /s × 든 발 수 | 서기와 같은 방향 |
| `feet_cadence_overrun` | 걸어라 | −1.0 /s | **충돌 → OFF** (should_stand에 포함) |
| `stand_still` (관절 편차) | 명령 < 0.1일 때만 | 0 | 명령이 있으므로 비활성 |
| `narrow_nose_up` | 수평 자세 | −3 /s × (nose-up − 10°)⁺ | 서기와 양립. 602의 13.6° 자세는 −0.19 /s로 가볍게 벌하고 v24의 18~27°는 −0.4~−0.9 /s. 꿈틀거림 자체는 ①(필터 판정)이 성공으로 잡아 주고, 이 항은 자세를 내린다 |
| `base_height` | 높이 0.52 유지 | −15 × 오차² | **충돌 → 래퍼로 해결**(§3.4). 갭 타일은 `boarding_family_mask` 분기로 보호돼 있었고 다른 타일은 전체 광선 평균을 쓴다 |
| `platform_gap_*`, `gap_nose_up` | 갭 타일 전용 마스크 | 0 | 확인됨: `terrain_family_gate("platform_gap")`는 패밀리 이름으로 잡는다 |
| `termination_penalty` | 추락·침범 | −2 (1회) | breach −4와 합쳐 −6: hold +4 + stand ≈ +6 보다 작지 않도록 → 충분 |

### 5.2 회전 가능 타일 (기준 변경)

| 항목 | 방향 | 크기 | 판정 |
|---|---|---|---|
| `track_ang_vel_z` | 돌아라 | +0.8 /s | 주 신호 |
| `narrow_success` (pass) | 끝까지 돌아라 | +8 (1회) | v2.5: drift ≤ 0.30이면 지급 |
| `turn_stall` | 멈추지 마라 | −2 /s (1 s 뒤) | |
| `turn_drift` | 제자리에 | −3 /s × (drift − 0.05) | 0.15 m에서 −0.3 /s: 추종 +0.8보다 작음 → 돌면서 조금 밀리는 쪽이 유리. **stall 유인 없음** |
| `undesired_contacts` | 닿지 마라 | −1 /s | 다리 접촉 시 |
| `narrow_leg_touch` | 닿지 마라 | −1 /s × 다리 수 | 두 다리 스치면 −2 /s: 추종 +0.8 + pass를 넘어서지 않음(6 s 회전에서 최악 −12 vs pass +8 + 추종 +4.8). 경계에 가까움 → **−0.5 /s로 완화** |
| `narrow_wall_proximity` | 여유 m 안으로 들어오지 마라 | −1 /s × 침입 합(점 ~22개) | 경계 근처 셀(slack +4~10 cm)에서는 수 /s까지 커질 수 있어 회전을 막는 마지막 반대 압력 → **회전 타일에서는 점 합을 2.0으로 캡** |
| `turn_overspin`, `turn_wrong_spin` | 과속·역회전 | −1 /s | 변동 없음 |
| 종료 `turn_drift_done` | drift > 0.30 | 에피소드 종료 | "도달 후 실패 종료" 삭제로 **도달 = 항상 보너스** |

### 5.3 회전 불가 타일, 막힌 통로, 막다른 길

v2.4d와 같다(검증된 조합). 바뀌는 것은 `narrow_nose_up`(추가)과 anchor 확장 제거뿐. 회전 가능 타일과 통로에서는 `narrow_nose_up` 문턱이 20°라 평소 보행(1~2 s의 23~28° 피크만 걸림)에는 거의 영향이 없고, 회전 타일(5° 미만)에는 없다. `turn_stand`는 플래그 1에만 지급된다(v2.4e 조사에서 확인).

### 5.4 지나갈 수 있는 통로 (바뀌면 안 되는 상황)

v2.5에서 **어떤 항목도 바뀌지 않는다**. anchor 확장 제거로 v24i의 통과 하락 요인이 사라진다. 관문: 온라인 `narrow_passage_rate` ≥ 0.85 (it50부터, v24h는 0.83~0.96).

## 6. 온라인 Top-5

점수 = 0.20 P + 0.15 H + 0.10 N + 0.05 L + 0.05 W + 0.15 G + 0.10 B + 0.05 T + 0.10 R + **0.05 C**
- C = 낭떠러지 서기율, (레벨) 층 macro. W 0.10 → 0.05.
- P/H/N/L/W 는 `narrow_cliff` 행을 **제외**하고 계산(통로 지표를 섞지 않음).
- R(회전 성공)은 새 pass 정의를 따른다. 보조: `turn_clean_rate`, `turn_leg_force_p90`.
- 관문 추가: `min_cliff_hold_rate` 0.85, `max_cliff_breach_rate` 0.05. 기존 `min_dead_end_hold_rate` 0.85 유지.
- `top5_start_iteration` 50 유지.
- `narrow_top5_v25`는 자체 collector를 설치해 행에 `turn_clean`, `turn_leg_force`, `nose_up_max`, `nose_up_t10`, cliff `fell`, `e_min`을 넣고, `install()`에서 새 관문 키를 `DEFAULT_TOP5_CRITERIA`에 setdefault 한다(reloader가 모르는 키를 거부함).

## 7. E2

- `turn` 묶음: 성공 = 새 정의, "깨끗한 회전" 열 추가, "도달·이동 초과" 열 삭제, "다리 최대력 p90" 열 추가.
- 새 묶음 `cliff`: 레벨 {3, 8, 13} × 방향 {앞, 뒤, 왼쪽, 오른쪽} × 속도 0.5(옆 0.3) × 32 env. 지표: 서기(hold), 침범(breach), 추락, 끝선까지 최소 여유 e_min. 세 센서 조건(nominal, mid360_rear, mid360_only) 모두.
- 하락 판정에 dead_end 서기와 cliff 서기를 추가한다.
- `run_e2.py`: trial 완료 판정에 suites를 포함(어제 retention 누락 원인).

## 8. 구현 전 확인 항목 — 답 (코드 검토 10-31, 독립 agent)

| # | 질문 | 답 | 반영 |
|---|---|---|---|
| 1 | origin z 전파 | IsaacLab `_add_sub_terrain`은 xy만 더하고 z는 그대로 env 원점으로 감 | 단 윗면 z = 1.0을 origin으로 돌려줌 (§3.1) |
| 2 | 센서가 −1 m 낙차를 보나 | **LiDAR 지도는 못 봄**(높이 띠 0.08~1.0). 높이 스캔은 포화값으로 봄(앞뒤 0.35 / 옆 0.25 m). Depth+IR은 앞뒤 카메라 시야 안만 | 구역 0.6 m, L 2.2→1.6, 옆 속도 0.2~0.3, 방향별 E2 (§3.3). LiDAR 낙차 표시는 v2.6 후보 |
| 3 | base_height 기준 | 갭 외 타일은 전체 광선 평균 → 끝선에서 기준이 내려가 −0.6~−3.8 /s | v25 래퍼 (§3.4) |
| 4 | 갭 마스크 | 패밀리 이름 기반, cliff 무관 | 없음 |
| 5 | NarrowStallPenalty | cliff에서 켜짐(blocked = 0, exit FAR) | v25 래퍼로 제외 |
| 6 | 커리큘럼 | v2 규칙은 `cmd_moving_time > 2 s` 조건이라 짧은 성공이 무시되고, L 1.2에서는 시작부터 구역 안 | cliff 자체 규칙, L_min 1.6 (§3.2) |
| 7 | 다리 접촉력 | `contact_forces` 센서, `.*_thigh`·`.*_calf` 바디(undesired_contacts와 같음), `net_forces_w_history` 최대 | hip 제외(센서 바디에 없음). 150 N은 히스토리 최대 |
| 8 | 653 로드 | 트레이너가 v24 체크포인트를 소스로 허용하고 PPO 옵티마이저를 복원해야 함(현 코드는 `source == TASK_ID`일 때만). `means`에 `narrow_cliff` 0 필요 | `train_narrow_v25.py` |
| 9 | E2 | MODE_CLIFF, spawn 방향·속도, 행 필드(held/breached/fell/e_min), TASKS에 v25, `leg_force_max` 추가(지금 행에는 적분만 있음) | §7 |
| 10 | nose-up 문턱 | 보행 피크 23~28°(1~2 s) → 보행 20°, 정지·느릴 때 10° | §2 |

추가로 나온 것: `geometry.build("cliff")`에 명시적 `block_s = 0`, `width = L`(기본 식은 3.4를 내서 `stop_s`와 critic 슬롯이 오염됨); `narrow_privileged_v24`의 drift 정규화 분모가 DRIFT_FAIL이라 0.20 → 0.30으로 바꾸면 critic 입력 스케일이 바뀜 → **정규화 분모는 0.20 상수(`DRIFT_NORM`)로 유지**; v2 P/H/N/L/W에서 cliff 층 제외는 필수(아니면 passage = 0 행이 P를 끌어내림); cliff 추락은 base_contact가 아니므로 행에 OR.

## 9. 학습 계획

| 단계 | 내용 | 관문 |
|---|---|---|
| 로컬 파일럿 | 1 GPU, 1024 env × 6 update: 오류 없음, cliff 행·지표 출력 | |
| a6000 8차 (`narrow_v24j`) | 653 시작, 5120 × 800, GPU0-2, anchor 목표 602 | it50: 통로 통과 ≥ 0.85, 막힌 통로 서기 ≥ 0.85, dead_end ≥ 0.7(상승 중), 회전 정지 ≥ 0.6, cliff 서기 > 0 (상승 중). it100: dead_end ≥ 0.85, cliff ≥ 0.5, 회전 성공(새 정의) ≥ 0.7. 어느 하나라도 it100에 못 미치면 중단하고 원인 분석 |
| E2 | Top-5 + 602 + 653, 세 센서 조건, cliff는 방향별(앞·뒤·좌·우) 표 | §1 목표 열. 옆 방향 서기 < 70 %면 v2.6(LiDAR 낙차 표시) 착수 |
| 한강 B | STAGE_A 모델 → v2.5 레시피, 1,200 update, 재제출 완주 | 별도 지시서 |

## 10. 파일

| 파일 | 변경 |
|---|---|
| `geometry.py` | KINDS `cliff` 6, `build("cliff", L)`: rects 없음, path `[(-0.5,0),(0.5,0)]`, exit FAR, `half` 필드 |
| `terrain.py` | cliff 메시(단 + 바닥), origin z |
| `mdp_cliff.py` (새) | `CliffEpisode`, `reset_cliff`, 명령 mixin `_apply_cliff`, `update_cliff`(monitor held/breached 직접 세움), `cliff_stand`, `cliff_zone`, `cliff_fall_done`, `base_height_cliff`, `narrow_privileged_v25`, 자체 커리큘럼, 집계 필드 |
| `mdp_turn.py` | 성공 정의(§4), `clean`, `peak_force`, `turn_drift_done` 단순화, `should_stand`에 cliff 구역 포함, `narrow_centering_no_turn` → cliff 포함, `stand_pitch` → `narrow_nose_up`(좁은 공간 타일 전부, 10°) |
| `tasks_v25.py` (새) | v2.4 Teacher/Runner 상속: cliff 패밀리 3열(65열), 보상 가중(§3.4, §4, §5), stall 래퍼, tracking 래퍼 |
| `anchor.py` | ③ 기본 꺼짐 (완료) |
| `narrow_top5_v25.py` (새) | §6 |
| `train_narrow_v25.py`, `train_narrow_v25_3gpu.sh` | 과제 ID v25; 소스로 v24(653)·v25·v23(602)·backgap 허용, v24/v25 소스는 PPO 옵티마이저 복원; `means`에 `narrow_cliff` 0 |
| `e2/*` | §7 |
| `configs/gast_narrow/online_top5_v25.json`, `e2/v25_*.yaml` | 관문·E2 설정 |

## 11. 구현 메모 (2026-10-31)

- 구현 파일: `geometry.py`(kind cliff, `CLIFF_H`), `terrain.py`(단 메시, origin z), `mdp_cliff.py`(새), `mdp_turn.py`(§4 성공 정의·`PASS_DRIFT`·`LEG_FORCE_MAX`·`DRIFT_NORM`, `narrow_nose_up`, `should_stand`+cliff, `narrow_wall_proximity_capped`), `tasks_v25.py`(새: `NarrowV25Monitor`, 래퍼들, 65열), `narrow_top5_v25.py`(새), `train_narrow_v25.py`·`train_narrow_v25_3gpu.sh`, `configs/gast_narrow/online_top5_v25.json`, E2 `e2_trial.py`(task v25, MODE_CLIFF, 행 필드)·`run_e2.py`(cliff 묶음, 부분 묶음 디렉터리 `__only-…`)·`summarize_e2.py`(깨끗한 회전·다리 최대력·낭떠러지 표)·`configs/gast_narrow/e2/v25_turn.yaml`.
- 성공 판정의 다리 접촉력은 `contact_forces` 센서의 `.*_thigh|.*_calf` 바디 히스토리 최대(hip은 센서 바디에 없음).
- v2.4e의 `stand_pitch` 항은 v2.5 과제에서 제거(`None`)하고 `narrow_nose_up`으로 대체. v2.4 과제(tasks_v24)는 그대로 둔다.
- 로컬 1-GPU 파일럿(602 시작, 1024 env × 6 update)으로 로더·지형·보상·collector 경로를 확인한다. 653 시작은 a6000 smoke에서 확인(`source_kind narrow_v24`, PPO 옵티마이저 복원).
