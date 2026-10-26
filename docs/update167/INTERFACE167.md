# UPDATE167 지도 제안 API — 5090 통합용

상태: CPU reference 구현 완료, 통합·실센서 검증 전. 기존 actor/OBS1 경로를 자동 전환하지 않는다.

모듈: `gd_lab.gast_nav.free_evidence167`.

| 함수/자료형 | 입력 | 출력 |
|---|---|---|
| `frusta_from_d4xx` | optical-z depth `[N,C,H,W]` m, K `[C,3,3]` 또는 `[N,C,3,3]`, camera→sample-yaw rotation `[N,C,3,3]`, translation `[N,C,3]`, `blocks [R,5]` = camera,u0,v0,u1,v1, 명시적 `volume_valid [N,R]` bool | `FreeFrusta`: origins `[N,R,3]`, inward unit normals `[N,R,6,3]`, offsets `[N,R,6]`, far_m `[N,R]`, volume_valid `[N,R]` |
| `FreeEvidenceHistory.push` | frusta, odom pose `[N,4]` x/y/z/yaw, 그 capture의 관측 support `[N,Y,X]`, time_ms/epoch `[N]` int64 | source 복사본 최대 6개. wide/fine grid별 인스턴스 |
| `.measurement` | 현재 pose/time/epoch, **누적 이동거리 상계** travel_m `[N,K]`(capture newest-first), delay용 linear/yaw speed bound `[N]` | `FreeMeasurement` |
| `limit_by_current_returns` | FreeMeasurement, current ResolvedRays | 현재 wall/self first-hit이 내부에 들어가는 footprint를 해당 표면 앞까지 축소한 새 measurement; 원본 cache 불변 |
| `volume_free_proof` | 현재 support `[N,Y,X]`, FreeMeasurement, `grid`, map config | `VolumeProof.free/age_s/fraction [N,3,Y,X]`. fraction은 진단용, 통과 비율로 쓰면 안 됨 |
| `obstacle_map.ray_evidence` | 기존 입력＋`measurement=...`, `certified=True`, `return_age=True` | state, 빈 sub-bin 진단 tuple, band evidence age. occupied 현재 hit 우선 |
| `ObstacleMap.update` | 기존 입력＋measurement/certified | 기존 MapFrame, 과거 free의 원 age 보존 |
| `near_clearance.build_near` | 기존 입력＋measurement/certified/`return_proof=True` | **(NearFrame, fine_state, support_known, band_age_s)**; state/age `[N,3,25,37]`, support `[N,25,37]` |

`certified=True, measurement=None`이면 current occupied는 보존하고 free 인증은 전부 unknown. 기존 호출에서 새 옵션을 쓰지 않으면 legacy actor/packet 경로가 그대로다. 새 measurement가 주어지면 과거 10 cm sub-bin의 any-ray 통과는 free 판정에 사용하지 않는다. 새 인증은 full micro-box containment만 쓴다. 이 경로는 Torch reference이며 기존 Triton sub-bin kernel로 보내지 않는다.

fine measurement에 넣는 support는 crop 전 확장 fine grid(기존 `extended_near_grid()`의 `[N,45,57]`), wide는 `[N,43,49]`다. 두 cache는 같은 센서 frusta를 공유할 수 있으나 support와 grid가 다르다. supervisor는 `fine_state[:,:2]`/`band_age[:,:2]`와 support로 인증하고 **occupied clearance veto를 별도로 유지**해야 한다. actor의 NearFrame unknown은 여전히 all-band OR다.

중요: `volume_valid`는 depth 유효성 bit가 아니다. 픽셀 footprint 전체에 대해 임시 5 cm 기둥 검출·first-hit 최소 거리 계약을 충족한 센서 보정/시뮬레이션 측정 생산자가 제공한다. 빈 depth·중심 ray·GT geometry로 임의 생성하면 안 된다. builder는 블록의 모든 픽셀이 유효할 때 최소 optical depth를 사용하고, self 포함 첫 표면 전에서 멈춘다. 중심 ray의 depth를 이웃에 퍼뜨리지 않는다. no-return/IR 실패도 현재 builder에서는 free를 만들지 않는다.

초기 설정은 `configs/gast_nav/free_evidence167.yaml`. 역사 증거는 물체 속도 상계가 주어지지 않으면 재사용하지 않는다(`historical_obstacle_speed_bound_mps: null`). 명시적으로 상계를 정한 경우 모든 frustum 면을 `v_bound × ceil_age`만큼 안쪽으로 줄이며, 무생성/유계 속도 가정이 깨지면 인증하지 않는다. 원 TTL=0.5 s, OBS1 3-frame/4-bit layout 유지. pose P2 3σ는 확률적 작업 가정이며 절대 오차 보장이 아니다.

`volume_free_proof`는 저수준 기하 검사다. 현재 raw ray와 융합할 때는 `ray_evidence`/`build_near`를 사용한다. 이 두 함수가 `limit_by_current_returns`를 먼저 적용해 최신 occupied/self 증거와 모순되는 과거/생산자 footprint의 뒤쪽을 자른다. 직접 기하 함수를 쓴다면 동일한 전처리가 필요하다. 이 보수적인 reference 전처리는 footprint×current-return 쌍을 검사하므로 실센서 규모의 성능 검증이 필요하다.
