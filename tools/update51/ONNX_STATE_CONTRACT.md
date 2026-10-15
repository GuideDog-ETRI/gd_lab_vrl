# UPDATE51 student v3 two-graph ABI

UPDATE49의 explicit state/host chronology 계약을 계승한다. 사용자 정정에 따라 **기존 4 view × depth/IR 동기 묶음**을 유지한다. 두 채널은 두 카메라가 아니다. 각 view를 비동기로 독립 갱신하는 모델로 바꾸지 않았다. `camera_contract`는 원본 checkpoint의 vendor_new calibration/order/normalization을 그대로 저장한다.

모든 float tensor는 float32, mask는 bool. B만 dynamic. `H=256(small)/896(large)`. 추론은 encode 도착 이벤트와 tick 10ms 두 graph이며 actor는 기존 고정 graph다.

| encode input | shape | 의미 |
|---|---|---|
| frames | B,4,2,45,80 | 원본 정규화 depth/IR |
| relative_pose | B,7 | 이전 accepted capture의 yaw frame에서 현재 capture dx,dy,dyaw + 같은 frame의 현재 wxyz quaternion |
| packet_age | B,1 | (delivery tick − capture tick) × .01초 |
| accepted | B | host 검사를 통과한 새 묶음 |
| reset | B | 이 epoch의 첫 encode, old visual 제거 |
| h_visual | B,6116 | 187×32 cell + 4×32 temporal slots + 4 bookkeeping |

| encode output | shape | 의미 |
|---|---|---|
| h_visual_out | B,6116 | accepted 행만 갱신. reset 후 미수신은 0, 다른 행 exact hold |
| raw_capture | B,32 | 새 auxiliary capture head. 훈련 raw(capture) loss 전용, actor에 사용 금지 |
| spatial_capture | B,187,6 | capture 공간 loss 전용. accepted 행만 소비 |
| quality | B,1 | 원본 sigmoid quality × max(0,1−packet_age/.3). accepted 행만 host에 저장 |

원본 visual CNN/geometry/spatial attention/quality branch를 고정한다. cell GRU/temporal attention/age embedding/spatial head는 34624에서 복사하고 학습 가능하다. `h_visual` 마지막 4개는 좌표 0,0,0 및 valid: 절대 pose는 graph 입력/hidden에 저장하지 않는다. host가 마지막 accepted pose를 소유하며 다음 상대 odometry를 계산한다. 첫 accepted packet은 자기 pose를 이전 pose로 삼아 translation/yaw delta=0, tilt는 유지한다.

| tick input | shape | 의미 |
|---|---|---|
| proprio | B,230 | **raw term-major** policy history: 원본 observation term별 oldest→newest 5 history. 정규화/단위는 원본 observation group 그대로. teacher normalizer/CENet 출력 아님 |
| previous_action | B,12 | 직전 wrapper clipping 후 실제 실행 action, reset 때 0 |
| dt | B,1 | .01초 정상 주기, 첫 tick .01 |
| camera_age | B,2 | capture age, delivery age 초; [0,1] clamp, 미수신 [1,1] |
| camera_fresh | B | 같은 epoch accepted 이력 및 0≤capture age<.3 |
| relative_motion | B,3 | 직전 tick yaw frame의 dx,dy,dyaw; 첫 tick 0 |
| reset | B | old belief 및 previous_action 제거. visual은 앞 encode/host reset 완료 상태여야 함 |
| h_visual | B,6116 | 최종 encode 결과 |
| h_belief | B,H | 직전 tick belief |

출력 `latent_raw [B,32]`, `h_belief_out [B,H]`. latent는 tanh이며 gate 적용 전이다. stale visual은 projection 이후에도 zero masking하여 bias 누출을 막는다. stale일 때도 몸 상태와 belief는 매 tick 갱신한다. reset tick에 새 epoch packet이 있으면 사용할 수 있다.

## Host 순서와 gate

1. reset/done/재연결: affected rows hidden, previous_action, quality, ready, stamps 제거; epoch 증가. 기존 epoch packet은 논리적으로 flush(수신 시 폐기). 실제 deployment queue도 해당 epoch를 flush한다.
2. 같은 monotonic integer clock으로 capture≤delivery≤now, capture≥epoch start, capture ID>last accepted ID, delivery≥last accepted delivery 검사. finite/shape 검사. 미래 delivery는 queue에 보류; negative delay, duplicate/reorder/old epoch/invalid는 hidden/stamps를 변경하지 않는다.
3. delivery 순서/sequence tie-break대로 encode 반복. quality 저장은 accepted 행에만. 두 번째 encode에서 reset 반복 금지. 미수신 reset은 host zero 또는 accepted=false reset=true encode로 처리.
4. tick당 한 번 belief 갱신. 정확히 30 ticks(300ms)는 stale. clock 역전/duplicate tick은 `BeliefBank`가 예외로 중단한다. 역전 후 새 epoch와 queue/clock 재구성 전 action 실행 금지.
5. `effective_gate = stored_quality * ready * (0 <= now-capture < 30 ticks)`. quality에는 delivery 시 capture-age decay가 이미 포함되어 있다. 실행 시 decay를 다시 곱하지 않는다.
6. `actor(concat(base203, latent_raw * effective_gate)) → action12`. teacher execution target도 같은 detached gate와 base203. actor 내부 또는 raw head에서 gate를 다시 적용하지 않는다.

GT 거리·phase·terrain family·force·teacher target은 label/sampling에만 존재한다. `relative_pose`/`relative_motion`은 로봇에서 실제 odometry/IMU로 산출해야 한다. 현재 runner의 simulator root pose 변환은 그 센서 품질을 검증한 것이 아니다.

## Pilot 변경 필요 목록 (배포 리포 미수정)

- 두 ORT session 로드, size별 hidden shape/manifest/hash 검증, state 재귀 전달.
- 4-view 묶음 encode 이벤트와 100Hz tick 스케줄 분리, epoch/수신 순서/미래 queue/중복 tick 처리.
- raw term-major 230 history를 원본 term/scale 순서대로 공급. 기존 cenet_obs가 time-major이면 명시적으로 재배열(teacher normalizer 이중 적용 금지).
- capture→capture 및 tick→tick 상대 odometry/quaternion 계산; reset 및 wrap-around 처리.
- clipping 후 실제 action feedback, reset first action=0.
- 기존 단일 packet quality/ready/age gate 1회 및 고정 actor 연결.
- encode/tick/actor/host 전 구간 p50/p95/p99/max, 10ms deadline miss, drop/reorder/stale/reconnect 측정.
- 모든 incoming frame이 같은 4-view capture 묶음인지 확인. 물리 센서가 독립 비동기면 별도 조립 계약 검토가 필요하며 임의 camera 평균 gate로 대체하지 않는다.
