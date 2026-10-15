# Pilot 후속 변경 목록 — 설계만, 코드 수정/접속/측정 없음

현재 ORT CPU 단일 thread actor가 2ms 지령 발행 루프 안에서 동기 실행된다.
100Hz decimation만으로 500Hz 발행 지연을 막을 수 없다. 다음은 별도 승인·통합
검증이 필요한 설계이며 이 UPDATE57이 Pilot을 바꾸거나 배포 승인하지 않는다.

1. **500Hz 송신/감시와 100Hz 추론 분리.** 송신 루프는 완성된 immutable
   command snapshot만 읽는다. 단일 policy worker가 proprio snapshot→belief
   tick→gate 1회→고정 actor→finite/range 검사→command publication을 소유한다.
   session.Run, encoder, 이미지 전처리, 디스크 IO, 무제한 queue, blocking lock을
   송신 루프에 넣지 않는다. 단일 ORT session thread 설정도 각각 명시한다.
2. **세대·순서·단일 writer.** vision worker만 visual hidden 및 projection을
   쓰고, policy worker만 belief hidden을 쓴다. publication은 epoch, packet seq,
   capture/arrival/encode-complete 시간, capture pose, quality, unmasked projection을
   하나의 snapshot으로 내보낸다. tick이 처음 읽은 seq만 new_packet=1.
   in-flight1 + 최신 완전 4-view packet1. 대기는 교체/drop하되 완료 결과는 유효하면
   commit하여 starvation을 막는다. 상대 pose는 마지막 commit 기준이다.
3. **reset/model 교체 barrier.** supervisor가 새 epoch를 발행하고 각 owner가
   자기 state를 초기화/ack한다. 이전 epoch의 encode/action 완료를 폐기한다.
   cache ready를 무효화하고 새로운 동일 epoch packet 이후만 복원한다.
   단순 포인터 교체나 timestamp 덮어쓰기로 reset을 대신하지 않는다.
4. **deadline miss arbitration.** command에 decision seq, observation stamp,
   release/start/finish, intended apply time, expiry와 epoch를 붙인다. 완료된
   시각만 최신인 늦은 action을 뒤늦게 무조건 적용하지 않는다. 송신/감시 경로가
   expiry 및 연속 miss를 독립 판정한다. hold 가능 시간과 연속 miss 상한은
   별도 로봇 검증으로 결정하고 기존 supervisor/watchdog 전환에 연결한다.
   기존 Damp/stand semantics를 보존하며 새로운 blind/teacher fallback은 없다.
   2ms publish miss, 10ms inference miss, vision stale는 별도 카운터다.
5. **actual applied action feedback.** tick 입력은 마지막 계산 action이 아닌
   clip/중재/late discard 후 실제 적용한 action과 적용 seq/time에서 만든다.
   적용 telemetry가 늦으면 호스트가 확정 가능한 이력/age를 명시한다. 지연
   pipeline 0/1/2tick 학습과 같은 정의로 closed-loop를 검증한다.
6. **timestamp trace.** 원래 센서 capture, 채널별 receive, complete bundle,
   preprocess 시작/끝, encode enqueue/start/end, snapshot commit, tick 예정 release/
   실제 시작/끝, actor 시작/끝, publish, robot receive/apply를 같은 trace ID로
   연결한다. clock domain 및 offset/skew uncertainty를 보존한다. capture age,
   arrival age, projection age, applied action age를 혼동하지 않는다.
7. **측정 및 판단.** 실제 Pilot PC CPU SKU/ORT/affinity/전력/RT권한과 동시 camera,
   DDS, odometry, logger 부하에서 10만 tick 이상 및 열평형 세션을 별도 승인 후
   측정한다. stage/combined/end-to-end의 p50/p95/p99/p99.9/max와 miss율, 연속 miss,
   최대 stale duration을 기록한다. Z890 Python ORT 참고치는 이 결과를 대체하지
   않는다. 300ms student stale gate와 현 Pilot vision timeout(예:1s)은 다른
   계약이므로 통합 때 의미를 정리해야 한다. gate0은 안전한 보행 보장이 아니다.

Fault 통합 시험 목록: encode50/80/120/250ms, burst/drop/reorder, 하나의 view만
지연, reset 중 encode 완료, clock 역전, 모델 교체, NaN action, 늦은 action,
연속 deadline miss, stale→복구. 500Hz 송신이 추론 때문에 block되지 않는지와
기존 안전 전환을 실제 검증하기 전에는 배포 준비 완료로 표시하지 않는다.
