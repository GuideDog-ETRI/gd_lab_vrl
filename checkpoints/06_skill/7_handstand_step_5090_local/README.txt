물구나무(앞발) + 제자리 앞발 걸음 기술, 블라인드, payload 0, LiDAR 없음 (RTX 5090 로컬, 4,096 env)
선정 모델 : s2c_model_500.pt (S2c 500 iteration)
코드      : experiments/skill_handstand/ (HANDSTAND_VARIANT=s2c), 설계 docs/SKILL_HANDSTAND_WALK_DESIGN.md

계보
  blind DWB 38000 (checkpoints/00_blind) -> S1 v3 model_400 (물구나무 진입/유지/복귀, 시작 자세 + 무릎 하중)
  -> S2b model_650 (걷기 플래그 추가, 관절 쿨롱 마찰 0-1 Nm, 과회전 벌점) -> S2c model_500 (제자리 유지 -3, 이탈 벌점 -2)

정책 입력 추가분 (기존 blind 관측 뒤): skill_command [요청, 중단, 전환 후 경과/3], skill_walk [걷기 플래그]
  모두 0이면 기존 4족 걷기와 같은 입력이다.

MuJoCo 확인 (scripts/mujoco_eval_batch.sh, step 시나리오, seed 10회)
  condim 3 (Isaac과 유사) : 넘어짐 0/10, 앞발 착지 평균 28.1회, 제자리 이탈 평균 0.25 m
  condim 6 (배포 MJCF, 구름 마찰) : 넘어짐 5/10, 착지 12.6회, 이탈 0.20 m
  -> condim 6에서 아직 불안정. 실기 투입 전 발 마찰·구름 마찰 무작위화 재학습이 필요하다.

Pilot 연동 전 필수: 기울기 Damp(RlWalker kTiltTripGravZ, 60도)를 스킬 모드에서만 pitch 0-110도 / roll +-30도로 교체.
확인: sha256sum -c SHA256SUMS
