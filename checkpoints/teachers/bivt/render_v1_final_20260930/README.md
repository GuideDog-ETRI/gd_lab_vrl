# BIVT-Render v1 — completed teacher bundle

- 수신 커밋: `c61cd08fc19d019612f78feccd33c3d16e87c2e0` (origin/main).
- 원격 원본 경로: `checkpoints/arm4_bivt_render_v1_final_20260930/`.
- 로컬 새 경로: `checkpoints/teachers/bivt/render_v1_final_20260930/`.
- 모델: `teacher/model_1299.pt`, checkpoint `iter=1299`.
- 실행명: `arm4_bs_2b_v1_from_trial999`.
- 시작 모델: `2026-09-30_09-18-56_arm4_blindstart_trial1000/model_999.pt`.
- 추가 학습 설정: 300 iterations. 로그 마지막에 `EXIT=0` 확인.
- 최종 저장 모델이며, Top-1 선정 모델이라는 의미는 아니다.

## 파일

```
teacher/model_1299.pt
teacher/params/agent.yaml
teacher/params/env.yaml
source/gd_lab_vrl.diff
training/console.log
```

5개 원본 파일 모두 수신 커밋의 Git blob과 바이트 단위 일치를 확인했다.
모델 SHA256: `5829065829089ed41ce4da8ec37f8c162bce6cb9162e9ec39e854e0cfbe206d6`.
원본 params와 diff의 옛 경로는 당시 기록이므로 수정하지 않았다.

## 새 코드에서 로드

원본 params의 `gd_lab_blind_start.actor_critic:DreamwaqVrlGatedActorCritic`은
새 구조의 `gd_lab.teachers.bivt.actor_critic:DreamwaqVrlGatedActorCritic`에 대응한다.
로더에서 이 경로를 명시적으로 매핑해야 한다. 원본 YAML을 그대로 동적 import하지 않는다.

현재 새 클래스에 strict state load 성공, CPU의 합성 관측 2개에 대해 유한한
12관절 출력 및 전체 지형 비가시 시 latent=0을 확인했다.
이는 보행 성능 평가 또는 전체 블라인드 모델과의 행동 일치 검증은 아니다.
패키지의 diff에는 실행 당시 미추적 `blind_start/` 표시가 있으므로,
strict load 성공만으로 실행 소스 전체의 동일성을 증명하지는 않는다.

수신 커밋은 패키징 커밋이며 실제 학습 소스 커밋을 뜻하지 않는다.
아직 이 모델로 학생 학습·BAVRL 교사 교체·배포 교체를 하지 않았다.
